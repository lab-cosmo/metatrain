from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import rascaline.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    System,
)
from metatensor.torch.learn.nn import Linear as LinearMap
from metatensor.torch.learn.nn import ModuleMap

from metatrain.utils.data.dataset import DatasetInfo

from ...utils.composition import apply_composition_contribution
from ...utils.dtype import dtype_to_str
from ...utils.export import export


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorMap) -> TensorMap:
        return x


class MLPMap(ModuleMap):
    def __init__(self, atomic_types: List[int], hypers: dict) -> None:
        # hardcoded for now, but could be a hyperparameter
        activation_function = torch.nn.SiLU()

        # Build a neural network for each species
        nns_per_species = []
        for _ in atomic_types:
            module_list: List[torch.nn.Module] = []
            for _ in range(hypers["num_hidden_layers"]):
                if len(module_list) == 0:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["input_size"], hypers["num_neurons_per_layer"]
                        )
                    )
                else:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["num_neurons_per_layer"],
                            hypers["num_neurons_per_layer"],
                        )
                    )
                module_list.append(activation_function)

            nns_per_species.append(torch.nn.Sequential(*module_list))
        in_keys = Labels(
            "central_species",
            values=torch.tensor(atomic_types).reshape(-1, 1),
        )
        out_properties = [
            Labels(
                names=["properties"],
                values=torch.arange(
                    hypers["num_neurons_per_layer"],
                ).reshape(-1, 1),
            )
            for _ in range(len(in_keys))
        ]
        super().__init__(in_keys, nns_per_species, out_properties)
        self.activation_function = activation_function


class LayerNormMap(ModuleMap):
    def __init__(self, atomic_types: List[int], n_layer: int) -> None:
        # one layernorm for each species
        layernorm_per_species = []
        for _ in atomic_types:
            layernorm_per_species.append(torch.nn.LayerNorm((n_layer,)))

        in_keys = Labels(
            "central_species",
            values=torch.tensor(atomic_types).reshape(-1, 1),
        )
        out_properties = [
            Labels(
                names=["properties"],
                values=torch.arange(n_layer).reshape(-1, 1),
            )
            for _ in range(len(in_keys))
        ]
        super().__init__(in_keys, layernorm_per_species, out_properties)


class SoapBpnn(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.soap_calculator = rascaline.torch.SoapPowerSpectrum(
            radial_basis={"Gto": {}}, **self.hypers["soap"]
        )

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )
            for key, value in dataset_info.targets.items()
        }

        # the model is always capable of outputting the last layer features
        self.outputs["mtt::aux::last_layer_features"] = ModelOutput(
            unit="unitless", per_atom=True
        )

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(self.outputs)
        self.register_buffer(
            "composition_weights",
            torch.zeros((n_outputs, max(self.atomic_types) + 1)),
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(self.outputs.keys())
        }

        soap_size = (
            (len(self.atomic_types) * (len(self.atomic_types) + 1) // 2)
            * self.hypers["soap"]["max_radial"] ** 2
            * (self.hypers["soap"]["max_angular"] + 1)
        )

        hypers_bpnn = self.hypers["bpnn"]
        hypers_bpnn["input_size"] = soap_size

        if hypers_bpnn["layernorm"]:
            self.layernorm = LayerNormMap(self.atomic_types, soap_size)
        else:
            self.layernorm = Identity()

        self.bpnn = MLPMap(self.atomic_types, hypers_bpnn)

        self.neighbors_species_labels = Labels(
            names=["neighbor_1_type", "neighbor_2_type"],
            values=torch.combinations(
                torch.tensor(self.atomic_types, dtype=torch.int),
                with_replacement=True,
            ),
        )
        self.center_type_labels = Labels(
            names=["center_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )

        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]

        self.last_layer_feature_size = n_inputs_last_layer * len(self.atomic_types)
        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(
                    Labels(
                        "central_species",
                        values=torch.tensor(self.atomic_types).reshape(-1, 1),
                    ),
                    in_features=n_inputs_last_layer,
                    out_features=1,
                    bias=False,
                    out_properties=[
                        Labels(
                            names=["energy"],
                            values=torch.tensor([[0]]),
                        )
                        for _ in self.atomic_types
                    ],
                )
                for output_name in self.outputs.keys()
                if "mtt::aux::" not in output_name
            }
        )

    def restart(self, dataset_info: DatasetInfo) -> "SoapBpnn":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = merged_info.atomic_types - self.dataset_info.atomic_types
        new_targets = merged_info.targets - self.dataset_info.targets

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The SOAP-BPNN model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for output_name in new_targets:
            self.add_output(output_name)

        self.dataset_info = merged_info
        self.atomic_types = sorted(self.dataset_info.atomic_types)

        for target_name, target in new_targets.items():
            self.outputs[target_name] = ModelOutput(
                quantity=target.quantity,
                unit=target.unit,
                per_atom=True,
            )
        self.new_outputs = list(new_targets.keys())

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # initialize the return dictionary
        return_dict: Dict[str, TensorMap] = {}

        soap_features = self.soap_calculator(systems, selected_samples=selected_atoms)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(
            self.neighbors_species_labels.to(device)
        )

        soap_features = self.layernorm(soap_features)

        last_layer_features = self.bpnn(soap_features)

        # output the hidden features, if requested:
        if "mtt::aux::last_layer_features" in outputs:
            last_layer_features_options = outputs["mtt::aux::last_layer_features"]
            out_features = last_layer_features.keys_to_properties(
                self.center_type_labels.to(device)
            )
            if not last_layer_features_options.per_atom:
                out_features = metatensor.torch.sum_over_samples(out_features, ["atom"])
            return_dict["mtt::aux::last_layer_features"] = (
                _remove_center_type_from_properties(out_features)
            )

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution(
                    output_layer(last_layer_features),
                    self.composition_weights[  # type: ignore
                        self.output_to_index[output_name]
                    ],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        for output_name, atomic_energy in atomic_energies.items():
            atomic_energy = atomic_energy.keys_to_samples("center_type")
            if outputs[output_name].per_atom:
                # this operation should just remove the center_type label
                return_dict[output_name] = metatensor.torch.remove_dimension(
                    atomic_energy, axis="samples", name="center_type"
                )
            else:
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    atomic_energy, ["atom", "center_type"]
                )

        return return_dict

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "SoapBpnn":

        # Load the checkpoint
        checkpoint = torch.load(path)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_hypers)
        dtype = next(iter(model_state_dict.values())).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        return model

    def export(self) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for SoapBpnn")

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=self.hypers["soap"]["cutoff"],
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return export(model=self, model_capabilities=capabilities)

    def set_composition_weights(
        self,
        output_name: str,
        input_composition_weights: torch.Tensor,
        atomic_types: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][  # type: ignore
            atomic_types
        ] = input_composition_weights.to(
            dtype=self.composition_weights.dtype,  # type: ignore
            device=self.composition_weights.device,  # type: ignore
        )

    def add_output(self, output_name: str) -> None:
        """Add a new output to the self."""
        # add a new row to the composition weights tensor
        # initialize it with zeros
        self.composition_weights = torch.cat(
            [
                self.composition_weights,  # type: ignore
                torch.zeros(
                    1,
                    self.composition_weights.shape[1],  # type: ignore
                    dtype=self.composition_weights.dtype,  # type: ignore
                    device=self.composition_weights.device,  # type: ignore
                ),
            ]
        )
        self.output_to_index[output_name] = len(self.output_to_index)
        # add a new linear layer to the last layers
        hypers_bpnn = self.hypers["bpnn"]
        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]
        self.last_layers[output_name] = LinearMap(
            Labels(
                "central_species",
                values=torch.tensor(self.atomic_types).reshape(-1, 1),
            ),
            in_features=n_inputs_last_layer,
            out_features=1,
            bias=False,
            out_properties=[
                Labels(
                    names=["energy"],
                    values=torch.tensor([[0]]),
                )
                for _ in self.atomic_types
            ],
        )


def _remove_center_type_from_properties(tensor_map: TensorMap) -> TensorMap:
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        new_blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(
                    names=["properties"],
                    values=torch.arange(
                        block.values.shape[-1], device=block.values.device
                    ).reshape(-1, 1),
                ),
            )
        )
    return TensorMap(keys=tensor_map.keys, blocks=new_blocks)
