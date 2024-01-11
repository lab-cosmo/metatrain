from typing import Dict, List, Optional

import metatensor.torch
import rascaline.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, System
from omegaconf import OmegaConf

from .. import ARCHITECTURE_CONFIG_PATH
from ..utils.composition import apply_composition_contribution


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "soap_bpnn.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]

ARCHITECTURE_NAME = "soap_bpnn"


class MLPMap(torch.nn.Module):
    def __init__(self, all_species: List[int], hypers: dict) -> None:
        super().__init__()

        activation_function_name = hypers["activation_function"]
        if activation_function_name == "SiLU":
            self.activation_function = torch.nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_function_name}"
            )

        # Build a neural network for each species
        nns_per_species = []
        for _ in all_species:
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
                module_list.append(self.activation_function)

            nns_per_species.append(torch.nn.Sequential(*module_list))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict(
            {str(species): nn for species, nn in zip(all_species, nns_per_species)}
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_blocks: List[TensorBlock] = []
        for species_str, network in self.layers.items():
            species = int(species_str)
            if species not in present_blocks:
                pass  # continue is not accepted by torchscript here
            else:
                block = features.block({"species_center": species})
                output_values = network(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        properties=Labels.range("properties", output_values.shape[-1]),
                    )
                )

        return TensorMap(keys=features.keys, blocks=new_blocks)


class LinearMap(torch.nn.Module):
    def __init__(self, all_species: List[int], n_inputs: int) -> None:
        super().__init__()

        # Build a neural network for each species
        layer_per_species = []
        for _ in all_species:
            layer_per_species.append(torch.nn.Linear(n_inputs, 1))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict(
            {
                str(species): layer
                for species, layer in zip(all_species, layer_per_species)
            }
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_blocks: List[TensorBlock] = []
        for species_str, layer in self.layers.items():
            species = int(species_str)
            if species not in present_blocks:
                pass  # continue is not accepted by torchscript here
            else:
                block = features.block({"species_center": species})
                output_values = layer(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        properties=Labels.single(),
                    )
                )

        return TensorMap(keys=features.keys, blocks=new_blocks)


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME

        # Check capabilities
        for output in capabilities.outputs.values():
            if output.quantity != "energy":
                raise ValueError(
                    "SOAP-BPNN only supports energy-like outputs, "
                    f"but a {next(iter(capabilities.outputs.values())).quantity} "
                    "was provided"
                )
            if output.per_atom:
                raise ValueError(
                    "SOAP-BPNN only supports per-structure outputs, "
                    "but a per-atom output was provided"
                )

        self.capabilities = capabilities
        self.all_species = capabilities.species
        self.hypers = hypers

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(capabilities.outputs)
        self.register_buffer(
            "composition_weights", torch.zeros((n_outputs, max(self.all_species) + 1))
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(capabilities.outputs.keys())
        }

        self.soap_calculator = rascaline.torch.SoapPowerSpectrum(**hypers["soap"])
        hypers_bpnn = hypers["bpnn"]
        hypers_bpnn["input_size"] = (
            len(self.all_species) ** 2
            * hypers["soap"]["max_radial"] ** 2
            * (hypers["soap"]["max_angular"] + 1)
        )

        self.bpnn = MLPMap(self.all_species, hypers_bpnn)
        self.neighbor_species_1_labels = Labels(
            names=["species_neighbor_1"],
            values=torch.tensor(self.all_species).reshape(-1, 1),
        )
        self.neighbor_species_2_labels = Labels(
            names=["species_neighbor_2"],
            values=torch.tensor(self.all_species).reshape(-1, 1),
        )

        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(self.all_species, n_inputs_last_layer)
                for output_name in capabilities.outputs.keys()
            }
        )

    def forward(
        self, systems: List[System], requested_outputs: Optional[List[str]] = None
    ) -> Dict[str, TensorMap]:
        if requested_outputs is None:  # default to all outputs
            requested_outputs = list(self.capabilities.outputs.keys())

        for requested_output in requested_outputs:
            if requested_output not in self.capabilities.outputs.keys():
                raise ValueError(
                    f"Requested output {requested_output} is not within "
                    "the model's capabilities."
                )

        soap_features = self.soap_calculator(systems)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(
            self.neighbor_species_1_labels.to(device)
        )
        soap_features = soap_features.keys_to_properties(
            self.neighbor_species_2_labels.to(device)
        )

        hidden_features = self.bpnn(soap_features)

        atomic_energies: Dict[str, metatensor.torch.TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in requested_outputs:
                atomic_energies[output_name] = apply_composition_contribution(
                    output_layer(hidden_features),
                    self.composition_weights[self.output_to_index[output_name]],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        total_energies: Dict[str, metatensor.torch.TensorMap] = {}
        for output_name, atomic_energy in atomic_energies.items():
            atomic_energy = atomic_energy.keys_to_samples("species_center")
            total_energies[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["center", "species_center"]
            )
            # Change the energy label from _ to (0, 1):
            total_energies[output_name] = metatensor.torch.TensorMap(
                keys=Labels(
                    names=["lambda", "sigma"],
                    values=torch.tensor([[0, 1]]),
                ),
                blocks=[total_energies[output_name].block()],
            )

        return total_energies

    def set_composition_weights(
        self, output_name: str, input_composition_weights: torch.Tensor
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][
            self.all_species
        ] = input_composition_weights
