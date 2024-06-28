from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
)
from torch_alchemical.models import AlchemicalModel as AlchemicalModelUpstream

from ...utils.data.dataset import DatasetInfo
from ...utils.dtype import dtype_to_str
from ...utils.export import export
from .utils import systems_to_torch_alchemical_batch


class AlchemicalModel(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        if len(dataset_info.targets) != 1:
            raise ValueError("The AlchemicalModel only supports a single target")

        target_name = next(iter(dataset_info.targets.keys()))
        if dataset_info.targets[target_name].quantity != "energy":
            raise ValueError("The AlchemicalModel only supports 'energies' as target")

        if dataset_info.targets[target_name].per_atom:
            raise ValueError("The AlchemicalModel does not support 'per-atom' training")

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=False,
            )
            for key, value in dataset_info.targets.items()
        }

        self.alchemical_model = AlchemicalModelUpstream(
            unique_numbers=self.atomic_types,
            **self.hypers["soap"],
            **self.hypers["bpnn"],
        )

        self.cutoff = self.hypers["soap"]["cutoff"]
        self.is_restarted = False

    def restart(self, dataset_info: DatasetInfo) -> "AlchemicalModel":
        if dataset_info != self.dataset_info:
            raise ValueError(
                "Alchemical model cannot be restarted with different "
                "dataset information"
            )
        self.is_restarted = True
        return self

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff,
                full_list=True,
            )
        ]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert len(outputs.keys()) == 1
        output_name = list(outputs.keys())[0]

        if selected_atoms is not None:
            raise NotImplementedError(
                "Alchemical Model does not support selected atoms."
            )
        options = self.requested_neighbor_lists()[0]
        batch = systems_to_torch_alchemical_batch(systems, options)
        predictions = self.alchemical_model(
            positions=batch["positions"],
            cells=batch["cells"],
            numbers=batch["numbers"],
            edge_indices=batch["edge_indices"],
            edge_offsets=batch["edge_offsets"],
            batch=batch["batch"],
        )

        total_energies: Dict[str, TensorMap] = {}
        keys = Labels(
            "_", torch.zeros((1, 1), dtype=torch.int32, device=predictions.device)
        )
        properties = Labels(
            "energy",
            torch.zeros((1, 1), dtype=torch.int32, device=predictions.device),
        )
        samples = Labels(
            names=["system"],
            values=torch.arange(
                len(predictions),
                device=predictions.device,
            ).view(-1, 1),
        )
        block = TensorBlock(
            samples=samples,
            components=[],
            properties=properties,
            values=predictions,
        )
        total_energies[output_name] = TensorMap(
            keys=keys,
            blocks=[block],
        )
        return total_energies

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "AlchemicalModel":

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
            raise ValueError(f"unsupported dtype {dtype} for AlchemicalModel")

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
        input_composition_weights: torch.Tensor,
        atomic_types: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        input_composition_weights = input_composition_weights.to(
            dtype=self.alchemical_model.composition_weights.dtype,
            device=self.alchemical_model.composition_weights.device,
        )
        index = [self.atomic_types.index(s) for s in atomic_types]
        composition_weights = input_composition_weights[:, index]
        self.alchemical_model.set_composition_weights(composition_weights)

    def set_normalization_factor(self, normalization_factor: torch.Tensor) -> None:
        """Set the normalization factor for output of the model."""
        self.alchemical_model.set_normalization_factor(normalization_factor)

    def set_basis_normalization_factor(
        self, basis_normalization_factor: torch.Tensor
    ) -> None:
        """Set the normalization factor for the basis functions of the model."""
        self.alchemical_model.set_basis_normalization_factor(basis_normalization_factor)

    def set_energies_scale_factor(self, energies_scale_factor: torch.Tensor) -> None:
        """Set the energies scale factor for the model."""
        self.alchemical_model.set_energies_scale_factor(energies_scale_factor)
