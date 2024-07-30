from typing import Dict, List, Optional, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System
from metatensor.torch.operations import sum_over_samples_block

from .data import Dataset, DatasetInfo, get_atomic_types
from .jsonschema import validate


class CompositionModel(torch.nn.Module):
    """Calculate the energy based on the stoichiometry in a system.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.

    :raises ValueError: If any target quantity in the dataset info is not an energy-like
        quantity.
    """

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        # Check capabilities
        for target in dataset_info.targets.values():
            if target.quantity != "energy":
                raise ValueError(
                    "CompositionModel only supports energy-like outputs, but a "
                    f"{target.quantity} output was provided."
                )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)
        self._weights: Dict[str, torch.Tensor[float]] = {}

    def train(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
    ) -> None:
        """Train/fit the composition weights for the datasets.

        :param datasets: Datasets to calculate the composition weights for.
        :raises ValueError: If the provided datasets contain unknown atomic types.
        :raises RuntimeError: If the linear system to calculate the composition weights
            cannot be solved.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        missing_types = sorted(get_atomic_types(datasets) - set(self.atomic_types))
        if missing_types:
            raise ValueError(
                f"Provided `datasets` contains unknown atomic types {missing_types}. "
                f"Known types from initilaization are {self.atomic_types}."
            )

        # Compute weights for each target in the dataset info
        for target_key in self.dataset_info.targets.keys():

            # CAVE: What if sample does not contain `target_key`
            targets = torch.stack(
                [
                    sample[target_key].block().values
                    for dataset in datasets
                    for sample in dataset
                ]
            )

            # remove component and property dimensions
            targets = targets.squeeze(dim=(1, 2))

            structure_list = [
                sample["system"] for dataset in datasets for sample in dataset
            ]

            dtype = structure_list[0].positions.dtype
            composition_features = torch.zeros(
                (len(structure_list), len(self.atomic_types)), dtype=dtype
            )
            for i_structure, structure in enumerate(structure_list):
                for i_types, atomic_type in enumerate(self.atomic_types):
                    composition_features[i_structure, i_types] = torch.sum(
                        structure.types == atomic_type
                    )

            regularizer = 1e-20
            while regularizer:
                if regularizer > 1e5:
                    raise RuntimeError(
                        "Failed to solve the linear system to calculate the "
                        "composition weights. The dataset is probably too small or "
                        "ill-conditioned."
                    )
                try:
                    self._weights[target_key] = torch.linalg.solve(
                        composition_features.T @ composition_features
                        + regularizer
                        * torch.eye(
                            composition_features.shape[1],
                            dtype=composition_features.dtype,
                            device=composition_features.device,
                        ),
                        composition_features.T @ targets,
                    )
                    break
                except torch._C._LinAlgError:
                    regularizer *= 10.0

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """
        return self({}, self.dataset_info.union(dataset_info))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy per atom.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            targets.
        :returns: A dictionary with the computed targets for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        :raises NotImplementedError: If `selected_atoms` is provided (not implemented).
        """

        if not self._weights:
            raise ValueError("No weights. Call `compute_weights` method first.")

        if outputs.keys() != self._weights.keys():
            raise ValueError(
                f"`outputs` keys ({', '.join(outputs.keys())}) contain unsupported "
                f"keys. Supported keys are ({', '.join(self._weights.keys())})."
            )

        if selected_atoms is not None:
            raise NotImplementedError("`selected_atoms` is not implemented.")

        # Compute the targets for each system by adding the composition weights times
        # number of atoms per atomic type.
        targets_out = {}
        for target_key, target in self.dataset_info.targets.items():
            weights = self._weights[target_key]
            targets: List[float] = []
            sample_values: List[List[int]] = []

            for i_system, system in enumerate(systems):
                targets_single = torch.zeros(len(system))

                for i_type, atomic_type in enumerate(self.atomic_types):
                    targets_single[atomic_type == system.types] = weights[i_type]

                targets += targets_single.tolist()
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]

            block = TensorBlock(
                values=torch.tensor(targets).reshape(-1, 1),
                samples=Labels(["system", "atom"], torch.tensor(sample_values)),
                components=[],
                properties=Labels.single(target_key),
            )

            if not target.per_atom:
                block = sum_over_samples_block(block, sample_names="atom")

            targets_out[target_key] = TensorMap(keys=Labels.single(), blocks=[block])

        return targets_out


def calculate_composition_weights(
    datasets: Union[Dataset, List[Dataset]], property: str
) -> Tuple[torch.Tensor, List[int]]:
    """Calculate the composition weights for a dataset.

    For now, it assumes per-system properties.

    :param dataset: Dataset to calculate the composition weights for.
    :returns: Composition weights for the dataset, as well as the
        list of species that the weights correspond to.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]

    # Note: `atomic_types` are sorted, and the composition weights are sorted as
    # well, because the species are sorted in the composition features.
    atomic_types = sorted(get_atomic_types(datasets))

    targets = torch.stack(
        [sample[property].block().values for dataset in datasets for sample in dataset]
    )
    targets = targets.squeeze(dim=(1, 2))  # remove component and property dimensions

    total_num_structures = sum([len(dataset) for dataset in datasets])
    dtype = datasets[0][0]["system"].positions.dtype
    composition_features = torch.empty(
        (total_num_structures, len(atomic_types)), dtype=dtype
    )
    structure_index = 0
    for dataset in datasets:
        for sample in dataset:
            structure = sample["system"]
            for j, s in enumerate(atomic_types):
                composition_features[structure_index, j] = torch.sum(
                    structure.types == s
                )
            structure_index += 1

    regularizer = 1e-20
    while regularizer:
        if regularizer > 1e5:
            raise RuntimeError(
                "Failed to solve the linear system to calculate the "
                "composition weights. The dataset is probably too small "
                "or ill-conditioned."
            )
        try:
            solution = torch.linalg.solve(
                composition_features.T @ composition_features
                + regularizer
                * torch.eye(
                    composition_features.shape[1],
                    dtype=composition_features.dtype,
                    device=composition_features.device,
                ),
                composition_features.T @ targets,
            )
            break
        except torch._C._LinAlgError:
            regularizer *= 10.0

    return solution, atomic_types


def apply_composition_contribution(
    atomic_property: TensorMap, composition_weights: torch.Tensor
) -> TensorMap:
    """Apply the composition contribution to an atomic property.

    :param atomic_property: Atomic property to apply the composition contribution to.
    :param composition_weights: Composition weights to apply.
    :returns: Atomic property with the composition contribution applied.
    """

    new_keys: List[int] = []
    new_blocks: List[TensorBlock] = []
    for key, block in atomic_property.items():
        atomic_type = int(key.values.item())
        new_keys.append(atomic_type)
        new_values = block.values + composition_weights[atomic_type]
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    new_keys_labels = Labels(
        names=["center_type"],
        values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
            -1, 1
        ),
    )

    return TensorMap(keys=new_keys_labels, blocks=new_blocks)
