import io
import math
import multiprocessing
import os
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
from metatensor.learn.data import Dataset, group_and_join
from metatensor.learn.data._namedtuple import namedtuple
from metatensor.torch import (
    Labels,
    TensorBlock,
    TensorMap,
    load_buffer,
    make_contiguous,
    make_contiguous_block,
    save_buffer,
)
from metatomic.torch import (
    ModelCapabilities,
    ModelOutput,
    System,
    load_system,
    load_system_buffer,
)
from metatomic.torch import save_buffer as save_system_buffer
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset

from metatrain.utils.data.readers.metatensor import (
    _check_tensor_map_metadata,
    _empty_tensor_map_like,
)
from metatrain.utils.data.smart_zip import SmartZip
from metatrain.utils.data.target_info import (
    TargetInfo,
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.external_naming import to_external_name
from metatrain.utils.units import get_gradient_units


def _set(values: List[int]) -> List[int]:
    """This function just does `list(set(values))`.

    But set is not torchscript compatible, so we do it manually.

    :param values: List of integer atomic types.
    :return: List of unique integer atomic types.
    """
    unique_values: List[int] = []
    for at_type in values:
        found = False
        for seen in unique_values:
            if seen == at_type:
                found = True
                break
        if not found:
            unique_values.append(at_type)

    return unique_values


class DatasetInfo:
    """A class that contains information about datasets.

    This class is used to communicate additional dataset details to the
    training functions of the individual models.

    :param length_unit: Unit of length used in the dataset.

        The list of possible units is available `here
        <https://docs.metatensor.org/metatomic/latest/torch/reference/misc.html#known-quantities-units>`_.
    :param atomic_types: List containing all integer atomic types present in the
        dataset. ``atomic_types`` will be stored as a sorted list of **unique** atomic
        types.
    :param targets: Information about targets in the dataset.
    :param extra_data: Optional dictionary containing additional data that is not
        used as a target, but is still relevant to the dataset.
    """

    def __init__(
        self,
        length_unit: str,
        atomic_types: List[int],
        targets: Dict[str, TargetInfo],
        extra_data: Optional[Dict[str, TargetInfo]] = None,
    ):
        # verify that `length_unit` and `atomic_types` are valid for metatomic
        _ = ModelCapabilities(
            outputs={"energy": ModelOutput(unit="eV")},
            length_unit=length_unit,
            atomic_types=atomic_types,
        )

        self.length_unit = length_unit
        self._atomic_types = _set(atomic_types)
        self.targets = targets
        self.extra_data: Dict[str, TargetInfo] = (
            extra_data if extra_data is not None else {}
        )

    @property
    def atomic_types(self) -> List[int]:
        """Sorted list of unique integer atomic types."""
        return sorted(self._atomic_types)

    @atomic_types.setter
    def atomic_types(self, value: List[int]) -> None:
        self._atomic_types = _set(value)

    @property
    def device(self) -> Optional[torch.device]:
        """Return the device where the tensors of DatasetInfo are located.

        This function only checks the device of the first target
        and assumes that all targets and extra data are on the same device.
        This is guaranteed if the ``to()`` method has been used to move
        the DatasetInfo to a specific device.
        """
        if len(self.targets) == 0:
            return None
        first_target = list(self.targets.values())[0]
        return first_target.device

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "DatasetInfo":
        """
        Return a copy with all tensors moved to the device and dtype.

        :param device: The device to move the tensors to.
        :param dtype: The dtype to move the tensors to.
        :return: A copy of the DatasetInfo with all tensors moved to the device and
            dtype.
        """
        new = self.copy()
        for key, target_info in new.targets.items():
            new.targets[key] = target_info.to(device=device, dtype=dtype)
        for key, extra_data in new.extra_data.items():
            new.extra_data[key] = extra_data.to(device=device, dtype=dtype)
        return new

    def __repr__(self) -> str:
        return "DatasetInfo(length_unit={!r}, atomic_types={!r}, targets={!r})".format(
            self.length_unit, self.atomic_types, self.targets
        )

    @torch.jit.unused
    def __eq__(self, other: Any) -> bool:
        """
        Equality operator.

        :param other: Another object to compare with.
        :return: True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, DatasetInfo):
            return False
        return (
            self.length_unit == other.length_unit
            and self._atomic_types == other._atomic_types
            and self.targets == other.targets
            and self.extra_data == other.extra_data
        )

    def copy(self) -> "DatasetInfo":
        """
        :return: A shallow copy of the DatasetInfo.
        """
        return DatasetInfo(
            length_unit=self.length_unit,
            atomic_types=self.atomic_types.copy(),
            targets=self.targets.copy(),
            extra_data=self.extra_data.copy(),
        )

    @torch.jit.unused
    def update(self, other: "DatasetInfo") -> None:
        """Update this instance with the union of itself and ``other``.

        :param other: Another :py:class:`DatasetInfo` instance to update this one with.
        :raises ValueError: If the ``length_units`` are different.
        """
        if self.length_unit != other.length_unit:
            raise ValueError(
                "Can't update DatasetInfo with a different `length_unit`: "
                f"('{self.length_unit}' != '{other.length_unit}')"
            )

        self.atomic_types = self.atomic_types + other.atomic_types

        intersecting_target_keys = self.targets.keys() & other.targets.keys()
        for key in intersecting_target_keys:
            if not self.targets[key].is_compatible_with(other.targets[key]):
                raise ValueError(
                    f"Can't update DatasetInfo with different target information for "
                    f"target '{key}': {self.targets[key]} is not compatible with "
                    f"{other.targets[key]}. If the units, quantity and keys of the two "
                    "targets are the same, this must be due to a mismatch in the "
                    "internal metadata of the layout."
                )
        self.targets.update(other.targets)

        intersecting_extra_data_keys = self.extra_data.keys() & other.extra_data.keys()
        for key in intersecting_extra_data_keys:
            if not self.extra_data[key].is_compatible_with(other.extra_data[key]):
                raise ValueError(
                    f"Can't update DatasetInfo with different extra data information "
                    f"for key '{key}': {self.extra_data[key]} is not compatible with "
                    f"{other.extra_data[key]}. If the units, quantity and keys of the "
                    "two extra data dictionaries are the same, this must be due to a "
                    "mismatch in the internal metadata of the layout."
                )
        self.extra_data.update(other.extra_data)

    def union(self, other: "DatasetInfo") -> "DatasetInfo":
        """
        Return the union of this instance with ``other``.

        :param other: Another :py:class:`DatasetInfo` instance to combine with this one.
        :return: A new :py:class:`DatasetInfo` instance containing the union of this
            instance and ``other``.
        """
        new = self.copy()
        new.update(other)
        return new

    @torch.jit.unused
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Custom ``__setstate__`` to allow loading old checkpoints where ``extra_data`` is
        missing.

        :param state: The state to set.
        """
        self.length_unit = state["length_unit"]
        self._atomic_types = state["_atomic_types"]
        self.targets = state["targets"]
        self.extra_data = state.get("extra_data", {})


def get_stats(dataset: Union[Dataset, Subset], dataset_info: DatasetInfo) -> str:
    """
    Returns the statistics of a dataset or subset as a string.

    :param dataset: The dataset or subset to analyze.
    :param dataset_info: The DatasetInfo associated with the dataset.
    :return: A string containing the computed statistics for the dataset.
    """

    dataset_len = len(dataset)
    stats = f"Dataset containing {dataset_len} structures"
    if dataset_len == 0:
        return stats

    # target_names will be used to store names of the targets,
    # along with their gradients
    target_names = []
    for key, tensor_map in dataset[0]._asdict().items():
        if key == "system":
            continue
        target_names.append(key)
        gradients_list = tensor_map.block(0).gradients_list()
        for gradient in gradients_list:
            target_names.append(f"{key}_{gradient}_gradients")

    sums = {key: 0.0 for key in target_names}
    sums_of_squares = {key: 0.0 for key in target_names}
    n_elements = {key: 0 for key in target_names}
    for sample in dataset:
        for key in target_names:
            if "_gradients" not in key:  # not a gradient
                tensors = [block.values for block in sample[key].blocks()]
            else:
                # The name is <basename>_<gradname>_gradients
                original_key = "_".join(key.split("_")[:-2])
                gradient_name = key.replace(f"{original_key}_", "").replace(
                    "_gradients", ""
                )
                tensors = [
                    block.gradient(gradient_name).values
                    for block in sample[original_key].blocks()
                ]
            sums[key] += sum(tensor.sum() for tensor in tensors)
            sums_of_squares[key] += sum((tensor**2).sum() for tensor in tensors)
            n_elements[key] += sum(tensor.numel() for tensor in tensors)
    means = {key: sums[key] / n_elements[key] for key in target_names}
    means_of_squares = {
        key: sums_of_squares[key] / n_elements[key] for key in target_names
    }
    stds = {
        key: (means_of_squares[key] - means[key] ** 2) ** 0.5 for key in target_names
    }

    # Find units
    units = {}
    for key in target_names:
        if key not in dataset_info.targets:
            continue
        # Gets the units of an output
        if key.endswith("_gradients"):
            # handling <base_name>_<gradient_name>_gradients
            base_name = key[:-10]
            gradient_name = base_name.split("_")[-1]
            base_name = base_name.replace(f"_{gradient_name}", "")
            base_unit = dataset_info.targets[base_name].unit
            unit = get_gradient_units(
                base_unit, gradient_name, dataset_info.length_unit
            )
        else:
            unit = dataset_info.targets[key].unit
        units[key] = unit

    # TODO: add extra data statistics?

    stats += "\n    Mean and standard deviation of targets:"
    for key in target_names:
        if key not in means or key not in units or key not in stds:
            continue
        stats += (
            f"\n    - {to_external_name(key, dataset_info.targets)}: "  # type: ignore
            + f"\n      - mean {means[key]:.4g}"
            + (f" {units[key]}" if units[key] != "" else "")
            + f"\n      - std  {stds[key]:.4g}"
            + (f" {units[key]}" if units[key] != "" else "")
        )

    return stats


def get_atomic_types(datasets: Union[Dataset, List[Dataset]]) -> List[int]:
    """List of all atomic types present in a dataset or list of datasets.

    :param datasets: the dataset, or list of datasets
    :return: sorted list of all atomic types present in the datasets
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    types = set()
    for dataset in datasets:
        for sample in dataset:
            system = sample["system"]
            types.update(set(system.types.tolist()))

    return sorted(types)


def get_all_targets(datasets: Union[Dataset, List[Dataset]]) -> List[str]:
    """Sorted list of all unique targets present in a dataset or list of datasets.

    :param datasets: the dataset(s).
    :return: Sorted list of all targets present in the dataset(s).
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    # The following does not work because the `dataset` can also
    # be a `Subset` object:
    # return list(dataset.targets.keys())

    # Iterate over all single instances of the dataset:
    target_names = []
    for dataset in datasets:
        for sample in dataset:
            # system not needed
            target_names += [key for key in sample._asdict().keys() if key != "system"]

    return sorted(set(target_names))


class CollateFn:
    def __init__(
        self,
        target_keys: List[str],
        callables: Optional[List[Callable]] = None,
        join_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.target_keys: Set[str] = set(target_keys)
        self.callables: List[Callable] = callables if callables is not None else []
        self.join_kwargs: Dict[str, Any] = join_kwargs or {"different_keys": "union"}

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[int], List[str], List[int], List[str], List[int]]:
        """
        :param batch: A batch
        :return: A tuple containing:
            - a single tensor containing all systems, targets and extra data
            - a list with the sizes of each system buffer
            - a list with the names of each target
            - a list with the sizes of each target buffer
            - a list with the names of each extra data
            - a list with the sizes of each extra data buffer
        """
        # group & join
        collated = group_and_join(batch, join_kwargs=self.join_kwargs)
        data = collated._asdict()

        # pull off systems
        systems = data.pop("system")

        # split into targets vs extra data
        targets: Dict[str, TensorMap] = {}
        extra: Dict[str, TensorMap] = {}

        for key, value in data.items():
            if key in self.target_keys:
                targets[key] = value
            else:
                extra[key] = value

        for callable in self.callables:
            systems, targets, extra = callable(systems, targets, extra)

        target_names = list(targets.keys())
        extra_names = list(extra.keys())

        system_buffers = [
            save_system_buffer(_make_system_contiguous(s)) for s in systems
        ]
        target_buffers = [
            save_buffer(make_contiguous(targets[name])) for name in target_names
        ]
        extra_buffers = [
            save_buffer(make_contiguous(extra[name])) for name in extra_names
        ]

        system_sizes = [len(b) for b in system_buffers]
        target_sizes = [len(b) for b in target_buffers]
        extra_sizes = [len(b) for b in extra_buffers]

        blob = torch.concatenate(system_buffers + target_buffers + extra_buffers)

        return blob, system_sizes, target_names, target_sizes, extra_names, extra_sizes


def unpack_batch(
    batch: Any,
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    """
    Unpacks a batch into its constituent parts.

    :param batch: The batch to unpack.
    :return: A tuple with the unpacked batch
    """
    blob, system_sizes, target_names, target_sizes, extra_names, extra_sizes = batch

    all_buffers = torch.split(blob, system_sizes + target_sizes + extra_sizes)
    systems = all_buffers[: len(system_sizes)]
    targets = {
        name: buf
        for name, buf in zip(
            target_names,
            all_buffers[len(system_sizes) : len(system_sizes) + len(target_names)],
            strict=True,
        )
    }
    extra_data = {
        name: buf
        for name, buf in zip(
            extra_names,
            all_buffers[len(system_sizes) + len(target_names) :],
            strict=True,
        )
    }

    systems = list(load_system_buffer(s) for s in systems)
    targets = {key: load_buffer(t) for key, t in targets.items()}
    extra_data = {key: load_buffer(t) for key, t in extra_data.items()}
    return systems, targets, extra_data


def check_datasets(train_datasets: List[Dataset], val_datasets: List[Dataset]) -> None:
    """Check that the training and validation sets are compatible with one another

    Although these checks will not fit all use cases, most models would be expected
    to be able to use this function.

    :param train_datasets: A list of training datasets to check.
    :param val_datasets: A list of validation datasets to check
    :raises TypeError: If the ``dtype`` within the datasets are inconsistent.
    :raises ValueError: If the `val_datasets` has a target that is not present in
        the ``train_datasets``.
    :raises ValueError: If the training or validation set contains chemical species
        or targets that are not present in the training set
    """
    # Check that system `dtypes` are consistent within datasets
    desired_dtype = None
    for train_dataset in train_datasets:
        if len(train_dataset) == 0:
            continue

        actual_dtype = train_dataset[0].system.positions.dtype
        if desired_dtype is None:
            desired_dtype = actual_dtype

        if actual_dtype != desired_dtype:
            raise TypeError(
                "`dtype` between datasets is inconsistent, "
                f"found {desired_dtype} and {actual_dtype} in training datasets"
            )

    for val_dataset in val_datasets:
        if len(val_dataset) == 0:
            continue

        actual_dtype = val_dataset[0].system.positions.dtype

        if desired_dtype is None:
            desired_dtype = actual_dtype

        if actual_dtype != desired_dtype:
            raise TypeError(
                "`dtype` between datasets is inconsistent, "
                f"found {desired_dtype} and {actual_dtype} in validation datasets"
            )

    # Get all targets in the training and validation sets:
    train_targets = get_all_targets(train_datasets)
    val_targets = get_all_targets(val_datasets)

    # Check that the validation sets do not have targets that are not in the
    # training sets:
    for target in val_targets:
        if target not in train_targets:
            raise ValueError(
                f"The validation dataset has a target ({target}) that is not present "
                "in the training dataset."
            )
    # Get all the species in the training and validation sets:
    all_train_species = get_atomic_types(train_datasets)
    all_val_species = get_atomic_types(val_datasets)

    # Check that the validation sets do not have species that are not in the
    # training sets:
    for species in all_val_species:
        if species not in all_train_species:
            raise ValueError(
                f"The validation dataset has a species ({species}) that is not in the "
                "training dataset. This could be a result of a random train/validation "
                "split. You can avoid this by providing a validation dataset manually."
            )


def _train_test_random_split(
    train_dataset: Dataset,
    train_size: float,
    test_size: float,
) -> List[Dataset]:
    if train_size <= 0:
        raise ValueError("Fraction of the train set is smaller or equal to 0!")

    # normalize the sizes
    size_sum = train_size + test_size
    train_size /= size_sum
    test_size /= size_sum

    # find number of samples in the train and test sets
    test_len = math.floor(len(train_dataset) * test_size)
    if test_len == 0:
        warnings.warn(
            "Requested dataset of zero length. This dataset will be empty.",
            UserWarning,
            stacklevel=2,
        )
    train_len = len(train_dataset) - test_len
    if train_len == 0:
        raise ValueError("No samples left in the training set.")

    # find train, test indices
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]

    return [
        Subset(train_dataset, train_indices),
        Subset(train_dataset, test_indices),
    ]


def load_indices(indices_spec: Union[List[int], str]) -> List[int]:
    """Load indices from a list or a file path.

    :param indices_spec: Either a list of integers or a path to a text file
        containing one index per line.
    :returns: List of integer indices (may be empty for empty val/test sets).
    :raises ValueError: If indices contain invalid values.
    """
    # Check for list-like objects (including OmegaConf ListConfig)
    if not isinstance(indices_spec, str):
        indices = list(indices_spec)
    else:
        # It's a path string
        path = Path(indices_spec)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise ValueError(f"Indices file not found: {path}")
        indices = np.loadtxt(path, dtype=int).reshape(-1).tolist()

    if indices and any(i < 0 for i in indices):
        raise ValueError("Indices must be non-negative integers")

    return indices


# The dataset-level informative files live under `metadata/`; currently only
# the atom counts, more may be added in the future.
ATOM_COUNTS_MEMBER = "metadata/atom_counts.npy"


def _parse_disk_dataset_member_name(name: str) -> Optional[Tuple[int, str]]:
    """Parse a DiskDataset zip member name into ``(entry_number, field_name)``.

    Valid members are exactly ``<N>/system.mta`` and ``<N>/<field>.mts`` with
    ``N`` an unpadded decimal number (``00/`` would silently alias ``0/``).

    :param name: The zip member name.
    :return: ``(entry, field)`` with ``field == "system"`` for the system
        member, or ``None`` when the name is not valid DiskDataset content.
    """
    slash = name.find("/")
    if slash <= 0:
        return None
    head = name[:slash]
    if not head.isdigit() or (len(head) > 1 and head[0] == "0"):
        return None
    tail = name[slash + 1 :]
    if "/" in tail:
        return None
    if tail == "system.mta":
        return int(head), "system"
    if tail.endswith(".mts") and len(tail) > len(".mts"):
        return int(head), tail[: -len(".mts")]
    return None


def _format_member_names(names: List[str], limit: int = 10) -> str:
    """Format a capped list of zip member names for messages.

    :param names: The member names.
    :param limit: Maximum number of names to spell out.
    :return: The formatted list.
    """
    formatted = str(names[:limit])
    if len(names) > limit:
        formatted += f" (+{len(names) - limit} more)"
    return formatted


class _MemberScan(NamedTuple):
    """Classification of a DiskDataset zip's members, one row per member."""

    entries: np.ndarray
    """Entry number of each member, ``-1`` for non-format members."""
    field_ids: np.ndarray
    """Field id of each member, ``-1`` for non-format members."""
    field_id_by_name: Dict[str, int]
    """Field name to field id, in order of first appearance."""
    ignored_names: List[str]
    """Members that are not part of the DiskDataset format."""
    has_atom_counts: bool
    """Whether the zip contains the ``metadata/atom_counts.npy`` member."""


def _scan_members(zip_file: SmartZip, path: Union[str, Path]) -> _MemberScan:
    """Classify every member of a DiskDataset zip into ``(entry, field)``.

    The entry members ``<N>/system.mta`` and ``<N>/<field>.mts`` must be
    STORED (uncompressed); any other file is collected as ignored, for
    :py:func:`_validate_format` to report. A zip of a dataset folder made
    with standard tools (from inside the folder, without compression)
    therefore classifies like DiskDatasetWriter output.

    :param zip_file: The indexed zip.
    :param path: The zip's path, for error messages.
    :return: The classification.
    """
    n_members = len(zip_file)
    entries = np.full(n_members, -1, dtype=np.int64)
    field_ids = np.full(n_members, -1, dtype=np.int64)
    field_id_by_name: Dict[str, int] = {}
    ignored_names: List[str] = []
    has_atom_counts = False
    for i, info in enumerate(zip_file.infoiter()):
        name = info.filename
        if name == ATOM_COUNTS_MEMBER:
            has_atom_counts = True
            continue
        parsed = _parse_disk_dataset_member_name(name)
        if parsed is None:
            ignored_names.append(name)
            continue
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"'{path}' is not a valid DiskDataset zip: file "
                f"'{name}' is compressed. DiskDataset zips must be "
                "uncompressed (STORED); re-create the zip without "
                "compression, e.g. `zip -rX -Z store <dataset>.zip .` "
                "from inside the dataset folder."
            )
        entry, field = parsed
        entries[i] = entry
        field_ids[i] = field_id_by_name.setdefault(field, len(field_id_by_name))
    return _MemberScan(
        entries, field_ids, field_id_by_name, ignored_names, has_atom_counts
    )


def _validate_format(scan: _MemberScan, path: Union[str, Path]) -> int:
    """Reject zips that cannot be read reliably, with actionable messages.

    Checks, in order: at least one system member; a dense ``0..N-1`` entry
    range (duplicated or missing entry numbers would be read positionally
    and silently alias other samples); the same fields in every entry (a
    missing member would otherwise only surface when that sample happens to
    be read); no duplicated or orphaned members. A zip can violate several
    at once: the first failing check is the one reported.

    :param scan: The member classification from :py:func:`_scan_members`.
    :param path: The zip's path, for error messages.
    :return: The number of samples in the dataset.
    """
    if "system" not in scan.field_id_by_name:
        message = (
            "Could not find any `<N>/system.mta` file in the zip. "
            "The dataset format might be wrong, or the dataset might be "
            "empty. Empty disk datasets are not supported."
        )
        if scan.ignored_names:
            message += (
                " The zip does contain files that are not part of the "
                f"format: {_format_member_names(scan.ignored_names)}."
            )
            first_component = scan.ignored_names[0].split("/", 1)[0]
            if all(n.startswith(f"{first_component}/") for n in scan.ignored_names):
                message += (
                    " They all share the prefix "
                    f"'{first_component}/'; if you zipped a dataset "
                    "folder from outside, re-create the zip from inside "
                    f"it: `cd {first_component} && zip -rX -Z store "
                    "../<dataset>.zip .`"
                )
        raise ValueError(message)

    # Entries are read positionally as "0/", "1/", ...: require exactly
    # the dense range, and fail loudly on duplicated entry names (e.g.
    # produced by appending with an older, buggy DiskDatasetWriter that
    # restarted indexing at 0 -- reading those would silently return
    # the last-written copy) or on gaps.
    system_id = scan.field_id_by_name["system"]
    entry_numbers = scan.entries[scan.field_ids == system_id]
    if not np.array_equal(np.sort(entry_numbers), np.arange(len(entry_numbers))):
        raise ValueError(
            "This DiskDataset zip file does not contain a dense range of "
            "entries `0/`, `1/`, ...: it has gaps or duplicated entry "
            "numbers and cannot be read reliably; re-write the dataset."
        )
    n_samples = len(entry_numbers)

    # Every entry must carry the same fields ("homogeneous" format).
    present = np.zeros((n_samples, len(scan.field_id_by_name)), dtype=bool)
    for field_id in scan.field_id_by_name.values():
        # entries are the dense range 0..N-1, so they index rows directly
        mask = (scan.field_ids == field_id) & (scan.entries < n_samples)
        present[scan.entries[mask], field_id] = True
    missing = np.argwhere(~present)
    if len(missing):
        entry, field_id = (int(v) for v in missing[0])
        field_name = list(scan.field_id_by_name)[field_id]
        raise ValueError(
            f"This DiskDataset zip is not homogeneous: entry {entry} has no "
            f"'{field_name}' file while other entries do "
            f"({len(missing)} missing files in total). Every entry must "
            "contain the same fields; re-write the dataset."
        )
    # With no member missing, a valid zip has exactly one member per
    # (entry, field) cell. A surplus means field members for entries
    # without a system, or duplicated names; both are out of spec.
    n_valid_members = int((scan.field_ids >= 0).sum())
    if n_valid_members != present.shape[0] * present.shape[1]:
        raise ValueError(
            "This DiskDataset zip contains duplicated files or field "
            "files for entries that have no `system.mta` (e.g. "
            "`7/energy.mts` without `7/system.mta`); re-write the dataset."
        )
    return n_samples


def _load_atom_counts(zip_file: SmartZip, n_samples: int) -> Optional[np.ndarray]:
    """Load the per-sample atom counts written by DiskDatasetWriter.

    Lets ``get_num_atoms()``/``get_all_atom_counts()`` answer without
    opening and deserializing every system, which is what makes DiskDataset
    usable with
    :class:`~metatrain.utils.data.samplers.MaxAtomDistributedBatchSampler`.

    :param zip_file: The indexed zip; must contain ``metadata/atom_counts.npy``.
    :param n_samples: The number of samples in the dataset.
    :return: Atom counts for all samples, or ``None`` (with a warning) when
        the stored counts are stale.
    """
    with zip_file.open(ATOM_COUNTS_MEMBER) as f:
        atom_counts = np.load(f)
    if len(atom_counts) >= n_samples:
        return atom_counts[:n_samples].astype(np.int64, copy=False)
    warnings.warn(
        f"Ignoring '{ATOM_COUNTS_MEMBER}' in this DiskDataset: "
        f"it has {len(atom_counts)} entries, fewer than the "
        f"{n_samples} samples found. It is likely stale.",
        stacklevel=2,
    )
    return None


class DiskDataset(torch.utils.data.Dataset):
    """
    A class representing a dataset stored on disk.

    The dataset is stored in a zip file, where each sample is stored in a separate
    directory. The directory's name is the index of the sample (e.g. ``0/``), and the
    files in the directory are the system (``system.mta``) and the targets (each named
    ``<target_name>.mts``). These are ``metatomic.torch.System`` and
    ``metatensor.torch.TensorMap`` objects, respectively. Extra data fields (e.g.
    ``charge.mts`` and ``spin_multiplicity.mts`` for charge and spin conditioning)
    can be stored in the same way and are read when the corresponding names appear
    in the ``extra_data`` section of the options file.

    Such a dataset can be created conveniently using the :py:class:`DiskDatasetWriter`
    class.

    :param path: Path to the zip file containing the dataset.
    :param fields: List of fields to read from the dataset.
        If None, all fields will be read. If it is a dictionary,
        it is treated as the target configuration, and we search for a "key"
        entry that maps the target name to the field name in the dataset.
    """

    def __init__(
        self,
        path: Union[str, Path],
        fields: Optional[list[str] | dict[str, Any]] = None,
    ):
        self.zip_file_path = path

        if isinstance(fields, dict):
            fields = {options.get("key", key): key for key, options in fields.items()}

        # SmartZip parses the zip's central directory once, into compact
        # numpy arrays of member locations, and reads members from those
        # (opening a zipfile.ZipFile per dataloader worker instead does not
        # scale to zips with millions of members). The scan and validation
        # only check the DiskDataset format on top of it.
        self._zip = SmartZip(path)
        scan = _scan_members(self._zip, path)
        self._len = _validate_format(scan, path)
        self._field_names = ["system"] + [
            f for f in scan.field_id_by_name if f != "system"
        ]
        self._atom_counts: Optional[np.ndarray] = (
            _load_atom_counts(self._zip, self._len) if scan.has_atom_counts else None
        )

        # Determine which fields are going to be read
        if fields is None:
            self._fields_to_read = self._field_names.copy()
        else:
            # Check that the requested fields are present in the dataset
            fields_to_read = ["system", *fields]
            missing_fields = set(fields_to_read) - set(self._field_names)
            if missing_fields:
                raise ValueError(
                    f"Fields {list(missing_fields)} were requested but "
                    "are not present in this disk dataset. "
                    f"Available fields: {self._field_names[1:]}"
                )
            self._fields_to_read = fields_to_read

        self._fields_to_read.append("mtt::aux::system_index")

        fields_map = fields if isinstance(fields, dict) else {}
        self._sample_class = namedtuple(
            "Sample", [fields_map.get(field, field) for field in self._fields_to_read]
        )

        # The dataset is valid: report skipped files, once.
        # DiskDatasetWriter output never triggers this.
        if scan.ignored_names:
            warnings.warn(
                f"Skipping {len(scan.ignored_names)} file(s) in '{path}' "
                "that are not part of the DiskDataset format: "
                f"{_format_member_names(scan.ignored_names)}. This is fine "
                "if they are extra files stored alongside the dataset; if "
                "they were meant to be read as data, name them "
                "`<N>/system.mta` or `<N>/<target>.mts`.",
                stacklevel=2,
            )

    def _read_member(self, index: int, field_name: str) -> io.BytesIO:
        """Read one zip member of the sample at ``index``.

        Member names are deterministic in the DiskDataset format, so the
        name is rebuilt from the index and field and read from the
        :py:class:`SmartZip`, which validates the local header and verifies
        the CRC-32 from the central directory.

        :param index: Positional dataset index.
        :param field_name: Field to read ("system" or a target/extra field).
        :return: The member contents.
        """
        if not 0 <= index < self._len:
            raise IndexError(
                f"index {index} is out of range for a DiskDataset with "
                f"{self._len} samples"
            )
        name = (
            f"{index}/system.mta"
            if field_name == "system"
            else f"{index}/{field_name}.mts"
        )
        with self._zip.open(name) as member:
            return io.BytesIO(member.read())

    def __len__(self) -> int:
        return self._len

    def _require_atom_counts(self) -> np.ndarray:
        """The dataset's atom counts, or a clear error when the zip lacks them.

        The reader never mutates the dataset file (doing so safely under
        distributed training and parallel filesystems is not worth the
        machinery), so datasets without ``metadata/atom_counts.npy`` must be
        updated
        by the user before atom-count-based sampling can be used.

        :return: Atom counts for all samples, in dataset order.
        """
        if self._atom_counts is None:
            raise ValueError(
                f"'{self.zip_file_path}' has no 'metadata/atom_counts.npy' "
                "file, which is required for atom-count-based sampling "
                "(e.g. `max_atoms_per_batch`). Re-write the dataset with the "
                "current DiskDatasetWriter (which always includes it), append "
                "to it with DiskDatasetWriter(..., append=True), or add "
                "'metadata/atom_counts.npy' (an int64 numpy array with the "
                "atom count of each entry, in entry order) to the zip "
                "yourself."
            )
        return self._atom_counts

    def get_num_atoms(self, i: int) -> int:
        """
        Return the atom count for sample index ``i``, using the
        ``metadata/atom_counts.npy`` file written by
        :py:class:`DiskDatasetWriter`. Enables use with
        :class:`~metatrain.utils.data.samplers.MaxAtomDistributedBatchSampler`.

        :param i: Dataset index.
        :return: Number of atoms in sample ``i``.
        """
        return int(self._require_atom_counts()[i])

    def get_all_atom_counts(self) -> np.ndarray:
        """
        Return atom counts for all samples in one vectorized call, using the
        ``metadata/atom_counts.npy`` file written by
        :py:class:`DiskDatasetWriter`.

        :return: Atom counts for all samples, in dataset order.
        """
        return self._require_atom_counts()

    def __getitem__(self, index: int) -> Any:
        system_and_targets = []
        for field_name in self._fields_to_read:
            if field_name == "system":
                system = load_system(self._read_member(index, field_name))
                system_and_targets.append(system)
            elif field_name == "mtt::aux::system_index":
                tensor_map = TensorMap(
                    keys=Labels(["_"], torch.tensor([[0]])),
                    blocks=[
                        TensorBlock(
                            # Integer values are not supported (coming soon)
                            values=torch.tensor([[index]]).to(torch.float64),
                            samples=Labels(
                                names=["system"],
                                values=torch.tensor([[index]]),
                            ),
                            components=[],
                            properties=Labels(["_"], torch.tensor([[0]])),
                        )
                    ],
                )
                system_and_targets.append(tensor_map)
            else:
                numpy_buffer = np.load(self._read_member(index, field_name))
                tensor_buffer = torch.from_numpy(numpy_buffer)
                tensor_map = load_buffer(tensor_buffer)
                system_and_targets.append(tensor_map)
        return self._sample_class(*system_and_targets)

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]

    def get_target_info(
        self, target_config: DictConfig, is_extra_data: bool = False
    ) -> Dict[str, TargetInfo]:
        """
        Get information about the targets in the dataset.

        :param target_config: The user-provided (through the yaml file) target
            configuration.
        :param is_extra_data: Whether the target information is for extra data or not.
            If ``True``, auxiliary variables will be added here, e.g. the system index.
        :return: A dictionary mapping target names to :py:class:`TargetInfo` objects.
        """
        target_info_dict = {}
        for target_key, target in target_config.items():
            is_energy = (
                (target["quantity"] == "energy")
                and target["sample_kind"] == "system"
                and target["num_subtargets"] == 1
                and target["type"] == "scalar"
            )
            tensor_map = self[0][target_key]  # always > 0 samples, see above
            if is_energy:
                if len(tensor_map) != 1:
                    raise ValueError("Energy TensorMaps should have exactly one block.")
                add_position_gradients = tensor_map.block().has_gradient("positions")
                add_strain_gradients = tensor_map.block().has_gradient("strain")
                target_info = get_energy_target_info(
                    target_key, target, add_position_gradients, add_strain_gradients
                )
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                target_info_dict[target_key] = target_info
            else:
                target_info = get_generic_target_info(target_key, target)
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                # make sure that the properties of the target_info.layout also match the
                # actual properties of the tensor maps
                if not target_info.is_atomic_basis:
                    target_info.layout = _empty_tensor_map_like(tensor_map)
                target_info_dict[target_key] = target_info

        if is_extra_data:
            # Add the system indices as extra data
            target_info_dict["mtt::aux::system_index"] = get_generic_target_info(
                "system_index",
                {
                    "quantity": "",
                    "unit": "",
                    "type": "scalar",
                    "sample_kind": "system",
                    "num_subtargets": 1,
                },
            )
        return target_info_dict


def _is_disk_dataset(dataset: Any) -> bool:
    # this also needs to detect if it's a ``torch.nn.utils.data.Subset`` object
    # with a ``DiskDataset`` object as its dataset, recursively
    if isinstance(dataset, DiskDataset):
        return True
    if isinstance(dataset, torch.utils.data.Subset):
        return _is_disk_dataset(dataset.dataset)
    return False


def _save_indices(
    train_indices: List[Optional[List[int]]],
    val_indices: List[Optional[List[int]]],
    test_indices: List[Optional[List[int]]],
    checkpoint_dir: Union[str, Path],
) -> None:
    # Save the indices of the training, validation, and test sets to the checkpoint
    # directory. This is useful for plotting errors and similar.

    # case 1: all indices are None (i.e. all datasets were user-provided explicitly)
    if all(indices is None for indices in train_indices):
        pass

    # case 2: there is only one dataset
    elif len(train_indices) == 1:  # val and test are the same length
        os.mkdir(os.path.join(checkpoint_dir, "indices/"))
        if train_indices[0] is not None:
            np.savetxt(
                os.path.join(checkpoint_dir, "indices/training.txt"),
                train_indices[0],
                fmt="%d",
            )
        if val_indices[0] is not None:
            np.savetxt(
                os.path.join(checkpoint_dir, "indices/validation.txt"),
                val_indices[0],
                fmt="%d",
            )
        if test_indices[0] is not None:
            np.savetxt(
                os.path.join(checkpoint_dir, "indices/test.txt"),
                test_indices[0],
                fmt="%d",
            )

    # case 3: there are multiple datasets
    else:
        os.makedirs(os.path.join(checkpoint_dir, "indices/"), exist_ok=True)
        for i, (train, val, test) in enumerate(
            zip(train_indices, val_indices, test_indices, strict=True)
        ):
            if train is not None:
                np.savetxt(
                    os.path.join(checkpoint_dir, f"indices/training_{i}.txt"),
                    train,
                    fmt="%d",
                )
            if val is not None:
                np.savetxt(
                    os.path.join(checkpoint_dir, f"indices/validation_{i}.txt"),
                    val,
                    fmt="%d",
                )
            if test is not None:
                np.savetxt(
                    os.path.join(checkpoint_dir, f"indices/test_{i}.txt"),
                    test,
                    fmt="%d",
                )


def get_num_workers() -> int:
    """
    Gets a good number of workers for data loading.

    :return: A good number of workers for data loading.
    """

    if not fork_is_available():
        return 0

    # len(os.sched_getaffinity(0)) detects thread counts set by slurm,
    # multiprocessing.cpu_count() doesn't but is more portable
    if hasattr(os, "sched_getaffinity"):
        num_threads = min(len(os.sched_getaffinity(0)), multiprocessing.cpu_count())
    else:
        num_threads = multiprocessing.cpu_count()

    reserve = 4  # main training process, NCCL, GPU driver, loggers, ...
    cap = 8  # above this can overwhelm the filesystem

    # can't go below 0, in that case the main training process will handle data loading
    num_workers = max(0, min(num_threads - reserve, cap))

    return num_workers


def fork_is_available() -> bool:
    """
    Whether the "fork" multiprocessing start method can be used for
    ``DataLoader`` workers.

    Restricted to Linux: fork is unsafe on macOS (system frameworks are not
    fork-safe) and unavailable on Windows. Checked against the list of
    available start methods rather than the default one, because the default
    changed from "fork" to "forkserver" on Linux in Python 3.14 while fork
    remains available; metatrain requests it explicitly when building
    ``DataLoader`` objects.

    :return: Whether "fork" can be used for ``DataLoader`` workers.
    """
    return (
        sys.platform.startswith("linux")
        and "fork" in multiprocessing.get_all_start_methods()
    )


def validate_num_workers(num_workers: int) -> None:
    """
    Gets a good number of workers for data loading.

    :param num_workers: The number of workers to validate.
    :raises ValueError: If the number of workers is greater than 0 and the
        "fork" multiprocessing start method is not available.
    """

    if num_workers > 0 and not fork_is_available():
        raise ValueError(
            "Parallel data loading relies on the 'fork' multiprocessing "
            "start method, which is only available on Linux. On this "
            "platform, num_workers must be set to 0."
        )


def _make_system_contiguous(system: System) -> System:
    """
    Return a copy of a ``System`` object with contiguous arrays.

    :param system: The system to make contiguous.
    :return: A copy of the system with contiguous arrays.
    """
    new_system = System(
        positions=system.positions.contiguous(),
        types=system.types.contiguous(),
        cell=system.cell.contiguous(),
        pbc=system.pbc.contiguous(),
    )
    for nl_options in system.known_neighbor_lists():
        nl = system.get_neighbor_list(nl_options)
        new_system.add_neighbor_list(
            nl_options,
            make_contiguous_block(nl),
        )
    for key in system.known_data():
        data = system.get_data(key)
        new_system.add_data(key, make_contiguous(data))
    return new_system


class MemmapArray:
    """
    Small helper to reopen a ``np.memmap`` lazily in each worker.

    :param path: Path to the binary file containing the memory-mapped array.
    :param shape: Shape of the array.
    :param dtype: Data type of the array.
    :param mode: Mode to open the array. See ``numpy.memmap`` for details.
    """

    def __init__(
        self,
        path: Union[str, Path],
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype],
        mode: str = "r",
    ) -> None:
        self.path = str(path)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self._mm = None

    def _ensure_open(self) -> None:
        if self._mm is None:
            self._mm = np.memmap(
                self.path, dtype=self.dtype, mode=self.mode, shape=self.shape
            )

    def __getitem__(self, idx: Any) -> np.ndarray:
        self._ensure_open()
        return self._mm[idx]  # type: ignore

    def close(self) -> None:
        if self._mm is not None:
            # np.memmap closes when deleted; explicit close via _mmap isn't public.
            self._mm._mmap.close()
            self._mm = None


class MemmapDataset(TorchDataset):
    """A class representing a dataset stored as a set of memory-mapped arrays.

    This dataset supports arbitrary scalar and cartesian vector/tensor targets, but
    not spherical tensors. Virials of energy targets are not supported in this type of
    dataset (stresses can be used instead to achieve the same goal).

    The dataset is stored in a directory, where the dataset is stored in a set of
    memory-mapped numpy arrays. These are:

    - ns.npy: total number of structures in the dataset. Shape: (), scalar.
    - na.npy: cumulative number of atoms per structure. na[-1] therefore corresponds to
        the total number of atoms in the dataset. Shape: (ns+1,). Must use ``int64``
        dtype (required for datasets with more than ~2 billion atoms).
    - x.bin: atomic positions of all atoms in the dataset, concatenated. Shape:
        (na[-1], 3).
    - a.bin: atomic types of all atoms in the dataset, concatenated. Shape: (na[-1],).
    - c.bin: cell matrices of all structures in the dataset. Shape: (ns, 3, 3).
    - <key>.bin: target values for each structure or atom, depending on the
        whether the target is defined per atom or per structure.
        Shape: (ns, ..., num_subtargets) if per-structures or
        (na[-1], ..., num_subtargets) if per-atom, where the
        ... depends on the type of target (scalar, vector, tensor, etc.). <key> can
        then be used in the "key" section of targets in metatrain input files to read
        the target(s).

    :param path: Path to the directory containing the dataset.
    :param target_options: Dictionary containing the target configurations, in the
        format corresponding to metatrain yaml input files.
    :param extra_data_options: Optional dictionary of extra per-system scalar arrays
        to load alongside targets. Keys are the metatensor-style names (e.g.
        ``"charge"``) and values are dicts with a ``"key"`` entry specifying
        the ``.bin`` filename stem. The data is returned as :py:class:`TensorMap`
        values in the sample namedtuple and forwarded to the ``extra`` argument of
        :py:class:`CollateFn` callables.
    """

    def __init__(
        self,
        path: Union[str, Path],
        target_options: Dict[str, Any],
        extra_data_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path)
        self.target_config = target_options
        self.extra_data_config: Dict[str, Any] = (
            extra_data_options if extra_data_options is not None else {}
        )
        overlapping = set(target_options.keys()) & set(self.extra_data_config.keys())
        if overlapping:
            raise ValueError(
                f"Extra data keys {overlapping} overlap with target keys. "
                "Please use unique keys for targets and extra data."
            )
        for key, opts in self.extra_data_config.items():
            if opts.get("type") != "scalar":
                raise ValueError(
                    f"Extra data key '{key}' has type '{opts.get('type')}'; "
                    "only 'scalar' is supported for MemmapDataset extra data."
                )
        self.sample_class = namedtuple(
            "Sample",
            ["system"]
            + list(self.target_config.keys())
            + list(self.extra_data_config.keys()),
        )

        # Information about the structures
        ns = np.load(path / "ns.npy")
        if ns.ndim != 0:
            raise ValueError(
                f"ns.npy must contain a scalar, got an array of shape {ns.shape}"
            )
        self.ns = int(ns)
        self.na = np.load(path / "na.npy")
        if self.na.dtype != np.int64:
            raise ValueError(
                f"na.npy must use int64 dtype, got {self.na.dtype}. "
                "This is required for datasets with more than ~2 billion atoms."
            )
        self.x = MemmapArray(path / "x.bin", (self.na[-1], 3), "float32", mode="r")
        self.a = MemmapArray(path / "a.bin", (self.na[-1],), "int32", mode="r")
        if os.path.exists(path / "c.bin"):
            self.c = MemmapArray(path / "c.bin", (self.ns, 3, 3), "float32", mode="r")
        self.momenta = None
        if os.path.exists(path / "momenta.bin"):  # for FlashMD
            self.momenta = MemmapArray(
                path / "momenta.bin", (self.na[-1], 3), "float32", mode="r"
            )
        self.masses = None
        if os.path.exists(path / "masses.bin"):
            self.masses = MemmapArray(
                path / "masses.bin", (self.na[-1],), "float32", mode="r"
            )

        # Register per-system (or per-atom) scalar arrays for extra_data keys
        self.extra_data_arrays: Dict[str, MemmapArray] = {}
        for key, opts in self.extra_data_config.items():
            n_samples = self.na[-1] if opts["sample_kind"] == "atom" else self.ns
            self.extra_data_arrays[key] = MemmapArray(
                path / f"{opts['key']}.bin",
                (n_samples, opts["num_subtargets"]),
                "float32",
                mode="r",
            )

        # Register arrays pointing to the targets
        self.target_arrays = {}
        for target_key, single_target_options in target_options.items():
            data_key = single_target_options["key"]
            per_atom = single_target_options["sample_kind"] == "atom"
            number_of_samples = self.na[-1] if per_atom else self.ns
            number_of_properties = single_target_options["num_subtargets"]
            if single_target_options["type"] == "scalar":
                self.target_arrays[target_key] = MemmapArray(
                    path / f"{data_key}.bin",
                    (number_of_samples, number_of_properties),
                    "float32",
                    mode="r",
                )
                if (
                    single_target_options["quantity"] == "energy"
                    and not per_atom
                    and single_target_options["num_subtargets"] == 1
                ):
                    # energy target: look into potential gradients
                    if single_target_options["forces"]:
                        self.target_arrays[f"{target_key}_forces"] = MemmapArray(
                            path / f"{single_target_options['forces']['key']}.bin",
                            (self.na[-1], 3, 1),
                            "float32",
                            mode="r",
                        )
                    if single_target_options["stress"]:
                        self.target_arrays[f"{target_key}_stress"] = MemmapArray(
                            path / f"{single_target_options['stress']['key']}.bin",
                            (self.ns, 3, 3, 1),
                            "float32",
                            mode="r",
                        )
                    if single_target_options["virial"]:
                        raise ValueError(
                            "Virial targets are not supported in MemmapDataset."
                        )
            elif isinstance(single_target_options["type"], DictConfig) or isinstance(
                single_target_options["type"], Dict
            ):
                if "spherical" in single_target_options["type"]:
                    raise ValueError(
                        "Spherical targets are not supported in MemmapDataset."
                    )
                else:  # cartesian
                    n_components = single_target_options["type"]["cartesian"]["rank"]
                    shape = (
                        (number_of_samples,)
                        + (3,) * n_components
                        + (number_of_properties,)
                    )
                    self.target_arrays[target_key] = MemmapArray(
                        path / f"{data_key}.bin", shape, "float32", mode="r"
                    )
            else:
                raise ValueError(
                    f"Unsupported target configuration: {single_target_options}"
                )

    def __len__(self) -> int:
        return self.ns

    def get_num_atoms(self, i: int) -> int:
        """Return the number of atoms in structure ``i`` without loading the full
        sample.

        :param i: The structure index.
        :return: The number of atoms in structure ``i``.
        """
        return int(self.na[i + 1] - self.na[i])

    def get_all_atom_counts(self) -> np.ndarray:
        """Return atom counts for all structures as a numpy int64 array.

        Much faster than calling ``get_num_atoms`` in a loop for large datasets.

        :return: Array of atom counts, one per structure.
        """
        return np.diff(self.na).astype(np.int64)

    def __getitem__(self, i: int) -> Any:
        a = torch.tensor(self.a[self.na[i] : self.na[i + 1]], dtype=torch.int32)
        x = torch.tensor(self.x[self.na[i] : self.na[i + 1]], dtype=torch.float64)
        momenta = None
        if self.momenta is not None:
            momenta = torch.tensor(
                self.momenta[self.na[i] : self.na[i + 1]], dtype=torch.float64
            )
        masses = None
        if self.masses is not None:
            masses = torch.tensor(
                self.masses[self.na[i] : self.na[i + 1]], dtype=torch.float64
            )
        if hasattr(self, "c"):
            c = torch.tensor(self.c[i], dtype=torch.float64)
        else:
            c = torch.zeros((3, 3), dtype=torch.float64)

        system = System(
            positions=x,
            types=a,
            cell=c,
            pbc=torch.logical_not(torch.all(c == 0.0, dim=1)),
        )

        # attach momenta to the system
        if momenta is not None:
            system.add_data(
                "momentum",
                TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=momenta.unsqueeze(-1),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.tensor(
                                    [
                                        [i, j]
                                        for j in range(self.na[i + 1] - self.na[i])
                                    ],
                                    dtype=torch.int32,
                                ),
                            ),
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range("momentum", 1),
                        )
                    ],
                ),
            )
        if masses is not None:
            system.add_data(
                "mass",
                TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=masses.unsqueeze(-1),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.tensor(
                                    [
                                        [i, j]
                                        for j in range(self.na[i + 1] - self.na[i])
                                    ],
                                    dtype=torch.int32,
                                ),
                            ),
                            components=[],
                            properties=Labels.single(),
                        )
                    ],
                ),
            )

        target_dict = {}
        for target_key, target_options in self.target_config.items():
            target_array = self.target_arrays[target_key]
            is_per_atom = target_array.shape[0] == (self.na[-1])
            if is_per_atom:
                samples = Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [[i, j] for j in range(self.na[i + 1] - self.na[i])],
                        dtype=torch.int32,
                    ),
                )
            else:
                samples = Labels(
                    names=["system"],
                    values=torch.tensor([[i]], dtype=torch.int32),
                )
            if len(target_array.shape) > 3:
                # Cartesian tensor with rank > 1
                n_components = len(target_array.shape) - 2
                components = [
                    Labels.range(f"xyz_{d + 1}", 3) for d in range(n_components)
                ]
            elif len(target_array.shape) == 3:
                # Cartesian vector
                components = [Labels.range("xyz", 3)]
            else:
                # Scalar
                components = []

            # Check if it is an energy target to set the correct property label name
            is_energy = (
                target_options["quantity"] == "energy"
                and target_options["sample_kind"] == "system"
                and target_options["num_subtargets"] == 1
            )

            target_block = TensorBlock(
                values=torch.tensor(
                    (
                        target_array[None, i]
                        if not is_per_atom
                        else target_array[self.na[i] : self.na[i + 1]]
                    ),
                    dtype=torch.float64,
                ),
                samples=samples,
                components=components,
                properties=Labels.range(
                    "energy" if is_energy else target_key.replace("mtt::", ""),
                    target_array.shape[-1],
                ),
            )

            # handle energy gradients
            if is_energy:
                if target_options["forces"]:
                    f = torch.tensor(
                        self.target_arrays[f"{target_key}_forces"][
                            self.na[i] : self.na[i + 1]
                        ],
                        dtype=torch.float64,
                    )
                    target_block.add_gradient(
                        "positions",
                        TensorBlock(
                            values=-f,
                            samples=Labels(
                                names=["sample", "atom"],
                                values=torch.tensor(
                                    [[0, j] for j in range(len(a))], dtype=torch.int32
                                ),
                            ),
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range("energy", 1),
                        ),
                    )
                if target_options["stress"]:
                    s = torch.tensor(
                        self.target_arrays[f"{target_key}_stress"][None, i],
                        dtype=torch.float64,
                    )
                    target_block.add_gradient(
                        "strain",
                        TensorBlock(
                            values=(s * torch.abs(torch.det(c))),
                            samples=Labels(
                                names=["sample"],
                                values=torch.tensor([[0]], dtype=torch.int32),
                            ),
                            components=[
                                Labels.range("xyz_1", 3),
                                Labels.range("xyz_2", 3),
                            ],
                            properties=Labels.range("energy", 1),
                        ),
                    )

            target_tensormap = TensorMap(
                keys=Labels.single(),
                blocks=[target_block],
            )
            target_dict[target_key] = target_tensormap

        # Build extra_data TensorMaps returned in the sample and forwarded to
        # the `extra` argument of CollateFn callables
        extra_data_dict = {}
        for key, arr in self.extra_data_arrays.items():
            is_per_atom = self.extra_data_config[key]["sample_kind"] == "atom"
            if is_per_atom:
                extra_samples = Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [[i, j] for j in range(self.na[i + 1] - self.na[i])],
                        dtype=torch.int32,
                    ),
                )
                extra_values = torch.tensor(
                    arr[self.na[i] : self.na[i + 1]], dtype=torch.float64
                )
            else:
                extra_samples = Labels("system", torch.tensor([[i]], dtype=torch.int32))
                extra_values = torch.tensor(arr[None, i], dtype=torch.float64)
            extra_data_dict[key] = TensorMap(
                keys=Labels.single(),
                blocks=[
                    TensorBlock(
                        values=extra_values,
                        samples=extra_samples,
                        components=[],
                        properties=Labels.range(
                            key.replace("mtt::", ""), arr.shape[-1]
                        ),
                    )
                ],
            )

        joint_dict = {"system": system}
        joint_dict.update(target_dict)
        joint_dict.update(extra_data_dict)
        sample = self.sample_class._make(
            [joint_dict[name] for name in self.sample_class._fields]
        )
        return sample

    def get_target_info(self) -> Dict[str, TargetInfo]:
        """
        Get information about the targets in the dataset.

        :return: A dictionary mapping target names to :py:class:`TargetInfo` objects.
        """
        target_info_dict = {}
        for target_key, target in self.target_config.items():
            is_energy = (
                (target["quantity"] == "energy")
                and target["sample_kind"] == "system"
                and target["num_subtargets"] == 1
                and target["type"] == "scalar"
            )
            tensor_map = self[0][target_key]
            if is_energy:
                if len(tensor_map) != 1:
                    raise ValueError("Energy TensorMaps should have exactly one block.")
                add_position_gradients = tensor_map.block().has_gradient("positions")
                add_strain_gradients = tensor_map.block().has_gradient("strain")
                target_info = get_energy_target_info(
                    target_key, target, add_position_gradients, add_strain_gradients
                )
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                target_info_dict[target_key] = target_info
            else:
                target_info = get_generic_target_info(target_key, target)
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                # make sure that the properties of the target_info.layout also match the
                # actual properties of the tensor maps
                target_info.layout = _empty_tensor_map_like(tensor_map)
                target_info_dict[target_key] = target_info
        return target_info_dict

    def get_extra_data_info(self) -> Dict[str, TargetInfo]:
        """
        Get information about the extra_data entries in the dataset.

        :return: A dictionary mapping extra_data keys to :py:class:`TargetInfo`
            objects describing each per-system scalar array.
        """
        extra_data_info_dict: Dict[str, TargetInfo] = {}
        if not self.extra_data_config:
            return extra_data_info_dict
        first_sample = self[0]
        for key, opts in self.extra_data_config.items():
            tensor_map = first_sample[key]
            target_info = get_generic_target_info(key, opts)
            target_info.layout = _empty_tensor_map_like(tensor_map)
            extra_data_info_dict[key] = target_info
        return extra_data_info_dict
