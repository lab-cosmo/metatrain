from pathlib import Path

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from omegaconf import OmegaConf

from metatrain.utils.data import (
    CollateFn,
    Dataset,
    DatasetInfo,
    TargetInfo,
    check_datasets,
    get_all_targets,
    get_atomic_types,
    get_stats,
    load_indices,
    read_extra_data,
    read_systems,
    read_targets,
    unpack_batch,
)
from metatrain.utils.data.dataset import MemmapDataset
from metatrain.utils.data.writers import MemmapWriter


RESOURCES_PATH = Path(__file__).parents[2] / "resources"


@pytest.fixture
def layout_scalar():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[],
                properties=Labels.range("energy", 1),
            )
        ],
    )


@pytest.fixture
def layout_spherical():
    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(0, 1, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
            TensorBlock(
                values=torch.empty(0, 5, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-2, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


@pytest.fixture
def layout_cartesian():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 3, 3, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["xyz_1"],
                        values=torch.arange(0, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                    Labels(
                        names=["xyz_2"],
                        values=torch.arange(0, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


def test_target_info_scalar(layout_scalar):
    target_info = TargetInfo(layout=layout_scalar, quantity="energy", unit="kcal/mol")

    assert target_info.quantity == "energy"
    assert target_info.unit == "kcal/mol"
    assert target_info.gradients == []
    assert target_info.sample_kind == "system"

    expected_start = f"TargetInfo(layout={layout_scalar}, quantity='energy', "
    assert target_info.__repr__()[: len(expected_start)] == expected_start


def test_target_info_spherical(layout_spherical):
    target_info = TargetInfo(
        layout=layout_spherical,
        quantity="mtt::spherical",
        unit="kcal/mol",
    )

    assert target_info.quantity == "mtt::spherical"
    assert target_info.unit == "kcal/mol"
    assert target_info.gradients == []
    assert target_info.sample_kind == "system"

    expected_start = f"TargetInfo(layout={layout_spherical}, quantity='mtt::spherical',"
    assert target_info.__repr__()[: len(expected_start)] == expected_start


def test_target_info_cartesian(layout_cartesian):
    target_info = TargetInfo(
        layout=layout_cartesian,
        quantity="mtt::cartesian",
        unit="kcal/mol",
    )

    assert target_info.quantity == "mtt::cartesian"
    assert target_info.unit == "kcal/mol"
    assert target_info.gradients == []
    assert target_info.sample_kind == "system"

    expected_start = f"TargetInfo(layout={layout_cartesian}, quantity='mtt::cartesian',"
    assert target_info.__repr__()[: len(expected_start)] == expected_start


def test_unit_none_conversion(layout_scalar):
    info = TargetInfo(
        layout=layout_scalar,
        quantity="energy",
        unit="",
    )
    assert info.unit == ""


def test_metadata_error(layout_scalar):
    with pytest.raises(ValueError, match="foo"):
        DatasetInfo(
            length_unit="foo",
            atomic_types=[1, 6, 8],
            targets={
                "energy": TargetInfo(
                    layout=layout_scalar,
                    quantity="energy",
                    unit="kcal/mol",
                )
            },
        )


def test_target_info_eq(layout_scalar):
    info1 = TargetInfo(layout=layout_scalar, quantity="energy", unit="kcal/mol")
    info2 = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")

    assert info1 == info1
    assert info1 != info2


def test_target_info_eq_other_objects(layout_scalar):
    info = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")

    assert not info == [1, 2, 3]


def test_dataset_info(layout_scalar):
    """Tests the DatasetInfo class."""

    targets = {
        "energy": TargetInfo(
            layout=layout_scalar,
            quantity="energy",
            unit="kcal/mol",
        )
    }
    targets["mtt::U0"] = TargetInfo(
        layout=layout_scalar,
        quantity="energy",
        unit="kcal/mol",
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[3, 1, 2], targets=targets
    )

    assert dataset_info.length_unit == "angstrom"
    assert dataset_info.atomic_types == [1, 2, 3]
    assert dataset_info.targets["energy"].quantity == "energy"
    assert dataset_info.targets["energy"].unit == "kcal/mol"
    assert dataset_info.targets["mtt::U0"].quantity == "energy"
    assert dataset_info.targets["mtt::U0"].unit == "kcal/mol"
    assert dataset_info.device == layout_scalar.device

    expected = (
        "DatasetInfo(length_unit='angstrom', atomic_types=[1, 2, 3], "
        f"targets={targets})"
    )
    assert dataset_info.__repr__() == expected


def test_set_atomic_types(layout_scalar):
    targets = {
        "energy": TargetInfo(layout=layout_scalar, quantity="energy", unit="kcal/mol")
    }
    targets["mtt::U0"] = TargetInfo(
        layout=layout_scalar,
        quantity="energy",
        unit="kcal/mol",
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[3, 1, 2], targets=targets
    )

    dataset_info.atomic_types = [5, 4, 1]
    assert dataset_info.atomic_types == [1, 4, 5]

    dataset_info.atomic_types += [7, 1]
    assert dataset_info.atomic_types == [1, 4, 5, 7]


def test_dataset_info_copy(layout_scalar, layout_cartesian):
    targets = {}
    targets["energy"] = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")
    targets["mtt::my-target"] = TargetInfo(
        layout=layout_cartesian, quantity="mtt::my-target", unit="eV/Angstrom"
    )
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    copy = info.copy()

    assert copy == info
    assert copy is not info


def test_dataset_info_update(layout_scalar, layout_spherical):
    targets = {}
    targets["energy"] = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = targets.copy()
    targets2["mtt::my-target"] = TargetInfo(
        layout=layout_spherical, quantity="mtt::my-target", unit="eV/Angstrom"
    )

    info2 = DatasetInfo(length_unit="angstrom", atomic_types=[8], targets=targets2)
    info.update(info2)

    assert info.atomic_types == [1, 6, 8]
    assert info.targets["energy"] == targets["energy"]
    assert info.targets["mtt::my-target"] == targets2["mtt::my-target"]


def test_dataset_info_update_non_matching_length_unit(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = targets.copy()
    targets2["mtt::my-target"] = TargetInfo(
        layout=layout_scalar, quantity="mtt::my-target", unit="eV/Angstrom"
    )

    info2 = DatasetInfo(length_unit="nanometer", atomic_types=[8], targets=targets2)

    match = (
        r"Can't update DatasetInfo with a different `length_unit`: "
        r"\('angstrom' != 'nanometer'\)"
    )

    with pytest.raises(ValueError, match=match):
        info.update(info2)


def test_dataset_info_eq(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = targets.copy()
    targets2["my-target"] = TargetInfo(
        layout=layout_scalar, quantity="mtt::my-target", unit="eV/Angstrom"
    )
    info2 = DatasetInfo(length_unit="nanometer", atomic_types=[8], targets=targets2)

    assert info == info
    assert info != info2


def test_dataset_info_eq_other_objects(layout_scalar):
    targets = {}
    targets["energy"] = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")

    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    assert not info == [1, 2, 3]


def test_dataset_info_update_different_target_info(layout_scalar):
    targets = {"energy": TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")}
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    targets2 = {
        "energy": TargetInfo(layout=layout_scalar, quantity="mtt::energy", unit="eV")
    }
    info2 = DatasetInfo(length_unit="angstrom", atomic_types=[8], targets=targets2)

    match = (
        "Can't update DatasetInfo with different target information for target 'energy'"
    )
    with pytest.raises(ValueError, match=match):
        info.update(info2)


def test_dataset_info_union(layout_scalar, layout_cartesian):
    """Tests the union method."""
    targets = {}
    targets["energy"] = TargetInfo(layout=layout_scalar, quantity="energy", unit="eV")
    targets["forces"] = TargetInfo(
        layout=layout_scalar, quantity="mtt::forces", unit="eV/Angstrom"
    )
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6], targets=targets)

    other_targets = targets.copy()
    other_targets["mtt::stress"] = TargetInfo(
        layout=layout_cartesian, quantity="mtt::stress", unit="GPa"
    )

    other_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1], targets=other_targets
    )

    union = info.union(other_info)

    assert union.length_unit == "angstrom"
    assert union.atomic_types == [1, 6]
    assert union.targets == other_targets


def test_dataset_info_no_targets():
    """Tests the properties of a DatasetInfo that has no targets."""
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1, 2, 3], targets={}
    )

    assert dataset_info.device is None
    # Setting the device should not fail:
    dataset_info.to(device="cpu")


def test_dataset():
    """Tests the readers and the dataset class."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    filename = str(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": filename,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    collate_fn = CollateFn(target_info_dict)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )

    for batch in dataloader:
        batch = unpack_batch(batch)
        assert batch[1]["energy"].block().values.shape == (10, 1)


def test_get_atomic_types():
    """Tests that the species list is correctly computed with get_atomic_types."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset.from_dict({"system": systems, **targets})
    dataset_2 = Dataset.from_dict({"system": systems_2, **targets_2})

    assert get_atomic_types(dataset) == [1, 6, 7, 8]
    assert get_atomic_types(dataset_2) == [1, 6, 8]
    assert get_atomic_types([dataset, dataset_2]) == [1, 6, 7, 8]


def test_get_all_targets():
    """Tests that the target list is correctly computed with get_all_targets."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset.from_dict({"system": systems, **targets})
    dataset_2 = Dataset.from_dict({"system": systems_2, **targets_2})
    assert get_all_targets(dataset) == ["mtt::U0"]
    assert get_all_targets(dataset_2) == ["energy"]
    assert get_all_targets([dataset, dataset_2]) == ["energy", "mtt::U0"]


def test_check_datasets():
    """Tests the check_datasets function."""

    systems_qm9 = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf_qm9 = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_ethanol = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_ethanol = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets_qm9, _ = read_targets(OmegaConf.create(conf_qm9))
    targets_ethanol, _ = read_targets(OmegaConf.create(conf_ethanol))

    # everything ok
    train_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    val_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    check_datasets([train_set], [val_set])

    # extra species in validation dataset
    train_set = Dataset.from_dict({"system": systems_ethanol, **targets_qm9})
    val_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    with pytest.raises(ValueError, match="The validation dataset has a species"):
        check_datasets([train_set], [val_set])

    # extra targets in validation dataset
    train_set = Dataset.from_dict({"system": systems_qm9, **targets_qm9})
    val_set = Dataset.from_dict({"system": systems_qm9, **targets_ethanol})
    with pytest.raises(ValueError, match="The validation dataset has a target"):
        check_datasets([train_set], [val_set])

    # wrong dtype
    systems_qm9_32bit = [system.to(dtype=torch.float32) for system in systems_qm9]
    targets_qm9_32bit = {
        name: [tensor.to(dtype=torch.float32) for tensor in values]
        for name, values in targets_qm9.items()
    }
    train_set_32_bit = Dataset.from_dict(
        {"system": systems_qm9_32bit, **targets_qm9_32bit}
    )

    match = (
        "`dtype` between datasets is inconsistent, found torch.float64 and "
        "torch.float32 in validation datasets"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([train_set], [train_set_32_bit])

    match = (
        "`dtype` between datasets is inconsistent, found torch.float64 and "
        "torch.float32 in training datasets"
    )
    with pytest.raises(TypeError, match=match):
        check_datasets([train_set, train_set_32_bit], [])


def test_collate_fn():
    """Tests the collate_fn function."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf_targets = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf_targets))

    conf_extra_data = {
        "U0": {
            "quantity": "",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
        }
    }
    extra_data, _ = read_extra_data(OmegaConf.create(conf_extra_data))

    dataset = Dataset.from_dict(
        {
            "system": systems,
            "mtt::U0": targets["mtt::U0"],
            "U0": extra_data["U0"],
        }
    )

    collate_fn = CollateFn(target_info_dict)
    batch = collate_fn([dataset[0], dataset[1], dataset[2]])
    batch = unpack_batch(batch)

    assert len(batch) == 3
    assert isinstance(batch[0], list)
    assert len(batch[0]) == 3
    assert isinstance(batch[1], dict)
    assert isinstance(batch[2], dict)


def test_get_stats(layout_scalar):
    """Tests the get_stats method of Dataset and Subset."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    targets_2, _ = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset.from_dict({"system": systems, **targets})
    dataset_2 = Dataset.from_dict({"system": systems_2, **targets_2})

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            "mtt::U0": TargetInfo(layout=layout_scalar, quantity="energy", unit="eV"),
            "energy": TargetInfo(layout=layout_scalar, quantity="energy", unit="eV"),
        },
    )

    stats = get_stats(dataset, dataset_info)
    stats_2 = get_stats(dataset_2, dataset_info)

    assert "100 structures" in stats
    assert "mtt::U0" in stats
    assert "energy" in stats_2
    assert "mean " in stats
    assert "std " in stats
    assert "mean " in stats_2
    assert "std " in stats_2
    assert "stress" not in stats_2
    assert "eV" in stats
    assert "eV" in stats_2


def test_instance_torchscript_compatible(layout_scalar):
    dataset_info = DatasetInfo(
        length_unit="",
        atomic_types=[1, 2, 3],
        targets={
            "energy": TargetInfo(
                layout=layout_scalar,
                quantity="energy",
                unit="kcal/mol",
            )
        },
    )

    torch.jit.script(dataset_info)


def test_memmap_per_atom_labels_use_local_indices(tmp_path):
    """Per-atom target samples must use local (0-based) atom indices, not global
    cumulative offsets. Global offsets overflow int32 for datasets with >2.1B atoms."""
    # 2 systems: system 0 has 2 atoms, system 1 has 3 atoms
    ns = 2
    na = np.array([0, 2, 5], dtype=np.int64)  # cumulative atom counts
    np.save(tmp_path / "ns.npy", ns)
    np.save(tmp_path / "na.npy", na)
    # positions, types, cells
    np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype="float32"
    ).tofile(tmp_path / "x.bin")
    np.array([1, 1, 6, 6, 8], dtype="int32").tofile(tmp_path / "a.bin")
    np.zeros((ns, 3, 3), dtype="float32").tofile(tmp_path / "c.bin")
    # per-atom scalar target: values 0..4 so system 1's atoms have values [2, 3, 4]
    np.arange(5, dtype="float32").reshape(5, 1).tofile(tmp_path / "charge.bin")

    target_options = {
        "atomic_charge": {
            "key": "charge",
            "sample_kind": "atom",
            "num_subtargets": 1,
            "type": "scalar",
            "quantity": "energy",
        }
    }
    dataset = MemmapDataset(tmp_path, target_options)

    # Load system 1 (the second structure, global atom offsets 2..4)
    sample = dataset[1]
    block = sample.atomic_charge.block()

    # Labels: atom dimension must be LOCAL [0, 1, 2], not global [2, 3, 4]
    atom_labels = block.samples.values[:, 1].tolist()
    assert atom_labels == [0, 1, 2], (
        f"Expected local atom indices [0, 1, 2], got {atom_labels}. "
        f"Global offsets would overflow int32 for large datasets."
    )

    values = block.values.squeeze(-1).tolist()
    assert values == [2.0, 3.0, 4.0], (
        f"Expected per-atom values [2.0, 3.0, 4.0] for system 1, got {values}."
    )


@pytest.mark.parametrize("bad_dtype", [np.int32, np.uint64, np.float64])
def test_memmap_rejects_non_int64_na(tmp_path, bad_dtype):
    """na.npy must be int64; int32 (overflow risk), uint64, float64 are all rejected."""
    ns = 1
    na = np.array([0, 3], dtype=bad_dtype)
    np.save(tmp_path / "ns.npy", ns)
    np.save(tmp_path / "na.npy", na)
    np.zeros((3, 3), dtype="float32").tofile(tmp_path / "x.bin")
    np.array([1, 1, 1], dtype="int32").tofile(tmp_path / "a.bin")
    np.zeros((1, 3, 3), dtype="float32").tofile(tmp_path / "c.bin")
    np.zeros((1, 1), dtype="float32").tofile(tmp_path / "e.bin")

    target_options = {
        "energy": {
            "key": "e",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "quantity": "energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    with pytest.raises(ValueError, match="int64 dtype"):
        MemmapDataset(tmp_path, target_options)


# ============================================================
# Helpers shared by MemmapDataset extra_data tests
# ============================================================


def _write_minimal_memmap(tmp_path, ns=3, values_per_system=None):
    """Write the minimum binary files needed to construct a MemmapDataset.

    Returns ``(target_options, energy_values)`` so callers can assert on targets.
    Each system has exactly 1 atom.

    *values_per_system* – list of length ``ns`` of per-system energy floats.
    If *None* defaults to [1.0, 2.0, 3.0, ...].
    """
    if values_per_system is None:
        values_per_system = list(range(1, ns + 1))

    na = np.array(
        [0] + list(range(1, ns + 1)), dtype=np.int64
    )  # cumulative atom counts, 1 atom per system
    np.save(tmp_path / "ns.npy", ns)
    np.save(tmp_path / "na.npy", na)
    np.zeros((ns, 3), dtype="float32").tofile(tmp_path / "x.bin")  # positions
    np.ones((ns,), dtype="int32").tofile(tmp_path / "a.bin")  # atom types (H)
    np.zeros((ns, 3, 3), dtype="float32").tofile(tmp_path / "c.bin")  # cells
    np.array(values_per_system, dtype="float32").reshape(ns, 1).tofile(
        tmp_path / "e.bin"
    )

    target_options = {
        "energy": {
            "key": "e",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "quantity": "energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    return target_options, values_per_system


def test_memmap_uses_pbc_file(tmp_path):
    target_options, _ = _write_minimal_memmap(tmp_path, ns=2)

    # a slab and a structure stored with a cell but no periodicity
    cells = np.array(
        [
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 100.0]],
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ],
        dtype="float32",
    )
    cells.tofile(tmp_path / "c.bin")
    np.save(
        tmp_path / "pbc.npy",
        np.array([[True, True, False], [False, False, False]]),
    )

    dataset = MemmapDataset(tmp_path, target_options)

    slab = dataset[0].system
    assert slab.pbc.tolist() == [True, True, False]
    assert slab.cell[2].tolist() == [0.0, 0.0, 0.0]
    assert slab.cell[0].tolist() == [3.0, 0.0, 0.0]

    isolated = dataset[1].system
    assert isolated.pbc.tolist() == [False, False, False]
    assert torch.all(isolated.cell == 0.0)


def test_memmap_infers_pbc_without_pbc_file(tmp_path):
    """without pbc.npy, periodicity is derived from the non-zero cell vectors"""
    target_options, _ = _write_minimal_memmap(tmp_path, ns=1)
    np.array(
        [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]], dtype="float32"
    ).tofile(tmp_path / "c.bin")

    dataset = MemmapDataset(tmp_path, target_options)

    assert dataset[0].system.pbc.tolist() == [True, True, False]


# ============================================================
# MemmapDataset extra_data tests
# ============================================================


def test_memmap_extra_data_values_in_sample(tmp_path):
    """extra_data array values are returned as TensorMaps in the sample namedtuple."""
    target_options, _ = _write_minimal_memmap(tmp_path)

    charge_values = [10.0, 20.0, 30.0]
    np.array(charge_values, dtype="float32").tofile(tmp_path / "charge.bin")

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    for i, expected in enumerate(charge_values):
        sample = dataset[i]
        tm = sample._asdict()["charge"]
        assert tm.block().values.item() == pytest.approx(expected)


def test_memmap_extra_data_system_label(tmp_path):
    """The sample label in the extra_data TensorMap matches the system index."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([0.0, 1.0, 2.0], dtype="float32").tofile(tmp_path / "feat.bin")

    extra_data_options = {
        "mtt::feat": {
            "key": "feat",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    for i in range(3):
        sample = dataset[i]
        tm = sample._asdict()["mtt::feat"]
        system_label = tm.block().samples["system"].tolist()
        assert system_label == [i]


def test_memmap_extra_data_property_name_from_key(tmp_path):
    """Properties label name is the extra_data key with the 'mtt::' prefix stripped."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "charge.bin")

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    tm = dataset[0]._asdict()["charge"]
    prop_name = tm.block().properties.names[0]
    assert prop_name == "charge"


def test_memmap_extra_data_no_options_empty(tmp_path):
    """MemmapDataset without extra_data_options has no extra fields in sample."""
    target_options, _ = _write_minimal_memmap(tmp_path)

    dataset = MemmapDataset(tmp_path, target_options)

    sample = dataset[0]
    assert set(sample._fields) == {"system", "energy"}


def test_memmap_extra_data_fields_present_in_sample(tmp_path):
    """extra_data keys appear as named fields in the sample namedtuple."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "charge.bin")

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    assert "charge" in dataset[0]._fields


def test_memmap_get_extra_data_info_returns_target_info(tmp_path):
    """get_extra_data_info() returns a TargetInfo for each extra_data key."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "charge.bin")

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
            "unit": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    info = dataset.get_extra_data_info()

    assert "charge" in info
    assert isinstance(info["charge"], TargetInfo)


def test_memmap_get_extra_data_info_empty_without_options(tmp_path):
    """get_extra_data_info() returns an empty dict when no extra_data_options given."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    dataset = MemmapDataset(tmp_path, target_options)
    assert dataset.get_extra_data_info() == {}


def test_memmap_extra_data_multiple_keys(tmp_path):
    """Multiple extra_data keys are all loaded and accessible."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "charge.bin")
    np.array([4.0, 5.0, 6.0], dtype="float32").tofile(
        tmp_path / "spin_multiplicity.bin"
    )

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
        "spin_multiplicity": {
            "key": "spin_multiplicity",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    sample = dataset[1]
    fields = sample._asdict()
    assert fields["charge"].block().values.item() == pytest.approx(2.0)
    assert fields["spin_multiplicity"].block().values.item() == pytest.approx(5.0)


def test_memmap_extra_data_overlapping_key_raises(tmp_path):
    """Passing an extra_data key that duplicates a target key raises ValueError."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "e.bin")

    extra_data_options = {
        "energy": {
            "key": "e",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    with pytest.raises(ValueError, match="overlap with target keys"):
        MemmapDataset(tmp_path, target_options, extra_data_options)


def test_memmap_extra_data_non_scalar_type_raises(tmp_path):
    """Passing an extra_data entry with type != 'scalar' raises ValueError."""
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "charge.bin")

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": {"cartesian": {"rank": 1}},
            "sample_kind": "system",
            "num_subtargets": 3,
            "quantity": "",
        },
    }
    with pytest.raises(ValueError, match="only 'scalar' is supported"):
        MemmapDataset(tmp_path, target_options, extra_data_options)


def test_memmap_extra_data_mtt_prefix_accessible(tmp_path):
    """mtt:: prefixed keys work: string-key access and field presence on the sample.

    metatensor's custom namedtuple (unlike collections.namedtuple) accepts field
    names containing '::' and supports sample["charge"] string-key access —
    the same behaviour used by regular targets with mtt:: names.
    """
    target_options, _ = _write_minimal_memmap(tmp_path)
    np.array([1.0, 2.0, 3.0], dtype="float32").tofile(tmp_path / "charge.bin")

    extra_data_options = {
        "charge": {
            "key": "charge",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)
    sample = dataset[0]

    # field is present under the full mtt:: name
    assert "charge" in sample._fields
    # string-key access works (metatensor namedtuple feature)
    assert sample["charge"].block().values.item() == pytest.approx(1.0)


def _write_heterogeneous_memmap(tmp_path, atoms_per_system):
    """Write a MemmapDataset with varying atom counts per system.

    *atoms_per_system* – list of int, e.g. [1, 3, 2].
    Returns target_options.
    """
    ns = len(atoms_per_system)
    na = np.array([0] + list(np.cumsum(atoms_per_system)), dtype=np.int64)
    total_atoms = int(na[-1])

    np.save(tmp_path / "ns.npy", ns)
    np.save(tmp_path / "na.npy", na)
    np.zeros((total_atoms, 3), dtype="float32").tofile(tmp_path / "x.bin")
    np.ones((total_atoms,), dtype="int32").tofile(tmp_path / "a.bin")
    np.zeros((ns, 3, 3), dtype="float32").tofile(tmp_path / "c.bin")
    # energy: one value per system
    np.arange(1, ns + 1, dtype="float32").reshape(ns, 1).tofile(tmp_path / "e.bin")

    target_options = {
        "energy": {
            "key": "e",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "quantity": "energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    return target_options, na


def test_memmap_extra_data_per_atom_heterogeneous(tmp_path):
    """Per-atom extra_data with heterogeneous atom counts returns the right atoms.

    Systems have 1, 3, 2 atoms respectively (total 6).  The per-atom array is
    written flat (all atoms concatenated) and __getitem__ must slice the correct
    rows for each system using the cumulative na index.
    """
    atoms_per_system = [1, 3, 2]
    target_options, na = _write_heterogeneous_memmap(tmp_path, atoms_per_system)

    # per-atom feature: one row per atom, flat across all systems
    # system 0 → atom 0         → value 10.0
    # system 1 → atoms 1,2,3   → values 20.0, 21.0, 22.0
    # system 2 → atoms 4,5     → values 30.0, 31.0
    per_atom_values = np.array([10.0, 20.0, 21.0, 22.0, 30.0, 31.0], dtype="float32")
    per_atom_values.tofile(tmp_path / "feat.bin")

    extra_data_options = {
        "mtt::feat": {
            "key": "feat",
            "type": "scalar",
            "sample_kind": "atom",
            "num_subtargets": 1,
            "quantity": "",
        },
    }
    dataset = MemmapDataset(tmp_path, target_options, extra_data_options)

    # system 0: 1 atom
    s0 = dataset[0]["mtt::feat"]
    assert s0.block().samples.column("atom").tolist() == [0]
    assert s0.block().values.flatten().tolist() == pytest.approx([10.0])

    # system 1: 3 atoms
    s1 = dataset[1]["mtt::feat"]
    assert s1.block().samples.column("atom").tolist() == [0, 1, 2]
    assert s1.block().values.flatten().tolist() == pytest.approx([20.0, 21.0, 22.0])

    # system 2: 2 atoms
    s2 = dataset[2]["mtt::feat"]
    assert s2.block().samples.column("atom").tolist() == [0, 1]
    assert s2.block().values.flatten().tolist() == pytest.approx([30.0, 31.0])


# ============================================================================
# Tests for load_indices
# ============================================================================


def test_load_indices_from_list():
    """Load indices directly from a Python list."""
    indices = load_indices([0, 5, 10, 15])
    assert indices == [0, 5, 10, 15]


def test_load_indices_from_file(tmp_path):
    """Load indices from a text file with one index per line."""
    idx_file = tmp_path / "indices.txt"
    idx_file.write_text("0\n5\n10\n15\n")
    indices = load_indices(str(idx_file))
    assert indices == [0, 5, 10, 15]


def test_load_indices_from_file_with_blanks(tmp_path):
    """Blank lines in indices file should be ignored."""
    idx_file = tmp_path / "indices.txt"
    idx_file.write_text("0\n\n5\n  \n10\n")
    indices = load_indices(str(idx_file))
    assert indices == [0, 5, 10]


def test_load_indices_relative_path(tmp_path, monkeypatch):
    """Test that relative paths are resolved against the cwd"""
    monkeypatch.chdir(tmp_path)
    idx_file = tmp_path / "indices.txt"
    idx_file.write_text("1\n2\n3\n")
    indices = load_indices("indices.txt")
    assert indices == [1, 2, 3]


def test_load_indices_empty_list():
    """Empty indices list"""
    indices = load_indices([])
    assert indices == []


def test_load_indices_empty_file(tmp_path):
    """Empty indices file raises numpy warning."""
    idx_file = tmp_path / "indices.txt"
    idx_file.write_text("")
    with pytest.warns(UserWarning, match="loadtxt: input contained no data"):
        indices = load_indices(str(idx_file))
    assert indices == []


def test_load_indices_negative_raises():
    """Negative indices raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        load_indices([0, -1, 2])


def test_load_indices_file_not_found(tmp_path):
    """Missing file raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        load_indices("nonexistent.txt")


# ============================================================
# MemmapDataset FlashMD momenta and masses tests
# ============================================================


def _write_memmap_via_writer(
    tmp_path, atoms_per_system, energy_values, momenta=None, masses=None
):
    """Write a MemmapDataset directory using the ``MemmapWriter``.

    :param atoms_per_system: number of atoms per system.
    :param energy_values: one energy value per system.
    :param momenta: optional (n_atoms, 3) array/list of per-atom momenta.
    :param masses: optional (n_atoms,) array/list of per-atom masses.
    """
    systems = [
        System(
            types=torch.ones(n, dtype=torch.int32),
            positions=torch.zeros(n, 3),
            cell=torch.zeros(3, 3),
            pbc=torch.zeros(3, dtype=torch.bool),
        )
        for n in atoms_per_system
    ]

    n_systems = len(systems)
    predictions = {
        "e": TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor(energy_values, dtype=torch.float64).reshape(
                        n_systems, 1
                    ),
                    samples=Labels("system", torch.arange(n_systems).reshape(-1, 1)),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
            ],
        )
    }

    system_atom_labels = torch.tensor(
        [[i, j] for i, n in enumerate(atoms_per_system) for j in range(n)]
    )
    if momenta is not None:
        predictions["momenta"] = TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor(momenta, dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(["system", "atom"], system_atom_labels),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("momentum", 1),
                )
            ],
        )
    if masses is not None:
        predictions["masses"] = TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor(masses, dtype=torch.float64).reshape(-1, 1),
                    samples=Labels(["system", "atom"], system_atom_labels),
                    components=[],
                    properties=Labels.range("mass", 1),
                )
            ],
        )

    writer = MemmapWriter(tmp_path)
    writer.write(systems, predictions)
    writer.finish()

    target_options = {
        "energy": {
            "key": "e",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "quantity": "energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    return target_options


def test_memmap_momenta_attached(tmp_path):
    """Momenta need to be attached and can be loaded."""
    # FlashMD stores the current momenta of every atom (1 atom per system here)
    target_options = _write_memmap_via_writer(
        tmp_path,
        atoms_per_system=[1],
        energy_values=[1.0],
        momenta=[[1.0, 2.0, 3.0]],
    )

    dataset = MemmapDataset(tmp_path, target_options)

    sample = dataset[0]  # used to raise ValueError

    assert sample.system.known_data() == ["momentum"], (
        f"momenta must be registered exactly once, got {sample.system.known_data()}"
    )
    values = sample.system.get_data("momentum").block().values
    assert values.squeeze(-1).tolist() == [[1.0, 2.0, 3.0]]


def test_memmap_momenta_attached_in_batch(tmp_path):
    """
    Every system in a collated batch must carry its own momenta exactly once,
    with local (0-based) atom labels and its own slice of ``momenta.bin``.
    """
    # 2 systems: system 0 has 2 atoms, system 1 has 3 atoms
    # per-atom momenta; the x component encodes the global atom index 0..4
    momenta = np.zeros((5, 3))
    momenta[:, 0] = np.arange(5)
    target_options = _write_memmap_via_writer(
        tmp_path,
        atoms_per_system=[2, 3],
        energy_values=[1.0, 2.0],
        momenta=momenta,
    )

    dataset = MemmapDataset(tmp_path, target_options)
    collate_fn = CollateFn(list(target_options.keys()))

    systems, _, _ = unpack_batch(collate_fn([dataset[0], dataset[1]]))

    assert len(systems) == 2
    # system 0 owns atoms 0-1, system 1 owns atoms 2-4; labels stay local
    expected = [([0, 1], [0.0, 1.0]), ([0, 1, 2], [2.0, 3.0, 4.0])]
    for system, (atom_labels, x_components) in zip(systems, expected, strict=True):
        assert system.known_data() == ["momentum"], (
            f"momenta must be registered exactly once, got {system.known_data()}"
        )
        block = system.get_data("momentum").block()
        assert block.samples.values[:, 1].tolist() == atom_labels
        assert block.values.squeeze(-1)[:, 0].tolist() == x_components


def test_memmap_masses_attached(tmp_path):
    """
    Masses must be attached with local (0-based) atom labels and the system's own
    slice of ``masses.bin``.
    """
    # FlashMD stores the mass of every atom (1 atom per system here)
    target_options = _write_memmap_via_writer(
        tmp_path,
        atoms_per_system=[1, 1],
        energy_values=[1.0, 2.0],
        masses=[1.0, 12.0],
    )

    dataset = MemmapDataset(tmp_path, target_options)

    system = dataset[1].system

    assert system.known_data() == ["mass"]
    block = system.get_data("mass").block()
    # the atom label is local (0), not the global offset (1)
    assert block.samples.values.tolist() == [[1, 0]]
    assert block.values.squeeze(-1).tolist() == [12.0]
