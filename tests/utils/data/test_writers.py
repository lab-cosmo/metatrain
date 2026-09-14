import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, ModelOutput, System

from metatrain.utils.data.dataset import DiskDataset, MemmapDataset
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.writers import (
    ASEWriter,
    DiskDatasetWriter,
    MemmapWriter,
    MetatensorWriter,
    get_writer,
)


def _make_system(n_atoms: int) -> System:
    return System(
        positions=torch.zeros((n_atoms, 3), dtype=torch.float64),
        types=torch.ones(n_atoms, dtype=torch.int32),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
    )


def systems_capabilities_predictions(
    cell: torch.tensor = None,
) -> Tuple[List[System], ModelCapabilities, Dict[str, TensorMap]]:
    if cell is None:
        cell = torch.zeros((3, 3))

    systems = 2 * [
        System(
            types=torch.tensor([1, 1]),
            positions=torch.tensor([[0, 0, 0], [0, 0, 0.74]]),
            cell=cell,
            pbc=torch.logical_not(torch.all(cell == 0, dim=1)),
        ),
    ]

    # Create a mock TensorMap for predictions
    n_systems = len(systems)
    values = torch.tensor([[1.0], [2.0]])
    block = TensorBlock(
        values=values.reshape(-1, 1),
        samples=Labels(["system"], torch.arange(n_systems).reshape(-1, 1)),
        components=[],
        properties=Labels(["energy"], torch.tensor([(0,)])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.arange(12, dtype=torch.get_default_dtype()).reshape(4, 3, 1),
            samples=Labels(
                ["sample", "atom"], torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels(["energy"], torch.tensor([(0,)])),
        ),
    )
    if not torch.all(cell == 0):
        block.add_gradient(
            "strain",
            TensorBlock(
                values=torch.arange(18, dtype=torch.get_default_dtype()).reshape(
                    2, 3, 3, 1
                ),
                samples=Labels(["sample"], torch.tensor([0, 1]).reshape(-1, 1)),
                components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                properties=Labels(["energy"], torch.tensor([(0,)])),
            ),
        )

    predictions = {"energy": TensorMap(Labels.single(), [block])}

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(quantity="energy", unit="kcal/mol")},
        interaction_range=1.0,
        dtype="float32",
    )

    return systems, capabilities, predictions


def test_ASEWriter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    assert len(frames) == len(systems)
    for i, atoms in enumerate(frames):
        assert atoms.info["energy"] == float(predictions["energy"].block().values[i, 0])
        assert all(atoms.pbc == 3 * [False])


def test_write_components_and_properties_xyz(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, _, _ = systems_capabilities_predictions()

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(quantity="dos", unit="")},
        interaction_range=1.0,
        dtype="float32",
    )

    predictions = {
        "dos": TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.rand(2, 3, 100),
                    samples=Labels(["system"], torch.tensor([[0], [1]])),
                    components=[
                        Labels.range("xyz", 3),
                    ],
                    properties=Labels(
                        ["property"],
                        torch.arange(100, dtype=torch.int32).reshape(-1, 1),
                    ),
                )
            ],
        )
    }

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    assert len(frames) == len(systems)
    for atoms in frames:
        assert atoms.info["dos"].shape == (3, 100)


def test_write_components_and_properties_xyz_per_atom(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, _, _ = systems_capabilities_predictions()

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(quantity="dos", unit="", sample_kind="atom")},
        interaction_range=1.0,
        dtype="float32",
    )

    predictions = {
        "dos": TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.rand(4, 3, 100),
                    samples=Labels(
                        ["system", "atom"],
                        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
                    ),
                    components=[
                        Labels.range("xyz", 3),
                    ],
                    properties=Labels(
                        ["property"],
                        torch.arange(100, dtype=torch.int32).reshape(-1, 1),
                    ),
                )
            ],
        )
    }

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    assert len(frames) == len(systems)
    for atoms in frames:
        assert atoms.arrays["dos"].shape == (2, 300)


def test_write_xyz_cell(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    cell_expected = torch.eye(3)
    systems, capabilities, predictions = systems_capabilities_predictions(
        cell=cell_expected
    )

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    for i, atoms in enumerate(frames):
        cell_actual = torch.tensor(atoms.cell.array, dtype=cell_expected.dtype)
        torch.testing.assert_close(cell_actual, cell_expected)
        assert all(atoms.pbc == 3 * [True])
        assert atoms.info["energy"] == float(predictions["energy"].block().values[i, 0])
        assert atoms.arrays["forces"].shape == (2, 3)
        assert atoms.info["stress"].shape == (3, 3)
        assert atoms.info["virial"].shape == (3, 3)


@pytest.mark.parametrize(
    "filename",
    (
        "test_output.xyz",
        "test_output.extxyz",
        "test_output.mts",
        "test_output.zip",
        "test_output/",
    ),
)
@pytest.mark.parametrize("fileformat", (None, "same_as_filename"))
@pytest.mark.parametrize("cell", (None, torch.eye(3)))
def test_write_predictions(filename, fileformat, cell, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions(cell=cell)

    if filename.endswith("/"):
        # memmap datasets are dispatched via a trailing path separator, not a
        # fileformat/extension
        writer = get_writer(filename, capabilities=capabilities)
    else:
        if fileformat == "same_as_filename" or fileformat is None:
            fileformat = "." + filename.split(".")[1]
        try:
            writer = get_writer(filename, capabilities=capabilities)
        except KeyError:
            raise ValueError(f"fileformat '{fileformat}' is not supported")

    writer.write(systems, predictions)
    writer.finish()

    if filename.endswith((".xyz", ".extxyz")):
        frames = read(filename, index=":")
        assert len(frames) == len(systems)
        for i, frame in enumerate(frames):
            assert frame.info["energy"] == float(
                predictions["energy"].block().values[i, 0]
            )
            assert frame.arrays["forces"].shape == (2, 3)
            if cell is not None:
                assert frame.info["stress"].shape == (3, 3)

    elif filename.endswith(".mts"):
        tensormap = mts.load(filename.split(".")[0] + "_energy.mts")
        assert tensormap.block().values.shape == (2, 1)
        assert tensormap.block().gradient("positions").values.shape == (4, 3, 1)
        if cell is not None:
            assert tensormap.block().gradient("strain").values.shape == (2, 3, 3, 1)

    elif filename.endswith("/"):
        directory = Path(filename)
        ns = int(np.load(directory / "ns.npy"))
        na = np.load(directory / "na.npy")
        assert ns == len(systems)
        assert na.tolist() == [0, 2, 4]

        x = np.fromfile(directory / "x.bin", dtype=np.float32).reshape(-1, 3)
        assert x.shape == (4, 3)

        a = np.fromfile(directory / "a.bin", dtype=np.int32)
        assert a.shape == (4,)

        c = np.fromfile(directory / "c.bin", dtype=np.float32).reshape(-1, 3, 3)
        assert c.shape == (2, 3, 3)

        energy = np.fromfile(directory / "energy.bin", dtype=np.float32).reshape(2, 1)
        np.testing.assert_allclose(
            energy, predictions["energy"].block().values.numpy(), atol=1e-6
        )

        forces = np.fromfile(directory / "energy_forces.bin", dtype=np.float32).reshape(
            4, 3, 1
        )
        expected_forces = -(
            predictions["energy"].block().gradient("positions").values.numpy()
        )
        np.testing.assert_allclose(forces, expected_forces, atol=1e-6)

        stress_path = directory / "energy_stress.bin"
        if cell is not None:
            assert stress_path.is_file()
        else:
            assert not stress_path.exists()

    else:
        ValueError("This test only does `.xyz` and `.mts`")


def test_write_predictions_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        get_writer("foo.bar", capabilities=None)


def test_write_predictions_no_extension_and_no_trailing_slash_errors():
    """A bare path with neither a recognized extension nor a trailing path separator
    must fail clearly, not silently guess it's a memmap directory."""
    with pytest.raises(ValueError, match="trailing path separator"):
        get_writer("predictions", capabilities=None)


def test_get_writer_trailing_slash_selects_memmap_writer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    writer = get_writer("predictions/", capabilities=None)
    assert isinstance(writer, MemmapWriter)
    assert writer.directory == Path("predictions")


@pytest.mark.parametrize("suffix", (".zip", ".mts", ".xyz"))
def test_get_writer_ambiguous_extension_and_trailing_slash_errors(suffix):
    """A path with both a recognized extension and a trailing separator (e.g.
    'predictions.zip/') must fail loudly rather than silently picking one
    interpretation, since it's ambiguous whether a file or a memmap directory
    was intended."""
    with pytest.raises(ValueError, match="is ambiguous"):
        get_writer(f"predictions{suffix}/", capabilities=None)


def test_write_disk_dataset_non_contiguous(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    systems, _, _ = systems_capabilities_predictions(cell=None)

    # create a non-contiguous TensorMap
    vals = torch.rand(100, 2, 3)
    non_contig_vals = vals.permute(1, 2, 0)
    assert not non_contig_vals.is_contiguous()
    block = TensorBlock(
        values=non_contig_vals,
        samples=Labels(["system"], torch.tensor([[0], [1]])),
        components=[
            Labels.range("xyz", 3),
        ],
        properties=Labels(
            ["property"],
            torch.arange(100, dtype=torch.int32).reshape(-1, 1),
        ),
    )
    predictions = {"dos": TensorMap(keys=Labels.single(), blocks=[block])}
    assert not mts.is_contiguous(predictions["dos"])

    # write to DiskDatasetWriter
    new_dataset = DiskDatasetWriter("test_output.zip")
    new_dataset.write(systems, predictions)
    new_dataset.finish()


def test_ase_writer_streams_to_disk(monkeypatch, tmp_path):
    """ASEWriter should write to disk in write(), not accumulate in memory."""
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()
    filename = "test_streaming.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)

    # first batch
    writer.write(systems, predictions)
    # writer should NOT hold accumulated data
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    # second batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    writer.finish()

    # verify all 4 structures present and correct
    frames = read(filename, index=":")
    assert len(frames) == 4
    for i, atoms in enumerate(frames):
        assert atoms.info["energy"] == float(
            predictions["energy"].block().values[i % 2, 0]
        )


def test_metatensor_writer_streams_to_disk(monkeypatch, tmp_path):
    """MetatensorWriter should save to temp files in write(), not accumulate."""
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()
    filename = "test_streaming.mts"

    writer = MetatensorWriter(filename, capabilities=capabilities)

    # first batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    # second batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    writer.finish()

    # verify output has all 4 systems with correct system label offsets
    tensormap = mts.load("test_streaming_energy.mts")
    assert tensormap.block().values.shape == (4, 1)
    system_labels = tensormap.block().samples.column("system").tolist()
    assert system_labels == [0, 1, 2, 3]
    assert tensormap.block().gradient("positions").values.shape == (8, 3, 1)


def test_disk_dataset_writer_multi_batch_gradients(monkeypatch, tmp_path):
    """DiskDatasetWriter with gradients must not corrupt gradient sample indices.

    Regression test for the bug where ``_split_tensormaps`` applied the
    ``istart_system`` offset to the gradient "sample" column (a positional
    reference into the parent block) instead of leaving it at 0.  On batches
    after the first (``istart_system > 0``), this produced an out-of-range
    sample index and metatensor raised::

        RuntimeError: invalid value for the 'sample' dimension in gradient
        samples: we got N, but the values contain 1 samples
    """
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()

    writer = DiskDatasetWriter("test_multi_batch.zip", capabilities=capabilities)
    # First batch (istart_system=0) -- always worked
    writer.write(systems, predictions)
    # Second batch (istart_system=2) -- previously crashed on gradients
    writer.write(systems, predictions)
    writer.finish()


def test_disk_dataset_writer_multi_batch_strain_gradients(monkeypatch, tmp_path):
    """Same regression test but with strain gradients (samples=["sample"] only).

    Strain gradients have a single "sample" column, so the eye-matrix offset
    hit them too. Exercises both positions (["sample", "atom"]) and strain
    (["sample"]) gradient layouts in a single run.
    """
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions(
        cell=torch.eye(3)
    )

    writer = DiskDatasetWriter("test_strain.zip", capabilities=capabilities)
    writer.write(systems, predictions)
    writer.write(systems, predictions)
    writer.finish()


def test_memmap_writer_streams_to_disk(monkeypatch, tmp_path):
    """MemmapWriter should append to .bin files in write(), not accumulate."""
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()
    filename = "test_streaming/"

    writer = MemmapWriter(filename, capabilities=capabilities)

    # first batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    # second batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    writer.finish()

    # verify all 4 structures (2 batches x 2 systems) present with correct offsets
    directory = Path(filename)
    ns = int(np.load(directory / "ns.npy"))
    na = np.load(directory / "na.npy")
    assert ns == 4
    assert na.tolist() == [0, 2, 4, 6, 8]

    energy = np.fromfile(directory / "energy.bin", dtype=np.float32).reshape(4, 1)
    expected_energy = torch.cat(2 * [predictions["energy"].block().values]).numpy()
    np.testing.assert_allclose(energy, expected_energy, atol=1e-6)


def test_memmap_writer_saves_pbc(monkeypatch, tmp_path):
    """The periodicity of each system is saved to pbc.npy"""

    monkeypatch.chdir(tmp_path)

    _, capabilities, predictions = systems_capabilities_predictions()

    slab_cell = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    systems = 2 * [
        System(
            types=torch.tensor([1, 1]),
            positions=torch.tensor([[0, 0, 0], [0, 0, 0.74]]),
            cell=slab_cell,
            pbc=torch.tensor([True, True, False]),
        )
    ]

    filename = "test_pbc/"
    writer = MemmapWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    pbc = np.load(Path(filename) / "pbc.npy")
    assert pbc.dtype == bool
    assert pbc.tolist() == [[True, True, False], [True, True, False]]

    target_options = {
        "energy": {
            "key": "energy",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
            "quantity": "energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }

    dataset = MemmapDataset(Path(filename), target_options)
    assert dataset[0].system.pbc.tolist() == [True, True, False]


def test_memmap_writer_append_not_supported(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Appending is not supported"):
        MemmapWriter("test/", append=True)


def test_memmap_writer_reuses_empty_existing_directory(monkeypatch, tmp_path):
    """A pre-existing but empty directory is reused as-is."""
    monkeypatch.chdir(tmp_path)

    directory = Path("test_preexisting")
    directory.mkdir()

    systems, capabilities, predictions = systems_capabilities_predictions()
    writer = MemmapWriter("test_preexisting/", capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    assert directory.is_dir()
    assert (directory / "ns.npy").is_file()


def test_memmap_writer_errors_on_non_empty_directory(monkeypatch, tmp_path):
    """Construction must raise immediately (before any writing happens) if the
    target directory already exists and contains anything at all, rather than
    silently overwriting or discovering a conflict only partway through a
    potentially very long evaluation run."""
    monkeypatch.chdir(tmp_path)

    directory = Path("test_conflict")
    directory.mkdir()
    sentinel = directory / "unrelated_file.txt"
    sentinel.write_text("do not delete me")

    _, capabilities, _ = systems_capabilities_predictions()
    with pytest.raises(FileExistsError, match="not empty"):
        MemmapWriter("test_conflict/", capabilities=capabilities)

    # the pre-existing content must not have been touched
    assert sentinel.read_text() == "do not delete me"


def test_memmap_writer_stress_sign_left_handed_cell(monkeypatch, tmp_path):
    """The stress round-trip through MemmapDataset must reconstruct the original
    strain gradient, including its sign, even for a left-handed (negative
    determinant) cell.

    ``MemmapWriter`` stores ``stress = strain_derivative / volume`` while
    ``MemmapDataset`` reconstructs the gradient as ``stress * |det(cell)|``. Both
    sides must use the *same* (absolute) volume convention, or the sign flips for
    any cell with a negative determinant.
    """
    monkeypatch.chdir(tmp_path)

    # left-handed cell: det < 0
    cell = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    assert torch.det(cell).item() < 0

    systems, capabilities, predictions = systems_capabilities_predictions(cell=cell)

    filename = "test_left_handed/"
    writer = MemmapWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    target_options = {
        "energy": {
            "key": "energy",
            "quantity": "energy",
            "sample_kind": "system",
            "type": "scalar",
            "num_subtargets": 1,
            "forces": False,
            "stress": {"key": "energy_stress"},
            "virial": False,
        }
    }
    dataset = MemmapDataset(Path(filename), target_options)

    expected_strain = predictions["energy"].block().gradient("strain").values
    for i in range(len(systems)):
        reconstructed = dataset[i].energy.block().gradient("strain").values
        torch.testing.assert_close(
            reconstructed[0],
            expected_strain[i].to(torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )


def test_disk_dataset_writer_append_continues_indices(monkeypatch, tmp_path):
    """append=True must continue indexing after the existing entries.

    Regression test: the writer used to always start at index 0 regardless of
    ``append``, so a second session silently overwrote ("0/", "1/", ...) the
    first session's entries instead of adding new ones.
    """
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_append.zip")
    writer.write([_make_system(2)], {})
    writer.write([_make_system(5)], {})
    writer.write([_make_system(3)], {})
    writer.finish()

    writer = DiskDatasetWriter("test_append.zip", append=True)
    writer.write([_make_system(8)], {})
    writer.write([_make_system(4)], {})
    writer.finish()

    with zipfile.ZipFile("test_append.zip", "r") as zf:
        names = zf.namelist()
    for i in range(5):
        assert f"{i}/system.mta" in names, f"missing entry {i}, index did not continue"

    dataset = DiskDataset("test_append.zip", fields=[])
    assert len(dataset) == 5
    # original entries must be untouched, not overwritten by the append session
    assert len(dataset[0].system) == 2
    assert len(dataset[1].system) == 5
    assert len(dataset[2].system) == 3
    assert len(dataset[3].system) == 8
    assert len(dataset[4].system) == 4


def test_disk_dataset_writer_append_extends_atom_counts(monkeypatch, tmp_path):
    """The atom-count file must stay complete and correct across appends."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_append_atom_counts.zip")
    writer.write([_make_system(2)], {})
    writer.write([_make_system(5)], {})
    writer.finish()

    writer = DiskDatasetWriter("test_append_atom_counts.zip", append=True)
    writer.write([_make_system(8)], {})
    writer.finish()

    dataset = DiskDataset("test_append_atom_counts.zip", fields=[])
    # The repo's pytest config turns warnings into errors; if the file were
    # incomplete this would raise via the "no atom-counts file, backfilling"
    # warning instead of just returning the (wrong) answer.
    counts = dataset.get_all_atom_counts()
    assert counts.tolist() == [2, 5, 8]


def test_disk_dataset_writer_append_chain_extends_atom_counts_repeatedly(
    monkeypatch, tmp_path
):
    """Three chained append sessions must each extend the atom counts correctly.

    Also checks that the (expected) duplicate ``metadata/atom_counts.npy`` entries
    this produces resolve to the last (complete) one on read.
    """
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_append_chain.zip")
    writer.write([_make_system(1)], {})
    writer.write([_make_system(2)], {})
    writer.finish()

    writer = DiskDatasetWriter("test_append_chain.zip", append=True)
    writer.write([_make_system(3)], {})
    writer.finish()

    writer = DiskDatasetWriter("test_append_chain.zip", append=True)
    writer.write([_make_system(4)], {})
    writer.write([_make_system(5)], {})
    writer.finish()

    with zipfile.ZipFile("test_append_chain.zip", "r") as zf:
        n_atom_counts_entries = zf.namelist().count("metadata/atom_counts.npy")
    assert n_atom_counts_entries == 3

    dataset = DiskDataset("test_append_chain.zip", fields=[])
    assert len(dataset) == 5
    assert dataset.get_all_atom_counts().tolist() == [1, 2, 3, 4, 5]


def test_disk_dataset_writer_append_without_prior_atom_counts(monkeypatch, tmp_path):
    """Appending onto a dataset with no atom-count file must reconstruct it.

    A complete `metadata/atom_counts.npy` is a writer-guaranteed invariant (the reader
    never mutates the dataset, it only errors when the file is missing), so
    the append session must compute the counts of the entries it didn't write
    by reading their systems once.
    """
    monkeypatch.chdir(tmp_path)

    with zipfile.ZipFile("test_append_no_atom_counts.zip", "w") as zf:
        for i, n in enumerate([6, 7]):
            with zf.open(f"{i}/system.mta", "w") as f:
                mta.save(f, _make_system(n))

    writer = DiskDatasetWriter("test_append_no_atom_counts.zip", append=True)
    writer.write([_make_system(9)], {})
    writer.finish()

    with zipfile.ZipFile("test_append_no_atom_counts.zip", "r") as zf:
        names = zf.namelist()
    assert "2/system.mta" in names, "index did not continue past the pre-existing 2"
    dataset = DiskDataset("test_append_no_atom_counts.zip", fields=[])
    assert len(dataset) == 3
    assert dataset.get_all_atom_counts().tolist() == [6, 7, 9]


def test_disk_dataset_rejects_non_dense_entries(monkeypatch, tmp_path):
    """Zips with duplicated or missing entry numbers must be rejected loudly:
    the reader is positional, so reading them would silently return the
    last-written copy (duplicates) or crash mid-epoch (gaps)."""
    monkeypatch.chdir(tmp_path)

    # writing the duplicated entry itself warns; that's the very situation
    # this guard exists for
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile("test_duplicates.zip", "w") as zf:
            for i, n in zip([0, 1, 0], [2, 3, 5], strict=True):
                with zf.open(f"{i}/system.mta", "w") as f:
                    mta.save(f, _make_system(n))
    with pytest.raises(ValueError, match="dense range"):
        DiskDataset("test_duplicates.zip", fields=[])

    with zipfile.ZipFile("test_gaps.zip", "w") as zf:
        for i, n in zip([0, 2], [2, 3], strict=True):
            with zf.open(f"{i}/system.mta", "w") as f:
                mta.save(f, _make_system(n))
    with pytest.raises(ValueError, match="dense range"):
        DiskDataset("test_gaps.zip", fields=[])


def test_disk_dataset_reads_bypass_central_directory(monkeypatch, tmp_path):
    """After construction, reading samples must never re-open zipfile.ZipFile.

    DiskDataset indexes member offsets once at construction; __getitem__ reads
    by direct seek. Re-parsing the central directory per process is what made
    dataloader workers OOM on large datasets (~10 GB RSS per open for a
    13M-member zip), so a read path that silently falls back to zipfile would
    reintroduce that.
    """
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_no_cd.zip")
    writer.write([_make_system(2)], {})
    writer.write([_make_system(4)], {})
    writer.finish()

    dataset = DiskDataset("test_no_cd.zip", fields=[])

    def _no_zipfile(*args, **kwargs):
        raise AssertionError("sample reads must not construct zipfile.ZipFile")

    monkeypatch.setattr(zipfile, "ZipFile", _no_zipfile)
    assert len(dataset[0].system) == 2
    assert len(dataset[1].system) == 4


def test_disk_dataset_rejects_compressed_members(monkeypatch, tmp_path):
    """DiskDataset zips are STORED-only: users re-zipping a dataset by hand
    must not compress it, and a compressed member is rejected at construction
    with a re-zipping hint (not read wrongly or slowly later)."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_stored.zip")
    writer.write([_make_system(3)], {})
    writer.finish()

    with (
        zipfile.ZipFile("test_stored.zip", "r") as src,
        zipfile.ZipFile("test_deflated.zip", "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info.filename))

    with pytest.raises(ValueError, match="must be uncompressed"):
        DiskDataset("test_deflated.zip", fields=[])


def test_disk_dataset_missing_atom_counts_is_a_clean_error(monkeypatch, tmp_path):
    """The reader never mutates the dataset zip (no locks, no write-back —
    in-place mutation under distributed training and parallel filesystems is
    a corruption hazard): a dataset without `metadata/atom_counts.npy` must produce a
    clear error telling the user how to update it, from both accessors."""
    monkeypatch.chdir(tmp_path)

    with zipfile.ZipFile("test_no_counts.zip", "w") as zf:
        for i, n in zip([0, 1], [2, 3], strict=True):
            with zf.open(f"{i}/system.mta", "w") as f:
                mta.save(f, _make_system(n))

    dataset = DiskDataset("test_no_counts.zip", fields=[])
    with pytest.raises(ValueError, match="has no 'metadata/atom_counts.npy'"):
        dataset.get_all_atom_counts()
    with pytest.raises(ValueError, match="has no 'metadata/atom_counts.npy'"):
        dataset.get_num_atoms(0)
    # normal reading is unaffected
    assert len(dataset[1].system) == 3


def test_disk_dataset_warns_once_about_ignored_members(monkeypatch, tmp_path):
    """Members that are not part of the format (`<N>/system.mta`,
    `<N>/<field>.mts`, `metadata/atom_counts.npy`) are tolerated, but a
    single warning at construction lists them: they are harmless to read,
    yet a clean DiskDatasetWriter output would contain none of them."""
    monkeypatch.chdir(tmp_path)

    with zipfile.ZipFile("test_junk.zip", "w") as zf:
        with zf.open("0/system.mta", "w") as f:
            mta.save(f, _make_system(2))
        zf.writestr("readme.txt", "hello")
        zf.writestr("0/notes.txt", "hello")
        zf.writestr("0/sub/system.mta", "x")  # nested: not an entry member

    with pytest.warns(UserWarning, match=r"Skipping 3 file.*readme\.txt"):
        dataset = DiskDataset("test_junk.zip", fields=[])
    assert len(dataset) == 1
    assert len(dataset[0].system) == 2

    # the append writer tolerates them too (silently: the reader is where
    # the diagnosis matters) and leaves them in place
    writer = DiskDatasetWriter("test_junk.zip", append=True)
    writer.write([_make_system(3)], {})
    writer.finish()
    with pytest.warns(UserWarning, match="Skipping 3 file"):
        dataset = DiskDataset("test_junk.zip", fields=[])
    assert dataset.get_all_atom_counts().tolist() == [2, 3]

    # zero-padded entry numbers are not entry members: with no valid entry
    # left, construction fails (and the ignored files give the clue)
    with zipfile.ZipFile("test_padded.zip", "w") as zf:
        with zf.open("00/system.mta", "w") as f:
            mta.save(f, _make_system(2))
    with pytest.raises(ValueError, match=r"Could not find any.*00/system\.mta"):
        DiskDataset("test_padded.zip", fields=[])


def test_disk_dataset_no_entries_error_carries_hints(monkeypatch, tmp_path):
    """The classic hand-zipping mistake — zipping the dataset folder from
    outside — leaves no valid entries at all; the error lists the ignored
    members and names the exact command to fix it."""
    monkeypatch.chdir(tmp_path)

    with zipfile.ZipFile("test_prefixed.zip", "w") as zf:
        with zf.open("data/0/system.mta", "w") as f:
            mta.save(f, _make_system(2))
    with pytest.raises(ValueError, match="cd data"):
        DiskDataset("test_prefixed.zip", fields=[])


def test_disk_dataset_tolerates_macos_junk(monkeypatch, tmp_path):
    """Zipping on a Mac sprinkles `__MACOSX/` resource forks and `.DS_Store`
    files into the archive; they are inert, so plain unzip/re-zip on macOS
    must remain a valid way to produce a DiskDataset — read (with the
    ignored-members warning) and appended to alike."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_macos.zip")
    writer.write([_make_system(2)], {})
    writer.finish()
    with zipfile.ZipFile("test_macos.zip", "a") as zf:
        zf.writestr("__MACOSX/0/._system.mta", "junk")
        zf.writestr(".DS_Store", "junk")
        zf.writestr("0/.DS_Store", "junk")

    with pytest.warns(UserWarning, match="Skipping 3 file"):
        dataset = DiskDataset("test_macos.zip", fields=[])
    assert len(dataset) == 1
    assert len(dataset[0].system) == 2
    assert dataset.get_all_atom_counts().tolist() == [2]

    writer = DiskDatasetWriter("test_macos.zip", append=True)
    writer.write([_make_system(3)], {})
    writer.finish()
    with pytest.warns(UserWarning, match="Skipping 3 file"):
        dataset = DiskDataset("test_macos.zip", fields=[])
    assert dataset.get_all_atom_counts().tolist() == [2, 3]


def test_disk_dataset_unzip_rezip_roundtrip(monkeypatch, tmp_path):
    """Unzipping a writer-produced dataset and re-zipping it with standard
    tools (STORED, from inside the folder) must read back identically —
    including the directory entries such tools materialize."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("original.zip")
    writer.write([_make_system(2)], {})
    writer.write([_make_system(5)], {})
    writer.finish()

    # simulate `cd folder && zip -rX -Z store ../rezipped.zip .`
    with (
        zipfile.ZipFile("original.zip", "r") as src,
        zipfile.ZipFile("rezipped.zip", "w", zipfile.ZIP_STORED) as dst,
    ):
        dst.writestr("0/", b"")  # directory entries
        dst.writestr("1/", b"")
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info.filename))

    original = DiskDataset("original.zip", fields=[])
    rezipped = DiskDataset("rezipped.zip", fields=[])
    assert len(rezipped) == len(original) == 2
    assert rezipped.get_all_atom_counts().tolist() == [2, 5]
    for i in range(2):
        torch.testing.assert_close(
            rezipped[i].system.positions, original[i].system.positions
        )


def test_disk_dataset_rejects_heterogeneous_entries(monkeypatch, tmp_path):
    """Every entry must carry the same fields: a member missing from some
    entries fails at construction with the entry and field named, instead of
    crashing whenever that sample happens to be read."""
    monkeypatch.chdir(tmp_path)

    energy = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0]], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]])),
                components=[],
                properties=Labels(["energy"], torch.tensor([[0]])),
            )
        ],
    )
    with zipfile.ZipFile("test_hetero.zip", "w") as zf:
        for i in range(2):
            with zf.open(f"{i}/system.mta", "w") as f:
                mta.save(f, _make_system(2))
        with zf.open("1/energy.mts", "w") as f:
            np.save(f, energy.save_buffer().numpy())

    with pytest.raises(ValueError, match="entry 0 has no 'energy'"):
        DiskDataset("test_hetero.zip", fields=[])


def test_disk_dataset_detects_corrupted_members(monkeypatch, tmp_path):
    """Reads verify the member's CRC-32 from the central directory: bit rot or
    a truncated copy must raise instead of feeding corrupted bytes (systems,
    coefficients) into training."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_crc.zip")
    writer.write([_make_system(4)], {})
    writer.finish()

    dataset = DiskDataset("test_crc.zip", fields=[])
    with zipfile.ZipFile("test_crc.zip") as zf:
        offset = zf.getinfo("0/system.mta").header_offset
    with open("test_crc.zip", "r+b") as f:
        f.seek(offset + 200)  # inside the member's data
        byte = f.read(1)
        f.seek(offset + 200)
        f.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(zipfile.BadZipFile, match="Bad CRC-32|corrupted"):
        dataset[0]


def test_disk_dataset_rejects_out_of_range_indices(monkeypatch, tmp_path):
    """Negative indices must not silently wrap to another sample (numpy
    semantics) — the returned system_index would then lie about which
    structure the data belongs to."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_range.zip")
    writer.write([_make_system(2)], {})
    writer.write([_make_system(3)], {})
    writer.finish()

    dataset = DiskDataset("test_range.zip", fields=[])
    with pytest.raises(IndexError, match="out of range"):
        dataset[-1]
    with pytest.raises(IndexError, match="out of range"):
        dataset[2]


def test_disk_dataset_writer_append_rejects_corrupt_zip(monkeypatch, tmp_path):
    """Appending to a corrupt zip must fail loudly, with the file untouched:
    zipfile mode 'a' would append blindly after the corrupt bytes and write a
    central directory listing only the new members, silently discarding every
    prior entry."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_corrupt.zip")
    writer.write([_make_system(2)], {})
    writer.write([_make_system(3)], {})
    writer.finish()

    # Truncation can leave the trailing bytes of an *embedded* archive
    # (system.mta members are zips themselves), whose end-of-directory record
    # zipfile then finds: depending on where the cut lands, the probe raises
    # BadZipFile directly or our format validation rejects the bogus member
    # names. Either way it must raise before anything is written.
    size = Path("test_corrupt.zip").stat().st_size
    before = Path("test_corrupt.zip").read_bytes()[: size // 2]
    with open("test_corrupt.zip", "r+b") as f:
        f.truncate(size // 2)

    with pytest.raises((zipfile.BadZipFile, ValueError)):
        DiskDatasetWriter("test_corrupt.zip", append=True)
    assert Path("test_corrupt.zip").read_bytes() == before, "file was modified"


def test_disk_dataset_writer_append_rejects_non_dense_zip(monkeypatch, tmp_path):
    """Appending to a gapped zip must fail while the file is still intact:
    the reader is guaranteed to reject the result, so appending hours of
    writes to it would produce an unreadable dataset."""
    monkeypatch.chdir(tmp_path)

    with zipfile.ZipFile("test_gapped.zip", "w") as zf:
        for i in [0, 2]:
            with zf.open(f"{i}/system.mta", "w") as f:
                mta.save(f, _make_system(2))

    with pytest.raises(ValueError, match="dense range"):
        DiskDatasetWriter("test_gapped.zip", append=True)


def test_disk_dataset_writer_append_fresh_file_writes_counts(monkeypatch, tmp_path):
    """append=True on a nonexistent (or empty) path is a fresh write: the
    counts file must be written like in write mode, not silently omitted."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("fresh.zip", append=True)
    writer.write([_make_system(4)], {})
    writer.finish()

    dataset = DiskDataset("fresh.zip", fields=[])
    assert dataset.get_all_atom_counts().tolist() == [4]

    Path("empty.zip").touch()
    writer = DiskDatasetWriter("empty.zip", append=True)
    writer.write([_make_system(6)], {})
    writer.finish()
    assert DiskDataset("empty.zip", fields=[]).get_all_atom_counts().tolist() == [6]


def test_disk_dataset_writer_finish_is_idempotent(monkeypatch, tmp_path):
    """finish() in a finally block is a natural idiom; a second call must be
    a no-op, not an 'archive already closed' crash."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_finish.zip")
    writer.write([_make_system(2)], {})
    writer.finish()
    writer.finish()

    assert DiskDataset("test_finish.zip", fields=[]).get_all_atom_counts().tolist() == [
        2
    ]


def test_disk_dataset_rejects_orphan_and_duplicate_members(monkeypatch, tmp_path):
    """Field members for entries without a system, and duplicated member
    names, are out of spec: both must fail at construction instead of being
    silently dropped or silently last-wins."""
    monkeypatch.chdir(tmp_path)

    energy = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0]], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]])),
                components=[],
                properties=Labels(["energy"], torch.tensor([[0]])),
            )
        ],
    )
    energy_buffer = energy.save_buffer().numpy()

    with zipfile.ZipFile("test_orphan.zip", "w") as zf:
        for i in range(2):
            with zf.open(f"{i}/system.mta", "w") as f:
                mta.save(f, _make_system(2))
            with zf.open(f"{i}/energy.mts", "w") as f:
                np.save(f, energy_buffer)
        with zf.open("7/energy.mts", "w") as f:  # no 7/system.mta
            np.save(f, energy_buffer)
    with pytest.raises(ValueError, match="duplicated files or field files"):
        DiskDataset("test_orphan.zip", fields=[])

    with zipfile.ZipFile("test_dup_field.zip", "w") as zf:
        with zf.open("0/system.mta", "w") as f:
            mta.save(f, _make_system(2))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name")
            for _ in range(2):
                with zf.open("0/energy.mts", "w") as f:
                    np.save(f, energy_buffer)
    with pytest.raises(ValueError, match="duplicated files or field files"):
        DiskDataset("test_dup_field.zip", fields=[])


def test_disk_dataset_metadata_folder(monkeypatch, tmp_path):
    """The atom counts can ONLY come from `metadata/atom_counts.npy`: other
    files (even a plausible-looking top-level `_atom_counts.npy`) are
    ignored-with-warning, never read as counts."""
    monkeypatch.chdir(tmp_path)

    writer = DiskDatasetWriter("test_meta.zip")
    writer.write([_make_system(2)], {})
    writer.finish()
    with zipfile.ZipFile("test_meta.zip", "r") as zf:
        assert "metadata/atom_counts.npy" in zf.namelist()
    dataset = DiskDataset("test_meta.zip", fields=[])
    assert dataset.get_all_atom_counts().tolist() == [2]

    with zipfile.ZipFile("test_toplevel.zip", "w") as zf:
        with zf.open("0/system.mta", "w") as f:
            mta.save(f, _make_system(3))
        with zf.open("_atom_counts.npy", "w") as f:
            np.save(f, np.array([3], dtype=np.int64))
    with pytest.warns(UserWarning, match=r"Skipping 1 file.*_atom_counts"):
        dataset = DiskDataset("test_toplevel.zip", fields=[])
    with pytest.raises(ValueError, match="has no 'metadata/atom_counts.npy'"):
        dataset.get_all_atom_counts()


def test_disk_dataset_writer_single_and_batched_writes_match(monkeypatch, tmp_path):
    """Writing systems one at a time or in one batched call must produce
    identical datasets: the stored "system" sample labels always hold the
    entry number. Without this, `mtt eval` with batch_size=1 (the default)
    produced zips whose targets all carried system=0, which crash at
    collation when training on them."""
    monkeypatch.chdir(tmp_path)

    def energy_map(values, system_ids):
        return TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor(values, dtype=torch.float64),
                    samples=Labels(["system"], torch.tensor(system_ids).reshape(-1, 1)),
                    components=[],
                    properties=Labels(["energy"], torch.tensor([[0]])),
                )
            ],
        )

    systems = [_make_system(n) for n in (2, 5, 3)]

    writer = DiskDatasetWriter("batched.zip")
    writer.write(systems, {"energy": energy_map([[1.0], [2.0], [3.0]], [0, 1, 2])})
    writer.finish()

    # one at a time, every map carrying system=0 (an eval batch of one), and
    # the last carrying an unrelated source id (a dumped-dataset scenario)
    writer = DiskDatasetWriter("singles.zip")
    writer.write(systems[:1], {"energy": energy_map([[1.0]], [0])})
    writer.write(systems[1:2], {"energy": energy_map([[2.0]], [0])})
    writer.write(systems[2:], {"energy": energy_map([[3.0]], [77])})
    writer.finish()

    batched = DiskDataset("batched.zip", fields=["energy"])
    singles = DiskDataset("singles.zip", fields=["energy"])
    assert len(batched) == len(singles) == 3
    for i in range(3):
        block_b = batched[i].energy.block()
        block_s = singles[i].energy.block()
        assert block_b.samples == block_s.samples
        assert block_s.samples.values[0, 0] == i, "label must be the entry number"
        torch.testing.assert_close(block_b.values, block_s.values)

    # a "single system" write with a genuinely multi-system map is an error
    writer = DiskDatasetWriter("bad.zip")
    with pytest.raises(ValueError, match="multiple system ids"):
        writer.write(systems[:1], {"energy": energy_map([[1.0], [2.0]], [0, 1])})
    writer.finish()


def test_disk_dataset_writer_batched_write_rejects_foreign_ids(monkeypatch, tmp_path):
    """Batched writes split samples by batch-local system ids (0..B-1); maps
    labeled with anything else used to produce silently EMPTY target members.
    They must be rejected loudly instead."""
    monkeypatch.chdir(tmp_path)

    systems = [_make_system(2), _make_system(3)]
    energy = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[5], [2]])),
                components=[],
                properties=Labels(["energy"], torch.tensor([[0]])),
            )
        ],
    )
    writer = DiskDatasetWriter("test_foreign.zip")
    with pytest.raises(ValueError, match=r"label its 2 systems 0\.\.1"):
        writer.write(systems, {"energy": energy})
    writer.finish()
