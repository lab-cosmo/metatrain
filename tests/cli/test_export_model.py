"""Test command line interface for the export functions.

Actual unit tests for the function are performed in `tests/utils/test_export`.
"""

import glob
import subprocess
from pathlib import Path

import pytest
import torch
from metatensor.torch.atomistic import load_atomistic_model

from metatrain.cli.export import export_model
from metatrain.experimental.soap_bpnn import __model__
from metatrain.utils.architectures import find_all_architectures
from metatrain.utils.data import DatasetInfo, TargetInfo

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("path", [Path("exported.pt"), "exported.pt"])
def test_export(monkeypatch, tmp_path, path):
    """Tests the export_model function."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1},
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="eV", per_atom=False, gradients=[]
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    export_model(model, path)

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    assert Path(path).is_file()


@pytest.mark.parametrize("output", [None, "exported.pt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_export_cli(monkeypatch, tmp_path, output, dtype):
    """Test that the export cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)

    dtype_string = "64" if dtype == torch.float64 else "32"
    command = [
        "mtt",
        "export",
        "experimental.soap_bpnn",
        str(RESOURCES_PATH / f"model-{dtype_string}-bit.ckpt"),
    ]

    if output is not None:
        command += ["-o", output]
    else:
        output = "exported-model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test that the model can be loaded
    model = load_atomistic_model(output, extensions_directory="extensions/")

    # Check that the model has the correct dtype and is on cpu
    assert next(model.parameters()).dtype == dtype
    assert next(model.parameters()).device.type == "cpu"


def test_export_cli_architecture_names_choices():
    stderr = str(subprocess.run(["mtt", "export", "foo"], capture_output=True).stderr)

    assert "invalid choice: 'foo'" in stderr
    for architecture_name in find_all_architectures():
        assert architecture_name in stderr


def test_reexport(monkeypatch, tmp_path):
    """Test that an already exported model can be loaded and again exported."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6, 7, 8},
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="eV", per_atom=False, gradients=[]
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    export_model(model, "exported.pt")

    model_loaded = load_atomistic_model("exported.pt")
    export_model(model_loaded, "exported_new.pt")

    assert Path("exported_new.pt").is_file()
