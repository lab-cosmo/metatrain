import glob
import logging
import re
import shutil
import subprocess
from pathlib import Path

import ase.io
import metatensor.torch  # noqa
import pytest
import torch
from jsonschema.exceptions import ValidationError
from omegaconf import OmegaConf

from metatrain import RANDOM_SEED
from metatrain.cli.train import train_model
from metatrain.utils.errors import ArchitectureError

from . import (
    DATASET_PATH_ETHANOL,
    DATASET_PATH_QM9,
    MODEL_PATH_64_BIT,
    OPTIONS_PATH,
    RESOURCES_PATH,
)


@pytest.fixture
def options():
    return OmegaConf.load(OPTIONS_PATH)


@pytest.mark.parametrize("output", [None, "mymodel.pt"])
def test_train(capfd, monkeypatch, tmp_path, output):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    shutil.copy(OPTIONS_PATH, "options.yaml")

    command = ["mtt", "train", "options.yaml"]

    if output is not None:
        command += ["-o", output]
    else:
        output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if restart_options.yaml file is written
    restart_glob = glob.glob("outputs/*/*/options_restart.yaml")
    assert len(restart_glob) == 1

    # Open restart options an check that default parameters are overwritten
    restart_options = OmegaConf.load(restart_glob[0])
    assert restart_options["architecture"]["training"]["num_epochs"] == 1

    # Test if logfile is written
    log_glob = glob.glob("outputs/*/*/train.log")
    assert len(log_glob) == 1

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Open the log file and check if the logging is correct
    with open(log_glob[0]) as f:
        file_log = f.read()

    stdout_log = capfd.readouterr().out

    assert file_log == stdout_log

    assert "This log is also available" in stdout_log
    assert "Running training for 'experimental.soap_bpnn' architecture"
    assert re.search(r"Random seed of this run is [1-9]\d*", stdout_log)
    assert "Training dataset:" in stdout_log
    assert "Validation dataset:" in stdout_log
    assert "Test dataset:" in stdout_log
    assert "size 50" in stdout_log
    assert "mean " in stdout_log
    assert "std " in stdout_log
    assert "[INFO]" in stdout_log
    assert "Epoch" in stdout_log
    assert "loss" in stdout_log
    assert "validation" in stdout_log
    assert "train" in stdout_log
    assert "energy" in stdout_log
    assert "with index" not in stdout_log  # index only printed for more than 1 dataset


@pytest.mark.parametrize(
    "overrides",
    [
        "architecture.training.num_epochs=2",
        "architecture.training.num_epochs=2 architecture.training.batch_size=3",
    ],
)
def test_command_line_override(monkeypatch, tmp_path, overrides):
    """Test that training options can be overwritten from the command line."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    shutil.copy(OPTIONS_PATH, "options.yaml")

    command = ["mtt", "train", "options.yaml", "-r", overrides]

    subprocess.check_call(command)

    restart_glob = glob.glob("outputs/*/*/options_restart.yaml")
    assert len(restart_glob) == 1

    restart_options = OmegaConf.load(restart_glob[0])
    assert restart_options["architecture"]["training"]["num_epochs"] == 2

    if len(overrides.split()) == 2:
        assert restart_options["architecture"]["training"]["batch_size"] == 3


def test_train_from_options_restart_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    # run training with original options
    options = OmegaConf.load(OPTIONS_PATH)
    train_model(options)

    # run training with options_restart.yaml
    options_restart = OmegaConf.load("options_restart.yaml")
    train_model(options_restart)


def test_train_unknonw_arch_options(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options_str = """
    architecture:
        name: experimental.soap_bpnn
        training:
            batch_size: 2
            num_epoch: 1

    training_set:
        systems:
            read_from: qm9_reduced_100.xyz
            length_unit: angstrom
        targets:
            energy:
            key: U0
            unit: eV

    test_set: 0.5
    validation_set: 0.1
    """
    options = OmegaConf.create(options_str)

    match = r"Additional properties are not allowed \('num_epoch' was unexpected\)"
    with pytest.raises(ValidationError, match=match):
        train_model(options)


@pytest.mark.parametrize("n_datasets", [1, 2])
@pytest.mark.parametrize("test_set_file", (True, False))
@pytest.mark.parametrize("validation_set_file", (True, False))
def test_train_explicit_validation_test(
    monkeypatch,
    tmp_path,
    caplog,
    n_datasets,
    test_set_file,
    validation_set_file,
    options,
):
    """Test that training via the training cli runs without an error raise
    also when the validation and test sets are provided explicitly."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    systems = ase.io.read(DATASET_PATH_QM9, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])

    options["training_set"] = OmegaConf.create(n_datasets * [options["training_set"]])

    if validation_set_file:
        ase.io.write("validation.xyz", systems[50:80])
        options["validation_set"] = options["training_set"][0].copy()
        options["validation_set"]["systems"]["read_from"] = "validation.xyz"
        options["validation_set"] = OmegaConf.create(
            n_datasets * [options["validation_set"]]
        )

    if test_set_file:
        ase.io.write("test.xyz", systems[80:])
        options["test_set"] = options["training_set"][0].copy()
        options["test_set"]["systems"]["read_from"] = "test.xyz"
        options["test_set"] = OmegaConf.create(n_datasets * [options["test_set"]])

    train_model(options)

    # Test log messages which are written to STDOUT
    log = caplog.text
    for set_type in ["training", "test", "validation"]:
        for i in range(n_datasets):
            if n_datasets == 1:
                extra_log_message = ""
            else:
                extra_log_message = f" with index {i}"

            assert f"Evaluating {set_type} dataset{extra_log_message}" in log

    assert Path("model.pt").is_file()


def test_train_multiple_datasets(monkeypatch, tmp_path, options):
    """Test that training via the training cli runs without an error raise
    also when learning on two different datasets."""
    monkeypatch.chdir(tmp_path)

    systems_qm9 = ase.io.read(DATASET_PATH_QM9, ":")
    systems_ethanol = ase.io.read(DATASET_PATH_ETHANOL, ":")

    ase.io.write("qm9_reduced_100.xyz", systems_qm9[:50])
    ase.io.write("ethanol_reduced_100.xyz", systems_ethanol[:50])

    options["training_set"] = OmegaConf.create(2 * [options["training_set"]])
    options["training_set"][1]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"][1]["targets"]["energy"]["key"] = "energy"
    options["training_set"][0]["targets"].pop("energy")
    options["training_set"][0]["targets"]["mtt::U0"] = OmegaConf.create({"key": "U0"})

    train_model(options)


def test_empty_training_set(monkeypatch, tmp_path, options):
    """Test that an error is raised if no training set is provided."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.6
    options["test_set"] = 0.4

    with pytest.raises(
        ValueError, match="Fraction of the train set is smaller or equal to 0!"
    ):
        train_model(options)


@pytest.mark.parametrize("split", [-0.1, 1.1])
def test_wrong_test_split_size(split, monkeypatch, tmp_path, options):
    """Test that an error is raised if the test split has the wrong size"""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.1
    options["test_set"] = split

    if split > 1:
        match = rf"{split} is greater than or equal to the maximum of 1"
    if split < 0:
        match = rf"{split} is less than the minimum of 0"

    with pytest.raises(ValidationError, match=match):
        train_model(options)


@pytest.mark.parametrize("split", [0.0, 1.1])
def test_wrong_validation_split_size(split, monkeypatch, tmp_path, options):
    """Test that an error is raised if the validation split has the wrong size"""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = split
    options["test_set"] = 0.1

    if split > 1:
        match = rf"{split} is greater than or equal to the maximum of 1"
    if split <= 0:
        match = rf"{split} is less than or equal to the minimum of 0"

    with pytest.raises(ValidationError, match=match):
        train_model(options)


def test_empty_test_set(caplog, monkeypatch, tmp_path, options):
    """Test that NO error is raised if no test set is provided."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.4
    options["test_set"] = 0.0

    train_model(options)

    # check if the logging is correct
    assert "This dataset is empty. No evaluation" in caplog.text


@pytest.mark.parametrize(
    "test_set_file, validation_set_file", [(True, False), (False, True)]
)
def test_unit_check_is_performed(
    monkeypatch,
    tmp_path,
    test_set_file,
    validation_set_file,
    options,
):
    """Test that error is raised if units are inconsistent between the datasets."""
    monkeypatch.chdir(tmp_path)

    systems = ase.io.read(DATASET_PATH_QM9, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])

    if validation_set_file:
        ase.io.write("test.xyz", systems[50:80])
        options["validation_set"] = options["training_set"].copy()
        options["validation_set"]["systems"]["read_from"] = "test.xyz"
        options["validation_set"]["systems"]["length_unit"] = "foo"

    if test_set_file:
        ase.io.write("validation.xyz", systems[80:])
        options["test_set"] = options["training_set"].copy()
        options["test_set"]["systems"]["read_from"] = "validation.xyz"
        options["test_set"]["systems"]["length_unit"] = "foo"

    with pytest.raises(ValueError, match="`length_unit`s are inconsistent"):
        train_model(options)


@pytest.mark.parametrize(
    "test_set_file, validation_set_file", [(True, False), (False, True)]
)
def test_inconsistent_number_of_datasets(
    monkeypatch, tmp_path, test_set_file, validation_set_file, options
):
    """Test that error is raised in inconsistent number datasets are provided.

    i.e one train dataset but two validation datasets. Same for the test dataset."""
    monkeypatch.chdir(tmp_path)

    systems = ase.io.read(DATASET_PATH_QM9, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])

    if validation_set_file:
        ase.io.write("validation.xyz", systems[50:80])
        options["validation_set"] = options["training_set"].copy()
        options["validation_set"]["systems"]["read_from"] = "validation.xyz"
        options["validation_set"] = OmegaConf.create(2 * [options["validation_set"]])

    if test_set_file:
        ase.io.write("test.xyz", systems[80:])
        options["test_set"] = options["training_set"].copy()
        options["test_set"]["systems"]["read_from"] = "test.xyz"
        options["test_set"] = OmegaConf.create(2 * [options["test_set"]])

    with pytest.raises(ValueError, match="different size than the training datatset"):
        train_model(options)


@pytest.mark.parametrize(
    "taining_set_file, test_set_file, validation_set_file",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_inconsistencies_within_list_datasets(
    monkeypatch,
    tmp_path,
    taining_set_file,
    test_set_file,
    validation_set_file,
    options,
):
    """Test error raise if inconsistency within one datasets config present."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    ref_dataset_conf = OmegaConf.create(2 * [options["training_set"]])
    broken_dataset_conf = ref_dataset_conf.copy()
    broken_dataset_conf[0]["systems"]["length_unit"] = "foo"
    broken_dataset_conf[1]["systems"]["length_unit"] = "bar"

    options["training_set"] = ref_dataset_conf
    options["validation_set"] = ref_dataset_conf
    options["test_set"] = ref_dataset_conf

    if taining_set_file:
        options["training_set"] = broken_dataset_conf
    if test_set_file:
        options["test_set"] = broken_dataset_conf
    if validation_set_file:
        options["validation_set"] = broken_dataset_conf

    with pytest.raises(ValueError, match="`length_unit`s are inconsistent"):
        train_model(options)


def test_continue(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    train_model(options, continue_from=MODEL_PATH_64_BIT)


def test_continue_different_dataset(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise
    with a different dataset than the original."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "ethanol_reduced_100.xyz", "ethanol_reduced_100.xyz")

    options["training_set"]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"]["targets"]["energy"]["key"] = "energy"

    train_model(options, continue_from=MODEL_PATH_64_BIT)


@pytest.mark.parametrize("seed", [None, 1234])
def test_model_consistency_with_seed(options, monkeypatch, tmp_path, seed):
    """Checks final model consistency with a fixed seed."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    if seed is not None:
        options["seed"] = seed

    train_model(options, output="model1.pt")

    if seed is None:
        options["seed"] = RANDOM_SEED + 1

    train_model(options, output="model2.pt")

    m1 = torch.load("model1.ckpt")
    m2 = torch.load("model2.ckpt")

    for i in m1["model_state_dict"]:
        tensor1 = m1["model_state_dict"][i]
        tensor2 = m2["model_state_dict"][i]

        if seed is None:
            assert not torch.allclose(tensor1, tensor2)
        else:
            torch.testing.assert_close(tensor1, tensor2)


def test_base_validation(options, monkeypatch, tmp_path):
    """Test that the base options are validated."""
    monkeypatch.chdir(tmp_path)

    options["base_precision"] = 67

    match = r"67 is not one of \[16, 32, 64\]"
    with pytest.raises(ValidationError, match=match):
        train_model(options)


# TODO add parametrize for 16-bit once we have a model that supports this.
@pytest.mark.parametrize("base_precision", [64])
def test_different_base_precision(options, monkeypatch, tmp_path, base_precision):
    """Test different `base_precision`s."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    options["base_precision"] = base_precision
    train_model(options)


def test_architecture_error(options, monkeypatch, tmp_path):
    """Test an error raise if there is problem wth the architecture."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["architecture"]["model"] = OmegaConf.create({"soap": {"cutoff": -1.0}})

    with pytest.raises(ArchitectureError, match="originates from an architecture"):
        train_model(options)
