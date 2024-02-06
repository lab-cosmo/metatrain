import re

import pytest
from omegaconf import OmegaConf

from metatensor.models.utils.omegaconf import check_units, expand_dataset_config


def test_file_format_resolver():
    conf = OmegaConf.create({"read_from": "foo.xyz", "file_format": "${file_format:}"})

    assert (conf["file_format"]) == ".xyz"


def test_expand_dataset_config():
    file_name = "foo.xyz"
    file_format = ".xyz"

    structure_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "structures": structure_section,
        "targets": {"energy": target_section, "my_target": target_section},
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["structures"]["read_from"] == file_name
    assert conf_expanded["structures"]["file_format"] == file_format
    assert conf_expanded["structures"]["length_unit"] == "angstrom"

    targets_conf = conf_expanded["targets"]
    assert len(targets_conf) == 2

    for target_key in ["energy", "my_target"]:
        assert targets_conf[target_key]["quantity"] == "energy"
        assert targets_conf[target_key]["read_from"] == file_name
        assert targets_conf[target_key]["file_format"] == file_format
        assert targets_conf[target_key]["file_format"] == file_format
        assert targets_conf[target_key]["unit"] is None

        assert targets_conf[target_key]["forces"]["read_from"] == file_name
        assert targets_conf[target_key]["forces"]["file_format"] == file_format
        assert targets_conf[target_key]["forces"]["key"] == "forces"

        assert targets_conf[target_key]["virial"]["read_from"] == "my_grad.dat"
        assert targets_conf[target_key]["virial"]["file_format"] == ".dat"
        assert targets_conf[target_key]["virial"]["key"] == "foo"

        assert targets_conf[target_key]["stress"] is False

    # If a virial is parsed as in the conf above the by default enabled section "stress"
    # should be disabled automatically
    assert targets_conf["energy"]["stress"] is False


def test_expand_dataset_config_not_energy():
    file_name = "foo.xyz"

    structure_section = {"read_from": file_name, "unit": "angstrom"}

    target_section = {
        "quantity": "my_dipole_moment",
    }

    conf = {
        "structures": structure_section,
        "targets": {"dipole_moment": target_section},
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["targets"]["dipole_moment"]["key"] == "dipole_moment"
    assert conf_expanded["targets"]["dipole_moment"]["quantity"] == "my_dipole_moment"
    assert conf_expanded["targets"]["dipole_moment"]["forces"] is False
    assert conf_expanded["targets"]["dipole_moment"]["stress"] is False
    assert conf_expanded["targets"]["dipole_moment"]["virial"] is False


def test_expand_dataset_config_min():
    file_name = "dataset.dat"
    file_format = ".dat"

    conf_expanded = expand_dataset_config(file_name)

    assert conf_expanded["structures"]["read_from"] == file_name
    assert conf_expanded["structures"]["file_format"] == file_format
    assert conf_expanded["structures"]["length_unit"] is None

    targets_conf = conf_expanded["targets"]
    assert targets_conf["energy"]["quantity"] == "energy"
    assert targets_conf["energy"]["read_from"] == file_name
    assert targets_conf["energy"]["file_format"] == file_format
    assert targets_conf["energy"]["file_format"] == file_format
    assert targets_conf["energy"]["key"] == "energy"
    assert targets_conf["energy"]["unit"] is None

    for gradient in ["forces", "stress"]:
        assert targets_conf["energy"][gradient]["read_from"] == file_name
        assert targets_conf["energy"][gradient]["file_format"] == file_format
        assert targets_conf["energy"][gradient]["key"] == gradient

    assert targets_conf["energy"]["virial"] is False


def test_expand_dataset_config_error():
    file_name = "foo.xyz"

    conf = {
        "structures": file_name,
        "targets": {
            "energy": {
                "virial": file_name,
                "stress": {"read_from": file_name, "key": "foo"},
            }
        },
    }

    with pytest.raises(
        ValueError, match="Cannot perform training with respect to virials and stress"
    ):
        expand_dataset_config(OmegaConf.create(conf))


def test_expand_dataset_gradient():
    conf = {
        "structures": "foo.xyz",
        "targets": {
            "my_energy": {
                "forces": "data.txt",
                "virial": True,
                "stress": False,
            }
        },
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["targets"]["my_energy"]["forces"]["read_from"] == "data.txt"
    assert conf_expanded["targets"]["my_energy"]["forces"]["file_format"] == ".txt"

    assert conf_expanded["targets"]["my_energy"]["stress"] is False
    conf_expanded["targets"]["my_energy"]["virial"]["read_from"]


def test_check_units():
    file_name = "foo.xyz"
    structure_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "unit": "eV",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    mytarget_section = {
        "quantity": "love",
        "forces": file_name,
        "unit": "heart",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "structures": structure_section,
        "targets": {"energy": target_section, "my_target": mytarget_section},
    }

    structure_section1 = {"read_from": file_name, "length_unit": "angstrom1"}

    target_section1 = {
        "quantity": "energy",
        "forces": file_name,
        "unit": "eV_",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    mytarget_section1 = {
        "quantity": "love",
        "forces": file_name,
        "unit": "heart_",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf1 = {
        "structures": structure_section1,
        "targets": {"energy": target_section, "my_target": mytarget_section},
    }
    conf0 = {
        "structures": structure_section,
        "targets": {"energy": target_section, "my_target0": mytarget_section},
    }
    conf2 = {
        "structures": structure_section,
        "targets": {"energy": target_section1, "my_target": mytarget_section},
    }
    conf3 = {
        "structures": structure_section,
        "targets": {"energy": target_section, "my_target": mytarget_section1},
    }

    train_options = expand_dataset_config(OmegaConf.create(conf))
    test_options = expand_dataset_config(OmegaConf.create(conf))

    test_options0 = expand_dataset_config(OmegaConf.create(conf0))

    test_options1 = expand_dataset_config(OmegaConf.create(conf1))
    test_options2 = expand_dataset_config(OmegaConf.create(conf2))

    test_options3 = expand_dataset_config(OmegaConf.create(conf3))

    check_units(actual_options=test_options, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "length units are inconsistent between dataset options."
            " angstrom1 != angstrom"
        ),
    ):
        check_units(actual_options=test_options1, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape("target 'my_target' is not present in the given dataset."),
    ):
        check_units(actual_options=test_options0, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "units of target 'energy' are inconsistent between dataset options."
            " eV_ != eV."
        ),
    ):
        check_units(actual_options=test_options2, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "units of target 'my_target' are inconsistent between dataset options."
            " heart_ != heart."
        ),
    ):
        check_units(actual_options=test_options3, desired_options=train_options)
