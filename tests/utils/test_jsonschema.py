import json

import pytest
from jsonschema.exceptions import ValidationError

from metatrain.utils.architectures import get_architecture_path
from metatrain.utils.jsonschema import validate


def schema():
    with open(
        get_architecture_path("experimental.soap_bpnn") / "schema-hypers.json", "r"
    ) as f:
        return json.load(f)


def test_validate_valid():
    instance = {
        "name": "experimental.soap_bpnn",
        "training": {"num_epochs": 1, "batch_size": 2},
    }
    validate(instance=instance, schema=schema())


def test_validate_single_suggestion():
    """Two invalid names; one to random that a useful suggestion can be given."""
    instance = {
        "name": "experimental.soap_bpnn",
        "training": {"nasdasd": 1, "batch_sizes": 2},
    }
    match = (
        r"Additional properties are not allowed \('batch_sizes', 'nasdasd' were "
        r"unexpected\). Do you mean 'batch_size'?"
    )
    with pytest.raises(ValidationError, match=match):
        validate(instance=instance, schema=schema())


def test_validate_multi_suggestion():
    instance = {
        "name": "experimental.soap_bpnn",
        "training": {"num_epoch": 1, "batch_sizes": 2},
    }
    match = (
        r"Additional properties are not allowed \('batch_sizes', 'num_epoch' "
        r"were unexpected\). Do you mean"
    )
    with pytest.raises(ValidationError, match=match):
        validate(instance=instance, schema=schema())
