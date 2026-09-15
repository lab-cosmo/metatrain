import os
from pathlib import Path

import pytest
import torch
from metatomic.torch import AtomisticModel

from metatrain.soap_bpnn.model import SoapBpnn
from metatrain.utils.io import (
    check_file_extension,
    is_exported_file,
    load_model,
    model_from_checkpoint,
    trainer_from_checkpoint,
)


@pytest.fixture(scope="module", params=["pathlib", "str", "file_url"])
def load_path(request, MODEL_PATH):
    """Fixture that provides the model checkpoint path in different formats.

    This is meant to be used in the `test_load_model_exported` test to
    check that loading works with all of these formats.
    """
    if request.param == "pathlib":
        return MODEL_PATH
    elif request.param == "str":
        return str(MODEL_PATH)
    elif request.param == "file_url":
        return f"file:{str(MODEL_PATH)}"


@pytest.fixture(scope="module", params=["pathlib", "str", "file_url"])
def load_path_ckpt(MODEL_PATH, request):
    """Fixture that provides the model checkpoint path in different formats.

    This is meant to be used in the `test_load_model_checkpoint` test to
    check that loading works with all of these formats.
    """
    ckpt_path = MODEL_PATH.with_suffix(".ckpt")
    if request.param == "pathlib":
        return ckpt_path
    elif request.param == "str":
        return str(ckpt_path)
    elif request.param == "file_url":
        return f"file:{str(ckpt_path)}"


@pytest.mark.parametrize("filename", ["example.txt", Path("example.txt")])
def test_check_suffix(filename):
    result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


@pytest.mark.parametrize("filename", ["example", Path("example")])
def test_warning_on_missing_suffix(filename):
    match = r"The file name should have a '\.txt' file extension."
    with pytest.warns(UserWarning, match=match):
        result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


def test_is_exported_file(MODEL_PATH):
    assert is_exported_file(MODEL_PATH)
    assert not is_exported_file(MODEL_PATH.with_suffix(".ckpt"))


def test_load_model_checkpoint(load_path_ckpt):
    model = load_model(load_path_ckpt)
    assert type(model) is SoapBpnn

    # TODO: test that weights are the expected if loading with `context == 'export'`.
    # One can use `list(model.bpnn[0].parameters())[0][0]` to get some weights. But,
    # currently weights of the `"export"` and the `"restart"` context are the same...


def test_load_model_checkpoint_newer_version(monkeypatch, tmp_path, MODEL_PATH_64_BIT):
    """A checkpoint version newer than the installed architecture should tell the
    user to upgrade metatrain, instead of trying (and failing) to upgrade the
    checkpoint."""
    monkeypatch.chdir(tmp_path)
    path = MODEL_PATH_64_BIT.with_suffix(".ckpt")
    model = torch.load(path, weights_only=False, map_location="cpu")
    model["model_ckpt_version"] = 5000000

    file = "model-version-5000000.ckpt"
    torch.save(model, file)

    message = (
        "Unable to load the model checkpoint for the 'soap_bpnn' architecture: the "
        r"checkpoint uses checkpoint format version 5000000, but the installed "
        r"'soap_bpnn' architecture only supports up to version \d+\. You are using "
        r"metatrain version .*, which is too old to read this checkpoint\. Please "
        "upgrade metatrain to a newer version."
    )
    with pytest.raises(RuntimeError, match=message):
        checkpoint = torch.load(file, weights_only=False, map_location="cpu")
        model_from_checkpoint(checkpoint, context="restart")


def test_load_model_checkpoint_older_version_upgrade_fails(
    monkeypatch, tmp_path, MODEL_PATH_64_BIT
):
    """A checkpoint version older than the installed architecture, for which the
    upgrade path fails, should keep raising the generic upgrade-failure error."""
    monkeypatch.chdir(tmp_path)
    path = MODEL_PATH_64_BIT.with_suffix(".ckpt")
    model = torch.load(path, weights_only=False, map_location="cpu")
    model["model_ckpt_version"] = 0

    file = "model-version-0.ckpt"
    torch.save(model, file)

    message = (
        "Unable to load the model checkpoint for the 'soap_bpnn' architecture: the "
        r"checkpoint is using version 0, while the current version is \d+; "
        "and trying to upgrade the checkpoint failed."
    )
    with pytest.raises(RuntimeError, match=message):
        checkpoint = torch.load(file, weights_only=False, map_location="cpu")
        model_from_checkpoint(checkpoint, context="restart")


def test_load_trainer_checkpoint_newer_version(
    monkeypatch, tmp_path, MODEL_PATH_64_BIT
):
    """A trainer checkpoint version newer than the installed architecture should tell the
    user to upgrade metatrain, instead of trying (and failing) to upgrade the
    checkpoint."""
    monkeypatch.chdir(tmp_path)
    path = MODEL_PATH_64_BIT.with_suffix(".ckpt")
    model = torch.load(path, weights_only=False, map_location="cpu")
    model["trainer_ckpt_version"] = 5000000

    file = "model-version-5000000.ckpt"
    torch.save(model, file)

    message = (
        "Unable to load the trainer checkpoint for the 'soap_bpnn' architecture: the "
        r"checkpoint uses checkpoint format version 5000000, but the installed "
        r"'soap_bpnn' architecture only supports up to version \d+\. You are using "
        r"metatrain version .*, which is too old to read this checkpoint\. Please "
        "upgrade metatrain to a newer version."
    )

    with pytest.raises(RuntimeError, match=message):
        checkpoint = torch.load(file, weights_only=False, map_location="cpu")
        trainer_from_checkpoint(checkpoint, context="restart", hypers={})


def test_load_model_exported(load_path):
    model = load_model(load_path)
    assert type(model) is AtomisticModel


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_load_model_yaml(suffix):
    match = f"path 'foo{suffix}' seems to be a YAML option file and not a model"
    with pytest.raises(ValueError, match=match):
        load_model(f"foo{suffix}")


def test_load_model_token():
    """Test that the export cli succeeds when exporting a private
    model from HuggingFace."""

    hf_token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if hf_token is None or len(hf_token) == 0:
        pytest.skip("HuggingFace token not found in environment.")

    path = "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt"
    load_model(path, hf_token=hf_token)


def test_load_model_token_invalid_url_style():
    hf_token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if hf_token is None or len(hf_token) == 0:
        pytest.skip("HuggingFace token not found in environment.")

    # change `resolve` to ``foo`` to make the URL scheme invalid
    path = "https://huggingface.co/metatensor/metatrain-test/foo/main/model.ckpt"

    with pytest.raises(
        ValueError,
        match=f"URL '{path}' has an invalid format for the Hugging Face Hub.",
    ):
        load_model(path, hf_token=hf_token)
