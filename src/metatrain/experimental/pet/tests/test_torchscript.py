import pytest
import torch
from pet.hypers import Hypers
from pet.pet import PET

from metatrain.experimental.pet import PET as WrappedPET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict


DEFAULT_HYPERS = get_default_hypers("experimental.pet")


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    match = (
        "The TorchScript type system doesn't support instance-level annotations on "
        "empty"
    )
    with pytest.warns(UserWarning, match=match):
        torch.jit.script(model)


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    match = (
        "The TorchScript type system doesn't support instance-level annotations on "
        "empty"
    )
    with pytest.warns(UserWarning, match=match):
        torch.jit.script(model)

    torch.jit.save(
        torch.jit.script(model),
        "pet.pt",
    )
    torch.jit.load("pet.pt")
