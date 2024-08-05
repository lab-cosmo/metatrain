import matplotlib.pyplot as plt
import numpy as np
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    load_atomistic_model,
)

from metatrain.utils.data import Dataset, collate_fn, read_systems, read_targets
from metatrain.utils.llpr import LLPRUncertaintyModel
from metatrain.utils.loss import TensorMapDictLoss
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


model = load_atomistic_model("model.pt", extensions_directory="extensions/")
model = model.to("cuda")

train_systems = read_systems("train.xyz")
train_target_config = {
    "energy": {
        "quantity": "energy",
        "read_from": "train.xyz",
        "file_format": ".xyz",
        "reader": "ase",
        "key": "energy",
        "unit": "kcal/mol",
        "forces": {
            "read_from": "train.xyz",
            "file_format": ".xyz",
            "reader": "ase",
            "key": "forces",
        },
        "stress": {
            "read_from": "train.xyz",
            "file_format": ".xyz",
            "reader": "ase",
            "key": "stress",
        },
        "virial": False,
    },
}
train_targets, _ = read_targets(train_target_config)

valid_systems = read_systems("valid.xyz")
valid_target_config = {
    "energy": {
        "quantity": "energy",
        "read_from": "valid.xyz",
        "file_format": ".xyz",
        "reader": "ase",
        "key": "energy",
        "unit": "kcal/mol",
        "forces": {
            "read_from": "valid.xyz",
            "file_format": ".xyz",
            "reader": "ase",
            "key": "forces",
        },
        "stress": {
            "read_from": "valid.xyz",
            "file_format": ".xyz",
            "reader": "ase",
            "key": "stress",
        },
        "virial": False,
    },
}
valid_targets, _ = read_targets(valid_target_config)

test_systems = read_systems("test.xyz")
test_target_config = {
    "energy": {
        "quantity": "energy",
        "read_from": "test.xyz",
        "file_format": ".xyz",
        "reader": "ase",
        "key": "energy",
        "unit": "kcal/mol",
        "forces": {
            "read_from": "test.xyz",
            "file_format": ".xyz",
            "reader": "ase",
            "key": "forces",
        },
        "stress": {
            "read_from": "test.xyz",
            "file_format": ".xyz",
            "reader": "ase",
            "key": "stress",
        },
        "virial": False,
    },
}
test_targets, target_info = read_targets(test_target_config)

requested_neighbor_lists = model.requested_neighbor_lists()
train_systems = [
    get_system_with_neighbor_lists(system, requested_neighbor_lists)
    for system in train_systems
]
train_dataset = Dataset({"system": train_systems, **train_targets})
valid_systems = [
    get_system_with_neighbor_lists(system, requested_neighbor_lists)
    for system in valid_systems
]
valid_dataset = Dataset({"system": valid_systems, **valid_targets})
test_systems = [
    get_system_with_neighbor_lists(system, requested_neighbor_lists)
    for system in test_systems
]
test_dataset = Dataset({"system": test_systems, **test_targets})

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)

loss_weight_dict = {
    "energy": 1.0,
    "energy_positions_grad": 1.0,
    "energy_grain_grad": 1.0,
}
loss_fn = TensorMapDictLoss(loss_weight_dict)

llpr_model = LLPRUncertaintyModel(model)

print("Last layer parameters:")
parameters = []
for name, param in llpr_model.named_parameters():
    if "last_layers" in name:
        parameters.append(param)
        print(name)

llpr_model.compute_covariance_as_pseudo_hessian(
    train_dataloader, target_info, loss_fn, parameters
)
llpr_model.compute_inverse_covariance()
llpr_model.calibrate(valid_dataloader)

exported_model = MetatensorAtomisticModel(
    llpr_model.eval(),
    ModelMetadata(),
    llpr_model.capabilities,
)

evaluation_options = ModelEvaluationOptions(
    length_unit="angstrom",
    outputs={
        "mtt::aux::last_layer_features": ModelOutput(per_atom=False),
        "mtt::aux::energy_uncertainty": ModelOutput(per_atom=False),
        "energy": ModelOutput(per_atom=False),
    },
    selected_atoms=None,
)

force_errors = []
force_uncertainties = []

for batch in test_dataloader:
    systems, targets = batch
    systems = [system.to("cuda", torch.float64) for system in systems]
    for system in systems:
        system.positions.requires_grad = True
    targets = {name: tmap.to("cuda", torch.float64) for name, tmap in targets.items()}

    outputs = exported_model(systems, evaluation_options, check_consistency=True)
    energy = outputs["energy"].block().values
    energy_sum = torch.sum(energy)
    energy_sum.backward(retain_graph=True)

    predicted_forces = -torch.concatenate(
        [system.positions.grad.flatten() for system in systems]
    )
    true_forces = -targets["energy"].block().gradient("positions").values.flatten()

    force_error = (predicted_forces - true_forces) ** 2
    force_errors.append(force_error.detach().clone().cpu().numpy())

    last_layer_features = outputs["mtt::aux::last_layer_features"].block().values
    last_layer_features = torch.sum(last_layer_features, dim=0)
    ll_feature_grads = []
    for ll_feature in last_layer_features.reshape((-1,)):
        ll_feature_grad = torch.autograd.grad(
            ll_feature.reshape(()),
            [system.positions for system in systems],
            retain_graph=True,
        )
        ll_feature_grad = torch.concatenate(
            [ll_feature_g.flatten() for ll_feature_g in ll_feature_grad]
        )
        ll_feature_grads.append(ll_feature_grad)
    ll_feature_grads = torch.stack(ll_feature_grads, dim=1)

    force_uncertainty = torch.einsum(
        "if, fg, ig -> i",
        ll_feature_grads,
        exported_model._module.inv_covariance,
        ll_feature_grads,
    )
    force_uncertainties.append(force_uncertainty.detach().clone().cpu().numpy())

force_errors = np.concatenate(force_errors)
force_uncertainties = np.concatenate(force_uncertainties)


plt.scatter(force_uncertainties, force_errors, s=1)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Predicted variance")
plt.ylabel("Squared error")

plt.savefig("figure.pdf")
