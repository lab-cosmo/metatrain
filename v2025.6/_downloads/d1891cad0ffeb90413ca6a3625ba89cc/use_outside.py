"""
Using metatrain architectures outside of metatrain
==================================================

This tutorial demonstrates how to use one of metatrain's implemented architectures
outside of metatrain. This will be done by taking internal representations of a
NanoPET model (as an example) and using them inside a user-defined torch ``Module``.

Only architectures which can output internal representations ("features" output) can
be used in this way.
"""

# %%
#

import torch
from metatensor.torch.atomistic import ModelOutput

from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, read_systems
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


# %%
#
# Read some sample systems. Metatrain always reads systems in float64, while torch
# uses float32 by default. We will convert the systems to float32.

systems = read_systems("qm9_reduced_100.xyz")
systems = [s.to(torch.float32) for s in systems]


# %%
#
# Define the custom model using the NanoPET architecture as a building block.
# The dummy architecture here adds a linear layer and a tanh activation function
# on top of the NanoPET model.


class NanoPETWithTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nanopet = NanoPET(
            get_default_hypers("experimental.nanopet")["model"],
            DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1, 6, 7, 8, 9],
                targets={},
            ),
        )
        self.linear = torch.nn.Linear(128, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, systems):
        model_outputs = self.nanopet(
            systems,
            {"features": ModelOutput()},
            # ModelOutput(per_atom=True) would give per-atom features
        )
        features = model_outputs["features"].block().values
        return self.tanh(self.linear(features))


# %%
#
# Now we can train the custom model. Here is one training step executed with
# some random targets.
my_targets = torch.randn(100, 1)

# instantiate the model
model = NanoPETWithTanh()

# all metatrain models require neighbor lists to be present in the input systems
systems = [
    get_system_with_neighbor_lists(sys, get_requested_neighbor_lists(model))
    for sys in systems
]

# define an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# this is one training step
predictions = model(systems)
loss = torch.nn.functional.mse_loss(predictions, my_targets)
loss.backward()
optimizer.step()
