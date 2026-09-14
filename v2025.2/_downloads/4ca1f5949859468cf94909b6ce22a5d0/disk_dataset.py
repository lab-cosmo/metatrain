"""
Saving a disk dataset
=====================

Large datasets may not fit into memory. In such cases, it is useful to save the
dataset to disk and load it on the fly during training. This example demonstrates
how to save a ``DiskDataset`` for this purpose. Metatrain will then be able to load
``DiskDataset`` objects saved in this way to execute on-the-fly data loading.
"""

# %%
#

import ase.io
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, systems_to_torch

from metatrain.utils.data import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


# %%
#
# As an example, we will use 100 structures from the QM9 dataset. In addition to the
# systems and targets (here the energy), we also need to save the neighbor lists that
# the model will use during training.

disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")
for i in range(100):
    frame = ase.io.read("qm9_reduced_100.xyz", index=i)
    system = systems_to_torch(frame, dtype=torch.float64)
    system = get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    energy = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[frame.info["U0"]]], dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[i]]),
                ),
                components=[],
                properties=Labels("energy", torch.tensor([[0]])),
            )
        ],
    )
    disk_dataset_writer.write_sample(system, {"energy": energy})
del disk_dataset_writer  # not necessary if the file ends here, but good in general

# %%
#
# The dataset is saved to disk. You can now provide it to ``metatrain`` as a
# dataset to train from, simply by replacing your ``.xyz`` file with the newly created
# zip file (e.g. ``read_from: qm9_reduced_100.zip``).
