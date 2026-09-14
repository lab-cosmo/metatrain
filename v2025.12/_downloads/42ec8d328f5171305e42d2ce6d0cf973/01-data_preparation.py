"""
How to prepare data for training
================================

.. attention::

    This tutorial is only relevant for users who need to prepare their data from scratch
    from several files or for big datasets. If you already have your data in a common
    file format (like XYZ or `ASE database`_), you can skip this tutorial and directly
    start training.

.. _ASE database: https://ase-lib.org/ase/db/db.html

XYZ, ASE databases, and also from metrain's
:class:`metatrain.utils.data.dataset.DiskDataset <DiskDataset>` file.

For the small datasets (<10k structures), you can simply provide an XYZ file or an ASE
database to ``metatrain``, and it will handle the data loading for you. Large datasets
(>10k structures) may not fit into the GPU memory. In such cases, it is useful to
pre-process the dataset, save it to disk and load it on the fly during training.

In this tutorial, we will show how to prepare data for training using three different
formats. You can choose the one that best fits your needs.

We start by importing the necessary packages.
"""

# %%
#
import subprocess
from pathlib import Path

import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


# %%
# Create a XYZ training file (small datasets)
# -------------------------------------------
#
# First, we will show how to create a XYZ file with fields corresponding to the target
# properties. On modern HPC systems, this format is suitable for datasets up to around
# 1M structures. As an example, we will use 100 structures from a file read by ASE_.
# Since files from reference calculations may be located in different directories, we
# first create a list of all path that we want to read from. Here, for simplicity, we
# assume that all files are located in the same directory.
#
# .. _ASE: https://ase-lib.org/

filelist = 100 * ["qm9_reduced_100.xyz"]

# %%
#
# We will now read the structures using the ASE package. Check the ase documentation for
# more details on how to read different file formats. Instead of creating the ``atoms``
# object by reading from disk, you can also create an
# :class:`ase.Atoms` object containing the chemical ``symbols``, ``positions``, the
# ``cell`` and the periodic boundary conditions (``pbc``) by hand using its constructor.
#
# .. hint::
#
#   If a property is not read by the :func:`ase.io.read` function, you can add custom
#   scalar properties to the ``info`` dictionary. Vector properties (e.g. forces) can be
#   added to the ``arrays`` dictionary. Tensor properties (e.g. stress) must
#   be flattened before adding them to the ``arrays`` dictionary.

frames = []
for i, fname in enumerate(filelist):
    atoms = ase.io.read(fname, index=i)

    n_atoms = len(atoms)
    # scalar
    atoms.info["U0"] = -100.0
    # vector
    atoms.arrays["forces"] = np.zeros((n_atoms, 3))
    # tensor
    atoms.arrays["my_tensor"] = np.zeros((n_atoms, 3, 3)).reshape(n_atoms, 9)

    frames.append(atoms)

ase.io.write("data.xyz", frames)

# %%
#
# .. note::
#
#   The names of the added properties (like, ``U0``, etc.) must be referenced correctly
#   in the ``options.yaml`` file.
#
# Create a ``DiskDataset`` (large datasets)
# -----------------------------------------
#
# In addition to the systems and targets (as above), we also save the neighbor
# lists that the model will use during training. We first create the writer object that
# will write the data to a zip file.

disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")

# %%
#
# Then we loop over all structures, convert them to the internal torch format using
# :func:`metatomic.torch.systems_to_torch`, compute the neighbor lists using
# :func:`metatrain.utils.neighbor_lists.get_system_with_neighbor_lists` and write
# everything to disk using the writer's ``write()`` method.

for i, fname in enumerate(filelist):
    atoms = ase.io.read(fname, index=i)

    system = systems_to_torch(atoms, dtype=torch.float64)
    system = get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    energy = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[atoms.info["U0"]]], dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[i]]),
                ),
                components=[],
                properties=Labels("energy", torch.tensor([[0]])),
            )
        ],
    )
    disk_dataset_writer.write([system], {"energy": energy})

disk_dataset_writer.finish()

# %%
#
# Alternatively, you can also write the whole dataset at once, which might be more
# efficient (but also potentially run into memory issues). We use the same ``frames``
# that we created above.

disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100_all_at_once.zip")

systems = systems_to_torch(frames, dtype=torch.float64)
systems = [
    get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    for system in systems
]
energy = TensorMap(
    keys=Labels.single(),
    blocks=[
        TensorBlock(
            values=torch.tensor(
                [frame.info["U0"] for frame in frames], dtype=torch.float64
            ).reshape(-1, 1),
            samples=Labels.range("system", len(frames)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
    ],
)

disk_dataset_writer.write(systems, {"energy": energy})
disk_dataset_writer.finish()

# %%
#
# The dataset is saved to disk. You can now provide it to ``metatrain`` as a
# dataset to train from, simply by replacing your ``.xyz`` file with the newly created
# zip file (e.g. ``read_from: qm9_reduced_100.zip``).
#
# Create a ``MemmapDataset`` (large datasets, parallel filesystems)
# -----------------------------------------------------------------
#
# If your dataset is large and you are using a parallel filesystem (e.g. on an HPC
# cluster), it is recommended to use a ``MemmapDataset`` instead of a ``DiskDataset``.
# The ``MemmapDataset`` stores the data inside memory-mapped numpy arrays instead of a
# zip file. Reading from this format avoids I/O bottlenecks, but it does not support
# spherical targets or storing neighbor lists.
#
# As an example, we will use 100 structures from a dataset of carbon structures. The
# numpy arrays must be saved inside a directory, using the following format.

structures = ase.io.read("carbon_reduced_100.xyz", index=":")

root = Path("carbon_reduced_100_memmap/")
root.mkdir(exist_ok=True)

ns_path = root / "ns.npy"
na_path = root / "na.npy"
a_path = root / "a.bin"
x_path = root / "x.bin"
c_path = root / "c.bin"
e_path = root / "e.bin"
f_path = root / "f.bin"
s_path = root / "s.bin"

ns = len(structures)
na = np.cumsum(np.array([0] + [len(s) for s in structures], dtype=np.int64))
np.save(ns_path, ns)
np.save(na_path, na)

a_mm = np.memmap(a_path, dtype="int32", mode="w+", shape=(na[-1],))
x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(na[-1], 3))
c_mm = np.memmap(c_path, dtype="float32", mode="w+", shape=(ns, 3, 3))
e_mm = np.memmap(e_path, dtype="float32", mode="w+", shape=(ns, 1))
f_mm = np.memmap(f_path, dtype="float32", mode="w+", shape=(na[-1], 3))
s_mm = np.memmap(s_path, dtype="float32", mode="w+", shape=(ns, 3, 3))

for i, s in enumerate(structures):
    a_mm[na[i] : na[i + 1]] = s.numbers
    x_mm[na[i] : na[i + 1]] = s.get_positions()
    c_mm[i] = s.get_cell()[:]
    e_mm[i] = s.get_potential_energy()
    f_mm[na[i] : na[i + 1]] = s.arrays["force"]
    s_mm[i] = -s.info["virial"] / s.get_volume()

a_mm.flush()
x_mm.flush()
c_mm.flush()
e_mm.flush()
f_mm.flush()
s_mm.flush()

# %%
#
# The dataset is saved to disk. You can now provide it to ``metatrain`` as a
# dataset to train from, simply by specifying the newly created
# directory as the path from which to read the systems
# (e.g. ``read_from: carbon_reduced_100_memmap/``).
#
# For example, you can use the following options file:
#
# .. literalinclude:: options-memmap.yaml
#    :language: yaml

# Here, we run training as a subprocess, in reality you would run this from the command
# line as ``mtt train options-memmap.yaml``.
subprocess.run(["mtt", "train", "options-memmap.yaml"], check=True)
