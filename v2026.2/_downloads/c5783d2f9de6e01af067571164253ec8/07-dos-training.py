"""
Training a DOS model
========================

This tutorial demonstrates how to train a model for the prediction
of the electronic density of states (DOS), while accounting for the unique properties
of the DOS using the :ref:`masked dos loss function <dos-loss>`. This procedure can be
used to train PET-MAD-DOS, a universal model for the electronic density
of states. (https://arxiv.org/abs/2508.17418)
"""

# %%
#

import subprocess

import ase
import ase.io
import numpy as np


# %%
#
# Data Loading
# ----------------
#
# In order to use the masked dos loss function, we need to extract the DOS,
# the mask, and prepare them in a way to facilitate the use of extra targets
# during training. The extra targets parameter gives the model freedom to shift
# the energy reference of the target DOS, which is important as the energy
# reference is ill-defined for bulk systems. In this example, we will demonstrate
# the entire data processing pipeline, using 200 extra targets, starting from the
# eigenvalues and k-point weights obtained from a DFT calculation.
#
n_extra_targets = 200
structures = ase.io.read("DOS_structures.xyz", ":")
# Each structure contains the eigenvalues and k-point weights as arrays in info.
eigenvalues = []
kweights = []
for structure_i in structures:
    eigenvalues.append(structure_i.info["eigenvalues"])  # shape (n_kpoints, n_bands)
    kweights.append(
        structure_i.info["k-point weights"]
    )  # shape (n_kpoints,), sums to 2 in this example

for index, eigenvalue_i in enumerate(eigenvalues):
    print("Range of structure ", index)
    print(np.min(eigenvalue_i), np.max(eigenvalue_i))

# Here we can see that the eigenvalue range for both structures differ
# very significantly from one another which complicates training, which
# is why we introduce the mask to focus the loss on the relevant energy
# ranges for each structure.

# %%
#
# Data Processing
# ----------------
#
# Next, we need to process the eigenvalues and k-point weights to obtain
# the DOS and the mask. We will use a Gaussian smearing to compute the
# DOS on a fixed energy grid.# The mask will be defined as 1 in the energy
# range where we are confident that all the relevant eigenvalues have been
# computed, and 0 elsewhere. In this example, we define the upper bound of
# the confident energy range as 0.9eV below the minimum eigenvalue of the
# highest energy band in the structure. First, let us define the fixed
# energy grid and pick a smearing width of 0.3eV.
smearing = 0.3
all_eigenvalues = np.concatenate([i.flatten() for i in eigenvalues])
e_min = np.min(all_eigenvalues) - 1.5
e_max = np.max(all_eigenvalues) + 1.5

energy_grid = np.arange(e_min, e_max, 0.05)

# Here we chose to use a grid of 0.05eV spacing and we pick the energy
# grid to extend 1.5eV, representing 5 standard deviations, beyond the
# min and max eigenvalues across all structures.

# After defining the energy grid, we can now compute the DOS and the
# mask for each structure.
for index in range(len(structures)):
    confident_energy_upper_bound = (
        np.min(eigenvalues[index][:, -1]) - 0.9
    )  # 0.9eV below the minimum of the highest band
    normalization = 1 / np.sqrt(
        2 * np.pi * smearing**2
    )  # Gaussian normalization factor
    eigenvalues_i = eigenvalues[index].flatten()
    # Flatten eigenvalues to shape (n_kpoints * n_bands,)
    kweights_i = kweights[index].repeat(eigenvalues[index].shape[1])
    # Ensure that the eigenvalues are mapped to the correct weight,
    # shape (n_kpoints * n_bands,)

    dos_i = (
        np.sum(
            kweights_i[:, None]
            * (np.exp(-0.5 * ((energy_grid - eigenvalues_i[:, None]) / smearing) ** 2)),
            axis=0,
        )
        * normalization
    )  # Apply Gaussian smearing and sum contributions
    mask_i = (energy_grid <= confident_energy_upper_bound).astype(
        int
    )  # Define the mask

    # padding the dos and mask with zeros in front based on extra targets,
    # these values will be ignored during loss computation
    dos_i_padded = np.concatenate([np.zeros(n_extra_targets), dos_i])
    mask_i_padded = np.concatenate([np.zeros(n_extra_targets), mask_i])

    # Store the dos and mask in the structure info for later use during training
    structures[index].info["dos"] = dos_i_padded.astype(np.float32)
    structures[index].info["dos_mask"] = mask_i_padded.astype(np.float32)


# Write the structures to an xyz file
ase.io.write("DOS.xyz", structures)

# %%
#
# Training the model
# ------------------
#
# The dataset is now ready for training. You can now provide it to ``metatrain`` and
# train your DOS model!
#
# For example, you can use the following options file:
#
# .. literalinclude:: options-dos.yaml
#    :language: yaml

# We disable composition contributions because it is difficult to fit the DOS
# using a composition model, accounting for the ill-defined energy reference of
# the DOS. ``scale_targets`` is also set to false because it does not support masks.
# For details regarding the parameters of the loss function, please refer
# to the :ref:`masked dos loss function <dos-loss>` documentation. Additionally,
# the mask should be provided as extra data and share the same name as the target
# DOS with a "_mask" suffix. Due to the small dataset in this example, we set the
# validation set to be identical to the train set. In practice, you should use a
# separate validation set or set it as a fraction of the training set.

# Here, we run training as a subprocess, in reality you
# would run this from the command line as ``mtt train options-dos.yaml``.
subprocess.run(["mtt", "train", "options-dos.yaml"], check=True)
