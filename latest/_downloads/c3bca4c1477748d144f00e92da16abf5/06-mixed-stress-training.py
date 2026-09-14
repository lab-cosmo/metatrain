"""
Training with Mixed Stress Structures
======================================

This tutorial demonstrates how to train a model on datasets containing structures
with and without stress information in the same training set. This is useful when
working with:

- Bulk materials with well-defined stress tensors
- Molecules where stress is not physically meaningful
- Slabs (surfaces) with non-periodic boundary conditions where stress is undefined

``metatrain`` handles this automatically by allowing structures with NaN stress values.
"""

# %%
#
# Creating a Mixed Dataset
# -------------------------
#
# We'll create a dataset containing three types of structures: bulk materials,
# molecules, and slabs. For structures where stress is not defined (molecules and
# slabs), we set the stress tensor to NaN.

import subprocess
import warnings

import ase.build
import ase.io
import numpy as np
from ase.calculators.emt import EMT


calculator = EMT()
structures = []

# Create bulk structures with valid stress
for _ in range(5):
    bulk = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    bulk.rattle(0.01)  # Small perturbation to make structures different
    bulk.calc = calculator
    bulk.info["energy"] = bulk.get_potential_energy()
    bulk.arrays["forces"] = bulk.get_forces()
    bulk.info["stress"] = bulk.get_stress(voigt=False)
    bulk.calc = None
    structures.append(bulk)

# Create molecules with NaN stress (stress not defined for molecules)
for i in range(5):
    molecule = ase.Atoms("Cu2", positions=[[0, 0, 0], [2.5 + 0.1 * i, 2.5, 2.5]])
    molecule.calc = calculator
    molecule.info["energy"] = molecule.get_potential_energy()
    molecule.arrays["forces"] = molecule.get_forces()
    molecule.info["stress"] = np.full(
        (3, 3), np.nan, dtype=np.float64
    )  # Use NaN for undefined stress
    molecule.calc = None
    structures.append(molecule)

# Create slabs with NaN stress (stress not defined for slabs with non-periodic BC)
for _ in range(5):
    slab = ase.build.fcc111("Cu", size=(2, 2, 4), vacuum=10.0)
    slab.pbc = (True, True, False)  # Periodic in xy, non-periodic in z
    slab.rattle(0.01)  # Small perturbation
    slab.calc = calculator
    slab.info["energy"] = slab.get_potential_energy()
    slab.arrays["forces"] = slab.get_forces()
    slab.info["stress"] = np.full(
        (3, 3), np.nan, dtype=np.float64
    )  # Use NaN for undefined stress
    slab.calc = None
    structures.append(slab)

# Write structures to file
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ase.io.write("mixed_structures.xyz", structures)

print(f"Created dataset with {len(structures)} structures:")
print("  - 5 bulk structures (with valid stress)")
print("  - 5 molecules (with NaN stress)")
print("  - 5 slabs (with NaN stress)")

# %%
#
# Configuration for Mixed Stress Training
# ----------------------------------------
#
# We use an options file that specifies training on energy, forces,
# and stress. The model will automatically handle structures with NaN stress
# during training.
#
# .. literalinclude:: options-mixed-stress.yaml
#    :language: yaml

# %%
#
# Training the Model
# ------------------
#
# Now we train the model using the mixed dataset. The training will proceed
# normally, with stress contributions to the loss only coming from structures
# where stress is well-defined (non-NaN values).

# Here, we run training as a subprocess, in reality you would run this from the command
# line as ``mtt train options-mixed-stress.yaml``.
subprocess.run(["mtt", "train", "options-mixed-stress.yaml"], check=True)

# %%
#
# Key Points
# ----------
#
# 1. **Use NaN for undefined stress**: When stress is not defined for a structure,
#    set the stress tensor to ``np.full((3, 3), np.nan, dtype=np.float64)``.
#
# 2. **Mixed training is automatic**: ``metatrain`` automatically handles the mixed
#    dataset during training, only computing stress loss for structures where stress
#    is defined (not NaN).
#
# 3. **All structures need stress field**: Even if stress is not defined for some
#    structures, the stress field must be present in the dataset. Use NaN values
#    for structures where stress is undefined.
#
# 4. **Force training works normally**: Forces can be trained for all structure types
#    (bulk, molecules, and slabs) without any special handling.
#
# This feature is particularly useful when:
#
# - Training models that need to work across different system types
# - Working with surface calculations where stress is only meaningful in periodic
#   directions
# - Combining datasets from different sources where stress information may not be
#   available for all structures
