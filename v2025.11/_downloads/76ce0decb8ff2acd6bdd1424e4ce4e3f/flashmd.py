"""
Training a FlashMD model
========================

This tutorial demonstrates how to train a FlashMD model for the direct prediction
of molecular dynamics. This type of model affords faster MD simulations compared to
MLIPs by a factor between 10 and 30 (https://arxiv.org/abs/2505.19350).
"""

# %%
#

import copy
import subprocess

import ase
import ase.build
import ase.io
import ase.units
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


# %%
#
# Data generation
# ---------------
#
# FlashMD models train on molecular dynamics trajectories in the NVE ensemble (i.e.,
# most often with the velocity Verlet integrator). These trajectories can be generated
# with almost any MD code (i-PI, LAMMPS, etc.). Here, for simplicity, we will use ASE
# and its built-in EMT potential. In reality, you might want to use a more accurate
# baseline such as ab initio MD or a machine-learned interatomic potential (MLIP).

# We start by creating a simple system (a small box of aluminum).
atoms = ase.build.bulk("Al", "fcc", cubic=True) * (2, 2, 2)

# We first equilibrate the system at 300K using a Langevin thermostat.
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
atoms.calc = EMT()
dyn = Langevin(
    atoms, 2 * ase.units.fs, temperature_K=300, friction=1 / (100 * ase.units.fs)
)
dyn.run(1000)  # 2 ps equilibration (around 10 ps is better in practice)

# Then, we run a production simulation in the NVE ensemble.
trajectory = []


def store_trajectory():
    trajectory.append(copy.deepcopy(atoms))


dyn = VelocityVerlet(atoms, 1 * ase.units.fs)
dyn.attach(store_trajectory, interval=1)
dyn.run(2000)  # 2 ps NVE run


# %%
#
# Data preparation
# ----------------
#
# Now, we need to generate the training data from the trajectory. FlashMD models
# require future positions and momenta as targets. We will save them in an `.xyz`
# file under the `future_positions` and `future_momenta` keys.

# The FlashMD model will be trained to predict 30 steps into the future, i.e., 30 fs
# since we ran the reference simulation with a time step of 1 fs. For this type
# of system, FlashMD is expected to perform well up to around 60-80 fs.
time_lag = 30

# We pick starting structures that are 200 steps apart. To avoid wasting training
# structures, this should be set to be around the expected velocity-velocity
# autocorrelation time for the system. This is the time scale that quantifies how long
# it takes for the system to forget its original velocities.
spacing = 200


def get_structure_for_dataset(frame_now, frame_ahead):
    s = copy.deepcopy(frame_now)
    s.arrays["future_positions"] = frame_ahead.get_positions()
    s.arrays["future_momenta"] = frame_ahead.get_momenta()
    return s


structures_for_dataset = []
for i in range(0, len(trajectory) - time_lag, spacing):
    frame_now = trajectory[i]
    frame_ahead = trajectory[i + time_lag]
    s = get_structure_for_dataset(frame_now, frame_ahead)
    structures_for_dataset.append(s)

    # Here, we also add the time-reversed pair (optional)
    frame_now_trev = copy.deepcopy(frame_now)
    frame_ahead_trev = copy.deepcopy(frame_ahead)
    frame_now_trev.set_momenta(-frame_now_trev.get_momenta())
    frame_ahead_trev.set_momenta(-frame_ahead_trev.get_momenta())
    s = get_structure_for_dataset(frame_ahead_trev, frame_now_trev)
    structures_for_dataset.append(s)

# Write the structures to an xyz file
ase.io.write("flashmd.xyz", structures_for_dataset)

# %%
#
# Training the model
# ------------------
#
# The dataset is now ready for training. You can now provide it to ``metatrain`` and
# train your FlashMD model!
#
# For example, you can use the following options file:
#
# .. literalinclude:: options.yaml
#    :language: yaml

subprocess.run(["mtt", "train", "options.yaml"])
