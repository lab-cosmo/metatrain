"""
Training a Symplectic FlashMD Model
===================================

This tutorial demonstrates how to train a symplectic FlashMD model using the FlashMD
framework. Symplectic integrators are designed to preserve the geometric properties of
Hamiltonian dynamics, making them particularly suitable for long molecular dynamics
simulations. By leveraging symplectic integrators, we can achieve more accurate and
stable simulations over extended periods.
"""

# %%

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
# Dataset Creation
# ----------------
#
# We will create a dataset of molecular dynamics trajectories using ASE and its built-in
# EMT potential. The dataset will consist of atomic configurations, forces, and energies
# obtained from NVE simulations. In reality, you might want to use a more accurate
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
trajectory: list[ase.Atoms] = []


def store_trajectory():
    trajectory.append(copy.deepcopy(atoms))


dyn = VelocityVerlet(atoms, 1 * ase.units.fs)
dyn.attach(store_trajectory, interval=1)
dyn.run(2000)  # 2 ps NVE run

# %%
#
# Data Preparation
# ----------------
#
# Note that the data preparation process is similar to the one in the `04-flashmd.py`
# example, with one key difference. Instead of storing a phase-space point and its
# future state after one time step, we store the input to the symplectic fixed-point
# solver. The input is a midpoint that is mapped to the difference in positions and
# momenta after one time step.

time_lag = 32
spacing = 200


def get_structure_for_dataset(frame_now, frame_ahead):
    s = copy.deepcopy(frame_now)
    s.arrays["delta_positions"] = (
        frame_ahead.get_positions() - frame_now.get_positions()
    )
    s.arrays["delta_momenta"] = frame_ahead.get_momenta() - frame_now.get_momenta()
    s.set_positions(0.5 * (frame_now.get_positions() + frame_ahead.get_positions()))
    s.set_momenta(0.5 * (frame_now.get_momenta() + frame_ahead.get_momenta()))
    return s


structures_for_dataset = []
for i in range(0, len(trajectory) - time_lag, spacing):
    frame_now = trajectory[i]
    frame_ahead = trajectory[i + time_lag]
    s = get_structure_for_dataset(frame_now, frame_ahead)
    structures_for_dataset.append(s)
    frame_now_trev = copy.deepcopy(frame_now)
    frame_ahead_trev = copy.deepcopy(frame_ahead)
    frame_now_trev.set_momenta(-frame_now_trev.get_momenta())
    frame_ahead_trev.set_momenta(-frame_ahead_trev.get_momenta())
    s = get_structure_for_dataset(frame_ahead_trev, frame_now_trev)
    structures_for_dataset.append(s)

ase.io.write("midpoint-to-delta.xyz", structures_for_dataset)

# %%
#
# Model Training
# --------------
#
# We can now train a symplectic FlashMD model using the prepared dataset.
#
# For example, you can use the following options file:
#
# .. literalinclude:: options-flashmd-symplectic.yaml
#    :language: yaml

subprocess.run(["mtt", "train", "options-flashmd-symplectic.yaml"], check=True)
