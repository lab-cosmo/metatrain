""".. _llprensembleexample:

Generating and training an LLPR-derived shallow ensemble model
==============================================================

This tutorial demonstrates how to generate and, optionally, further train an
LLPR-derived shallow ensemble model using metatrain. Building on the LLPR approach, this
more advanced technique allows for 1) generation of a last-layer ensemble model from an
LLPR model and 2) gradient-based tuning of ensemble weights using a negative
log-likelihood (NLL) loss, often leading to improved uncertainty estimates at the cost
of further training.

We first train a baseline model without uncertainties:

.. literalinclude:: options-model.yaml
   :language: yaml

Then we create an LLPR ensemble model. This involves creating the LLPR model, which is
very cheap (one pass through the training data without backpropagation), and then
sampling last-layer ensemble weights using the LLPR covariance (extremely cheap), as
explained in https://arxiv.org/html/2403.02251v1. Specifying `num_ensemble_members`
enables the latter step in addition to the basic LLPR model.

.. literalinclude:: options-llpr-ensemble.yaml
   :language: yaml

In addition, you can decide to perform further backpropagation-based training on the
resulting shallow ensemble, which is more expensive but can lead to better uncertainty
estimates. This is done by setting `num_epochs` to the number of epochs that you
want to train for.

.. literalinclude:: options-llpr-ensemble-train.yaml
   :language: yaml

You can train these models yourself with the following code:
"""
# %%
#

import subprocess

import ase.io
import numpy as np
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator


# %%
#
# We first train the baseline model without uncertainties, then train the LLPR
# ensemble models.

# Here, we run training as a subprocess. In practice, you would run this from
# the command line, e.g., ``mtt train options-model.yaml -o model.pt``.

print("Training baseline model...")
subprocess.run(["mtt", "train", "options-model.yaml", "-o", "model.pt"], check=True)

print("Training LLPR ensemble model...")
subprocess.run(
    ["mtt", "train", "options-llpr-ensemble.yaml", "-o", "model-llpr-ens.pt"],
    check=True,
)

print("Training LLPR ensemble model with further backpropagation...")
subprocess.run(
    ["mtt", "train", "options-llpr-ensemble-train.yaml", "-o", "model-llpr-ens-tr.pt"],
    check=True,
)
# %%
#
# You can now use the uncertainties from the LLPR, as well as the ensemble model,
# as follows.

# Load some test structures
structures = ase.io.read("ethanol_reduced_100.xyz", ":5")

# Load the ensemble-trained model
calc = MetatomicCalculator("model-llpr-ens.pt", extensions_directory="extensions/")

# Get predictions with both ensemble and analytical uncertainties
# (note that all these quantities are also available per-atom with ``per_atom=True``)
predictions = calc.run_model(
    structures,
    {
        "energy": ModelOutput(per_atom=False),
        "energy_uncertainty": ModelOutput(per_atom=False),  # LLPR analytical
        "energy_ensemble": ModelOutput(per_atom=False),  # ensemble predictions
    },
)

energies = predictions["energy"].block().values.squeeze().cpu().numpy()
llpr_uncertainties = (
    predictions["energy_uncertainty"].block().values.squeeze().cpu().numpy()
)
ensemble_predictions = (
    predictions["energy_ensemble"].block().values.squeeze().cpu().numpy()
)

# Calculate ensemble mean and standard deviation
ensemble_mean = np.mean(ensemble_predictions, axis=1)
ensemble_std = np.std(ensemble_predictions, axis=1)

print(f"Energies: {energies}")
print(f"LLPR analytical uncertainties: {llpr_uncertainties}")
print(f"Ensemble mean: {ensemble_mean}")
print(f"Ensemble std: {ensemble_std}")
