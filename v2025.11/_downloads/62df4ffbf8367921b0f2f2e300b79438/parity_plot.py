"""
Model validation with parity plots for energies and forces
==========================================================

This tutorial shows how to visualise your model output using parity plots. In the
:ref:`train-from-scratch` we learned how to evaluate a trained model on a test set and
save the results to an output file. Here we will show how to create parity plots from
these results.

"""

# %%
# Import necessary libraries
import ase.io
import chemiscope
import matplotlib.pyplot as plt
import numpy as np


# %%
# Load the target and predicted data
targets = ase.io.read(
    "../train_from_scratch/ethanol_reduced_100.xyz", ":"
)  # reference data (ground truth)
predictions = ase.io.read("output.xyz", ":")  # predicted data from the model


# %%
# Extract the energies from the loaded frames
e_targets = np.array(
    [frame.get_total_energy() / len(frame) for frame in targets]
)  # target energies
e_predictions = np.array(
    [frame.get_total_energy() / len(frame) for frame in predictions]
)  # predicted energies
f_targets = np.array(
    [frame.get_forces().flatten() for frame in targets]
).flatten()  # target forces
f_predictions = np.array(
    [frame.get_forces().flatten() for frame in predictions]
).flatten()  # predicted forces


# %%
# Create parity plots to compare predicted vs. target energies and forces
# -----------------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Parity plot for energies
axs[0].scatter(e_targets, e_predictions)
axs[0].axline((np.min(e_targets), np.min(e_targets)), slope=1, ls="--", color="red")
axs[0].set_xlabel("Target energy / kcal")
axs[0].set_ylabel("Predicted energy / kcal")
min_e = np.min(np.array([e_targets, e_predictions])) - 2
max_e = np.max(np.array([e_targets, e_predictions])) + 2
axs[0].set_xlim([min_e, max_e])
axs[0].set_ylim([min_e, max_e])
axs[0].set_title("Energy Parity Plot")

# Parity plot for forces
axs[1].scatter(f_targets, f_predictions, alpha=0.5)
axs[1].axline((np.min(f_targets), np.min(f_targets)), slope=1, ls="--", color="red")
axs[1].set_xlabel("Target force / kcal/Å")
axs[1].set_ylabel("Predicted force / kcal/Å")
min_f = np.min(np.array([f_targets, f_predictions])) - 2
max_f = np.max(np.array([f_targets, f_predictions])) + 2
axs[1].set_xlim([min_f, max_f])
axs[1].set_ylim([min_f, max_f])
axs[1].set_title("Force Parity Plot")

plt.tight_layout()
plt.show()

print(
    "RMSE energy (per atom):",
    np.sqrt(np.mean((e_targets - e_predictions) ** 2)),
    "kcal",
)
print("RMSE forces:", np.sqrt(np.mean((f_targets - f_predictions) ** 2)), "kcal/Å   ")
# %%
# The results are a bit poor here because the model was not trained well enough and
# was created only for demonstration purposes. In the case of a well-trained model, the
# points should be closer to the diagonal line.

# %%
# Check outliers with ``Chemiscope``
# ----------------------------------
# With the approach above, you can inspect the whole dataset, but it might be difficult
# to identify outliers. `Chemiscope <https://chemiscope.org/docs/index.html>` is a
# visualisation tool, allowing you to explore the dataset interactively. The following
# example shows how to use it to check the structure of probable outliers and the atomic
# forces.

for frame in targets + predictions:
    frame.arrays["forces"] = frame.get_forces()
# a workaround, because the chemiscope interface for getting forces is broken with ASE
# 3.23

# %%
# Plot the energy parity plot with Chemiscope. This can be rendered as a widget in a
# Jupyter notebook.
cs = chemiscope.show(
    targets,  # reading structures from the dataset
    properties={
        "Target energy": {"values": e_targets, "target": "structure", "units": "kcal"},
        "Predicted energy": {
            "values": e_predictions,
            "target": "structure",
            "units": "kcal",
        },
    },  # plotting the energy parity plot
    mode="default",
    shapes={
        "target_forces": chemiscope.ase_vectors_to_arrows(
            targets, key="forces", scale=0.05, radius=0.15
        ),
        "predicted_forces": chemiscope.ase_vectors_to_arrows(
            predictions, key="forces", scale=0.05, radius=0.15
        ),
    },  # plotting the atomic forces
    settings=chemiscope.quick_settings(
        trajectory=True,
        map_settings={"joinPoints": False},
        structure_settings={
            "unitCell": True,
            "environments": {"activated": False},
            "shape": "predicted_forces",  # show predicted forces by defalut
        },
    ),
)
cs

# %%
# You can check the structures by clicking the red dots on the parity plot, or
# dragging the bar in the bottom right corner. Currently, plotting the diagonal line is
# not supported in chemiscope, so please check the parity plot above to see the
# outliers.
#
# The atomic forces are shown in arrows. The predicted forces are shown here. The target
# forces can be toggled by clicking the "target_forces" option in the menu in the upper
# right corner of the right panel.
