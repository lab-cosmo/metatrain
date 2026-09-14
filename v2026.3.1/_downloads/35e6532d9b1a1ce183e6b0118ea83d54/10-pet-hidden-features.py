""".. _pet-hidden-features:

Extracting Hidden Layer Features from PET
==========================================

PET computes rich intermediate representations at every stage of its forward pass —
atom-type embeddings, embeddings of the neighborhood geometries, transformer outputs,
and more. This tutorial shows how to retrieve any of those tensors using the
``mtt::feature::`` prefix in the ``outputs`` dictionary passed to ``PET.forward()``.
Note that this only applies to non-exported models, i.e. checkpoints.

All outputs are returned as :class:`~metatensor.torch.TensorMap` objects, so they carry
full sample metadata (atom or pair indices, cell-shift vectors) and are immediately
compatible with the rest of the metatensor/metatomic ecosystem.

The tutorial is structured as follows:

1. :ref:`Standard outputs <standard-outputs>` — backbone feature, last-layer features,
   and ``energy``.
2. :ref:`Unprocessed backbone and last layer features <backbone-features>` — per-atom
   and per-pair tensors as used internally by the model.
3. :ref:`Raw featurizer inputs <featurizer-inputs>` — distances, displacement vectors,
   and element-type indices, tied back to the water geometry.
4. :ref:`Discovering available paths <discovering-paths>` — how to enumerate every
   capturable module path.
5. :ref:`Deep-dive into a GNN layer <gnn-deep-dive>` — capturing transformer internals
   and analysing how features evolve.

"""

# %%
#
# Setup: model and system
# -----------------------
#
# We initialise an untrained PET model (random weights) and build a single water
# molecule. The absolute values of the features will not be meaningful, but the shapes,
# sample labels, and structure of the returned TensorMaps are identical to those of a
# trained model.
#
# Here we use the default hypers for the PET model, but make some modifications to
# better illustrate the dimensions of the captured layer outputs.

import ase
import torch
from metatomic.torch import ModelOutput, NeighborListOptions, systems_to_torch

from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


dtype = torch.float64

hypers = get_default_hypers("pet")["model"]

# Modify some hypers related to the token sizes
hypers["d_pet"] = 128
hypers["d_node"] = 256
hypers["d_head"] = 64

# Print some of the key default model hypers for reference.
print("Key PET hypers:")
print(f"  d_pet: {hypers['d_pet']}")
print(f"  d_node: {hypers['d_node']}")
print(f"  d_head: {hypers['d_head']}")
print(f"  num_gnn_layers: {hypers['num_gnn_layers']}")
print(f"  num_attention_layers: {hypers['num_attention_layers']}")
print(f"  featurizer_type: {hypers['featurizer_type']}")

dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=[1, 6, 7, 8, 16],
    targets={"energy": get_energy_target_info("energy", dict(unit="eV"))},
)

model = PET(hypers=hypers, dataset_info=dataset_info).to(dtype)

# A single water molecule: O at the origin, H along x, H along y.
frames = [
    ase.Atoms(
        ["O", "H", "H"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
    )
]
systems = systems_to_torch(frames)

nl_options = NeighborListOptions(cutoff=4.5, full_list=True, strict=True)
systems = [
    get_system_with_neighbor_lists(system, [nl_options]).to(dtype) for system in systems
]


# %%
# .. _standard-outputs:
#
# 1. Standard model outputs
# -------------------------
#
# PET exposes three standard outputs: the backbone features (``"feature"``), the
# last-layer features before the readout (``"mtt::aux::energy_last_layer_features"``),
# and the predicted energy. These are the normal production outputs. Setting
# ``per_atom=False`` aggregates atom features into a single per-structure vector.

outputs = {
    "feature": ModelOutput(per_atom=True),
    "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=True),
    "energy": ModelOutput(per_atom=False),
}
predictions = model(systems, outputs)

print(outputs, predictions)

backbone_block = predictions["feature"].block()
head_block = predictions["mtt::aux::energy_last_layer_features"].block()
energy_block = predictions["energy"].block()

print("backbone features shape:    ", backbone_block.values.shape)
print("last-layer features shape:  ", head_block.values.shape)
print("energy shape:               ", energy_block.values.shape)
print("backbone samples:           ", backbone_block.samples.names)

# %%
#
# The sample labels confirm that the backbone and last layer features are per-atom
# (``["system", "atom"]``), while the energy is per-structure (``["system"]``).
#
# .. note::
#
#    These standard outputs go through PET's full aggregation pipeline: edge
#    contributions are summed over neighbors with cutoff weights applied, and the result
#    is concatenated to the node stream. They are great for feature analysis but do
#    **not** give direct access to the raw intermediate tensors used inside the GNN
#    layers.
#
# The backbone features have feature size 384, which due to the concatenated node
# (d_node=256) and edge (d_pet=128) features. The last layer features (128) have a
# dimension resulting from the concatenated node and edge last layer features (d_head =
# 64 for both).

# %%
# .. _backbone-features:
#
# 2. Unprocessed backbone and head features
# -----------------------------------------
#
# To access the raw per-atom and per-pair tensors as they are used *inside* the
# model — before neighbor-aggregation — use the ``mtt::feature::`` prefix with
# the module path of the corresponding readout layer.
#
# The integer index (``"0"`` below) selects the readout layer. In the default
# ``"feedforward"`` featurizer mode all GNN outputs are combined into a single
# readout, so only index ``0`` exists. In ``"residual"`` mode each GNN layer
# has its own readout and the index selects which one. Here we are uses PET in the
# default feedforward mode.

outputs = {
    "mtt::feature::node_backbone.0": ModelOutput(per_atom=True),
    "mtt::feature::edge_backbone.0": ModelOutput(per_atom=True),
    "mtt::feature::node_heads.energy.0": ModelOutput(per_atom=True),
    "mtt::feature::edge_heads.energy.0": ModelOutput(per_atom=True),
}
predictions = model(systems, outputs)

node_bb = predictions["mtt::feature::node_backbone.0"].block()
edge_bb = predictions["mtt::feature::edge_backbone.0"].block()

print("node_backbone.0  shape:   ", node_bb.values.shape)
print("node_backbone.0  samples: ", node_bb.samples.names)
print()
print("edge_backbone.0  shape:   ", edge_bb.values.shape)
print("edge_backbone.0  samples: ", edge_bb.samples.names)

# %%
#
# This highlights the two kinds of tensors returned by ``mtt::feature::``:
#
# **Node-like** tensors have shape ``(n_atoms, d)`` and samples
# ``["system", "atom"]`` — one row per atom.
#
# **Edge-like** tensors have shape ``(n_edges, d)`` and samples
# ``["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b",
# "cell_shift_c"]`` — one row per directed pair. The cell-shift columns follow
# the standard metatensor neighbor-list convention.


# %%
# .. _featurizer-inputs:
#
# 3. Raw featurizer inputs
# ------------------------
#
# The very first quantities computed in a PET forward pass are the raw geometry
# descriptors fed into the featurizer. These can be captured directly:
#
# * ``edge_distances`` — scalar distance for each directed pair, shape ``(n_edges, 1)``
# * ``edge_vectors`` — displacement vector **r**\⁻ⱼ − **r**\ᵢ for each pair, shape
#   ``(n_edges, 3)``
# * ``element_indices_nodes`` — element-type index for each centre atom, shape
#   ``(n_atoms, 1)``
# * ``element_indices_neighbors`` — element-type index for each neighbor atom, shape
#   ``(n_edges, 1)``

outputs = {
    "mtt::feature::edge_distances": ModelOutput(per_atom=True),
    "mtt::feature::edge_vectors": ModelOutput(per_atom=True),
    "mtt::feature::element_indices_nodes": ModelOutput(per_atom=True),
    "mtt::feature::element_indices_neighbors": ModelOutput(per_atom=True),
}
predictions = model(systems, outputs)

distances = predictions["mtt::feature::edge_distances"].block()
vectors = predictions["mtt::feature::edge_vectors"].block()
node_types = predictions["mtt::feature::element_indices_nodes"].block()
nbr_types = predictions["mtt::feature::element_indices_neighbors"].block()

# %%
#
# Let's print the distances and tie them back to the geometry of the water molecule
# created at the start of this example.  Our molecule has:
#
# * O–H₁: 1.000 Å  (displacement [−1, 0, 0] from H₁ to O)
# * O–H₂: 0.500 Å  (displacement [0, −0.5, 0] from H₂ to O)
# * H₁–H₂: √(1² + 0.5²) ≈ 1.118 Å
#
# With a full neighbor list each pair appears twice (both directions), giving 6
# directed edges in total.

print("Edge distances (Å):")
for i in range(distances.values.shape[0]):
    s = distances.samples[i]
    d = distances.values[i, 0].item()
    print(f"  {int(s['first_atom'])} → {int(s['second_atom'])}:  {d:.4f} Å")

# %%

print("\nDisplacement vectors (Å):")
for i in range(vectors.values.shape[0]):
    s = vectors.samples[i]
    v = vectors.values[i].tolist()
    print(
        f"  {int(s['first_atom'])} → {int(s['second_atom'])}:  "
        f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"
    )

# %%

# Element indices correspond to the position of each atomic number in the
# sorted list of unique elements across the dataset.  Here atomic_types =
# [1, 6, 7, 8, 16], so H (Z=1) → index 0 and O (Z=8) → index 3.
print("\nElement indices per atom (0=H, 3=O):")
print(" ", node_types.values.long().squeeze(-1).tolist())


# %%
# .. _discovering-paths:
#
# 4. Discovering available paths
# ------------------------------
#
# Every sub-module visible in ``print(model)`` can be captured. The helper
# below formats the full list of valid ``mtt::feature::`` keys in one go.


def print_all_module_paths(model):
    _skip = ("additive_models", "scaler", "long_range_featurizer")

    print(f"{'Module path':<65} {'Module type'}")
    print("-" * 95)
    for name, module in model.named_modules():
        if name and not any(name.startswith(p) for p in _skip):
            print(f"  mtt::feature::{name:<50}  {type(module).__name__}")


print_all_module_paths(model)

# %%
#
# .. tip::
#
#    Modules that return a *tuple* of ``(node_features, edge_features)`` —
#    such as :class:`~metatrain.pet.modules.transformer.CartesianTransformer`
#    and :class:`~metatrain.pet.modules.transformer.TransformerLayer` — require
#    a ``_node`` or ``_edge`` suffix to select one element of the tuple:
#
#    .. code-block:: python
#
#       "mtt::feature::gnn_layers.0_node"   # node output of GNN layer 0
#       "mtt::feature::gnn_layers.0_edge"   # edge output of GNN layer 0
#
#    The suffix is only needed when no module with that exact name exists; an
#    :class:`AttributeError` is raised if neither the exact path nor the
#    suffix-stripped variant can be found.


# %%
# .. _gnn-deep-dive:
#
# 5. Deep-dive into a GNN layer
# -----------------------------
#
# We can hook any intermediate sub-module, including layers *inside* the first
# :class:`~metatrain.pet.modules.transformer.CartesianTransformer`. Below we
# capture six tensors from different depths of the first GNN layer to see how
# the representations evolve from raw embeddings through the transformer stack.

outputs = {
    # Initial edge-type embedding (before the GNN)
    "mtt::feature::edge_embedder": ModelOutput(per_atom=True),
    # Edge embedding re-computed inside the first CartesianTransformer
    "mtt::feature::gnn_layers.0.edge_embedder": ModelOutput(per_atom=True),
    # Node and edge output of the first TransformerLayer
    "mtt::feature::gnn_layers.0.trans.layers.0_node": ModelOutput(per_atom=True),
    "mtt::feature::gnn_layers.0.trans.layers.0_edge": ModelOutput(per_atom=True),
    # MLP sub-module inside the same TransformerLayer (node-like)
    "mtt::feature::gnn_layers.0.trans.layers.0.mlp": ModelOutput(per_atom=True),
    # Full node output of the first CartesianTransformer
    "mtt::feature::gnn_layers.0_node": ModelOutput(per_atom=True),
}
predictions = model(systems, outputs)

# %%
#
# Print the shape and nature (node/edge) of each captured tensor.

for key, tmap in predictions.items():
    block = tmap.block()
    kind = "node" if block.samples.names == ["system", "atom"] else "edge"
    print(
        f"{key[len('mtt::feature::') :]:45s}  {str(block.values.shape):20s}  ({kind})"
    )
