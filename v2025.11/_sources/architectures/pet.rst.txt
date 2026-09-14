.. _architecture-pet:

PET
===

PET is a cleaner, more user-friendly reimplementation of the original
PET model :footcite:p:`pozdnyakov_smooth_2023`. It is designed for better
modularity and maintainability, while preseving compatibility with the original
PET implementation in ``metatrain``. It also adds new features like long-range
features, better fine-tuning implementation, a possibility to train on
arbitrarty targets, and a faster inference due to the ``fast attention``.

Installation
------------

PET model is included in the ``metatrain`` package and doesn't require any
additional installation steps. To install ``metatrain`` run:

.. code-block:: bash

    pip install metatrain

This will install the PET model along with the ``metatrain`` package.

Default Hyperparameters
-----------------------

The default hyperparameters for the PET model are:

.. literalinclude:: ../../../src/metatrain/pet/default-hypers.yaml
   :language: yaml

Tuning Hyperparameters
----------------------

PET offers a number of tuning knobs for flexibility across datasets:

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. In general, the most important
hyperparameters to tune are (in decreasing order of importance):

- ``cutoff``: This should be set to a value after which most of the interactions between
  atoms is expected to be negligible. A lower cutoff will lead to faster models.
- ``learning_rate``: The learning rate for the neural network. This hyperparameter
  controls how much the weights of the network are updated at each step of the
  optimization. A larger learning rate will lead to faster training, but might cause
  instability and/or divergence.
- ``batch_size``: The number of samples to use in each batch of training. This
  hyperparameter controls the tradeoff between training speed and memory usage. In
  general, larger batch sizes will lead to faster training, but might require more
  memory.
- ``d_pet``: This hyperparameters controls width of the neural network. In general,
  increasing it might lead to better accuracy, especially on larger datasets, at the
  cost of increased training and evaluation time.
- ``d_node``: The dimension of the node features. Increasing this hyperparameter
  might lead to better accuracy, with a relatively small increase in inference time.
- ``num_gnn_layers``: The number of graph neural network layers. In general, decreasing
  this hyperparameter to 1 will lead to much faster models, at the expense of accuracy.
  Increasing it may or may not lead to better accuracy, depending on the dataset, at the
  cost of increased training and evaluation time.
- ``num_attention_layers``: The number of attention layers in each layer of the graph
  neural network. Depending on the dataset, increasing this hyperparameter might lead to
  better accuracy, at the cost of increased training and evaluation time.
- ``loss``: This section describes the loss function to be used. See the
  :doc:`dedicated documentation page <../advanced-concepts/loss-functions>` for more
  details.
- ``long_range``: In some systems and datasets, enabling long-range Coulomb interactions
  might be beneficial for the accuracy of the model and/or its physical correctness.
  See below for a breakdown of the long-range section of the model hyperparameters.

All Hyperparameters
-------------------

:param name: ``pet``

model
#####

:param cutoff: Cutoff radius for neighbor search
:param cutoff_width: Width of the smoothing function at the cutoff
:param d_pet: Dimension of the edge features
:param d_head: Dimension of the attention heads
:param d_node: Dimension of the node features
:param d_feedforward: Dimension of the feedforward network in the attention layer
:param num_heads: Attention heads per attention layer
:param num_attention_layers: Number of attention layers per GNN layer
:param num_gnn_layers: Number of GNN layers
:param normalization: Layer normalization type. Currently available options are
  ``RMSNorm`` or ``LayerNorm``.
:param activation: Activation function. Currently available options are ``SiLU``,
  and ``SwiGLU``.
:param transformer_type: The order in which the layer normalization and attention
  are applied in a transformer block. Available options are ``PreLN``
  (normalization before attention) and ``PostLN`` (normalization after attention).
:param featurizer_type: Implementation of the featurizer of the model to use. Available
  options are ``residual`` (the original featurizer from the PET paper, that uses
  residual connections at each GNN layer for readout) and ``feedforward`` (a modern
  version that uses the last representation after all GNN iterations for readout).
  Additionally, the feedforward version uses bidirectional features flow during the
  message passing iterations, that favors features flowing from atom ``i`` to atom
  ``j`` to be not equal to the features flowing from atom ``j`` to atom ``i``.
:param zbl: Use ZBL potential for short-range repulsion
:param long_range: Long-range Coulomb interactions parameters:
  - ``enable``: Toggle for enabling long-range interactions
  - ``use_ewald``: Use Ewald summation. If False, P3M is used
  - ``smearing``: Smearing width in Fourier space
  - ``kspace_resolution``: Resolution of the reciprocal space grid
  - ``interpolation_nodes``: Number of grid points for interpolation (for PME only)

training
########

:param distributed: Whether to use distributed training
:param distributed_port: Port for DDP communication
:param batch_size: Training batch size
:param num_epochs: Number of epochs
:param warmup_fraction: Fraction of training steps used for learning rate warmup
:param learning_rate: Learning rate
:param log_interval: Interval to log metrics
:param checkpoint_interval: Interval to save checkpoints
:param scale_targets: Normalize targets to unit std during training
:param fixed_composition_weights: Weights for atomic contributions
:param per_structure_targets: Targets to calculate per-structure losses
:param log_mae: Log MAE alongside RMSE
:param log_separate_blocks: Log per-block error
:param grad_clip_norm: Maximum gradient norm value, by default inf (no clipping)
:param loss: Loss configuration (see above)
:param best_model_metric: Metric used to select best checkpoint (e.g., ``rmse_prod``)
:param num_workers: Number of workers for data loading. If not provided, it is set
  automatically.

References
----------

.. footbibliography::
