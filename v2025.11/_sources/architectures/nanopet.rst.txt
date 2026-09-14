.. _architecture-nanopet:

NanoPET (deprecated)
======================

.. warning::

  This is an **deprecated model**. You should not use it for anything important, and
  support for it will be removed in future versions of metatrain. Please use the
  :ref:`PET model <architecture-pet>` instead.

Installation
------------

To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install metatrain[nanopet]

This will install the package with the nanoPET dependencies.


Default Hyperparameters
-----------------------

The default hyperparameters for the nanoPET model are:

.. literalinclude:: ../../../src/metatrain/deprecated/nanopet/default-hypers.yaml
   :language: yaml


Tuning Hyperparameters
----------------------

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

:param name: ``deprecated.nanopet``

model
#####

The model-related hyperparameters are

:param cutoff: Spherical cutoff to use for atomic environments
:param cutoff_width: Width of the shifted cosine cutoff function
:param d_pet: Width of the neural network
:param num_heads: Number of attention heads
:param num_attention_layers: Number of attention layers in each GNN layer
:param num_gnn_layers: Number of GNN layers
:param heads: The type of head ("linear" or "mlp") to use for each target (e.g.
  ``heads: {"energy": "linear", "mtt::dipole": "mlp"}``). All omitted targets will use a
  MLP (multi-layer perceptron) head. MLP heads consist of two hidden layers with
  dimensionality ``d_pet``.
:param zbl: Whether to use the ZBL short-range repulsion as the baseline for the model
:param long_range: Parameters related to long-range interactions. ``enable``: whether
  to use long-range interactions; ``use_ewald``: whether to use an Ewald calculator
  (faster for smaller systems); ``smearing``: the width of the Gaussian function used
  to approximate the charge distribution in Fourier space; ``kspace_resolution``: the
  spatial resolution of the Fourier-space used for calculating long-range interactions;
  ``interpolation_nodes``: the number of grid points used in spline
  interpolation for the P3M method.

training
########

The hyperparameters for training are

:param distributed: Whether to use distributed training
:param distributed_port: Port to use for distributed training
:param batch_size: Batch size for training
:param num_epochs: Number of epochs to train for
:param learning_rate: Learning rate for the optimizer
:param scheduler_patience: Patience for the learning rate scheduler
:param scheduler_factor: Factor to reduce the learning rate by
:param log_interval: Interval at which to log training metrics
:param checkpoint_interval: Interval at which to save model checkpoints
:param scale_targets: Whether to scale the targets to have unit standard deviation
  across the training set during training.
:param fixed_composition_weights: Weights for fixed atomic contributions to scalar
  targets
:param per_structure_targets: Targets to calculate per-structure losses for
:param log_mae: Also logs MAEs in addition to RMSEs.
:param log_separate_blocks: Whether to log the errors each block of the targets
  separately.
:param loss: The loss function to use, with the subfields described in the previous
  section
:param best_model_metric: specifies the validation set metric to use to select the best
  model, i.e. the model that will be saved as ``model.ckpt`` and ``model.pt`` both in
  the current directory and in the checkpoint directory. The default is ``rmse_prod``,
  i.e., the product of the RMSEs for each target. Other options are ``mae_prod`` and
  ``loss``.
:param num_workers: Number of workers for data loading. If not provided, it is set
  automatically.

References
----------

.. footbibliography::
