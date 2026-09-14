.. _architecture-soap-bpnn:

SOAP-BPNN
=========

This is a Behler-Parrinello neural network :footcite:p:`behler_generalized_2007` with
using features based on the Smooth overlab of atomic positions (SOAP)
:footcite:p:`bartok_representing_2013`. The SOAP features are calculated wit `torch-spex
<https://github.com/lab-cosmo/torch-spex>`_.

Installation
------------
To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install metatrain[soap-bpnn]

This will install the package with the SOAP-BPNN dependencies.


Default Hyperparameters
-----------------------
The default hyperparameters for the SOAP-BPNN model are:

.. literalinclude:: ../../../src/metatrain/soap_bpnn/default-hypers.yaml
   :language: yaml


Tuning Hyperparameters
----------------------
The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. In general, the most important
hyperparameters to tune are (in decreasing order of importance):

- ``cutoff``: This should be set to a value after which most of the interactions between
  atoms is expected to be negligible.
- ``learning_rate``: The learning rate for the neural network. This hyperparameter
  controls how much the weights of the network are updated at each step of the
  optimization. A larger learning rate will lead to faster training, but might cause
  instability and/or divergence.
- ``batch_size``: The number of samples to use in each batch of training. This
  hyperparameter controls the tradeoff between training speed and memory usage. In
  general, larger batch sizes will lead to faster training, but might require more
  memory.
- ``num_hidden_layers``, ``num_neurons_per_layer``, ``max_radial``, ``max_angular``:
  These hyperparameters control the size and depth of the descriptors and the neural
  network. In general, increasing these hyperparameters might lead to better accuracy,
  especially on larger datasets, at the cost of increased training and evaluation time.
- ``radial_scaling`` hyperparameters: These hyperparameters control the radial scaling
  of the SOAP descriptor. In general, the default values should work well, but they
  might need to be adjusted for specific datasets.
- ``layernorm``: Whether to use layer normalization before the neural network. Setting
  this hyperparameter to ``false`` will lead to slower convergence of training, but
  might lead to better generalization outside of the training set distribution.
- ``loss``: This section describes the loss function to be used. See the
  :doc:`dedicated documentation page <../advanced-concepts/loss-functions>` for more
  details.
- ``long_range``: In some systems and datasets, enabling long-range Coulomb interactions
  might be beneficial for the accuracy of the model and/or its physical correctness.
  See below for a breakdown of the long-range section of the model hyperparameters.


All Hyperparameters
-------------------
:param name: ``soap_bpnn``

model
#####

:param heads: The type of head ("linear" or "mlp") to use for each target (e.g.
  ``heads: {"energy": "linear", "mtt::dipole": "mlp"}``). All omitted targets will use a
  MLP (multi-layer perceptron) head. MLP heads consists of one hidden layer with as
  many neurons as the SOAP-BPNN (i.e. ``num_neurons_per_layer`` below).
:param zbl: Whether to use the ZBL short-range repulsion as the baseline for the model
:param long_range: Parameters related to long-range interactions. ``enable``: whether
  to use long-range interactions; ``use_ewald``: whether to use an Ewald calculator
  (faster for smaller systems); ``smearing``: the width of the Gaussian function used
  to approximate the charge distribution in Fourier space; ``kspace_resolution``: the
  spatial resolution of the Fourier-space used for calculating long-range interactions;
  ``interpolation_nodes``: the number of grid points used in spline
  interpolation for the P3M method.

soap
^^^^
:param cutoff: Spherical cutoff (Å) to use for atomic environments
:param max_radial: Number of radial basis function to use
:param max_angular: Number of angular basis function to use also denoted by the  maximum
    degree of spherical harmonics
:param atomic_gaussian_width: Width of the atom-centered gaussian creating the atomic
    density
:param center_atom_weight: Weight of the central atom contribution to the features. If
    1.0 the center atom contribution is weighted the same as any other contribution. If
    0.0 the central atom does not contribute to the features at all.
:param cutoff_function: cutoff function used to smooth the behavior around the cutoff
    radius. The supported cutoff function are

    - ``Step``: Step function, 1 if ``r < cutoff`` and 0 if ``r >= cutoff``. This cutoff
      function takes no additional parameters and can set as in ``.yaml`` file:

      .. code-block:: yaml

        cutoff_function:
          Step:

    - ``ShiftedCosine``: Shifted cosine switching function ``f(r) = 1/2 * (1 + cos(π (r
      - cutoff + width) / width ))``. This cutoff function takes the ``width``` as
      additional parameter and can set as in ``options.yaml`` file as:

      .. code-block:: yaml

        cutoff_function:
          ShiftedCosine:
            width: 1.0

:param radial_scaling: Radial scaling can be used to reduce the importance of neighbor
    atoms further away from the center, usually improving the performance of the model.
    The supported radial scaling functions are

    - ``None``: No radial scaling.

      .. code-block:: yaml

        radial_scaling:
          None:

    - ``Willatt2018`` Use a long-range algebraic decay and smooth behavior at :math:`r
      \rightarrow 0`: as introduced by :footcite:t:`willatt_feature_2018` as ``f(r) =
      rate / (rate + (r / scale) ^ exponent)`` This radial scaling function can be set
      in the ``options.yaml`` file as.

      .. code-block:: yaml

        radial_scaling:
          Willatt2018:
            rate: 1.0
            scale: 2.0
            exponent: 7.0

.. note::

  Currently, we only support a Gaussian type orbitals (GTO) as radial basis functions
  and radial integrals.

bpnn
^^^^
:param layernorm: whether to use layer normalization
:param num_hidden_layers: number of hidden layers
:param num_neurons_per_layer: number of neurons per hidden layer

training
########
The parameters for training are

:param batch_size: batch size
:param num_epochs: number of training epochs
:param learning_rate: learning rate
:param log_interval: number of epochs that elapse between reporting new training results
:param checkpoint_interval: Interval to save a checkpoint to disk.
:param scale_targets: Whether to scale the targets to have unit standard deviation
  across the training set during training.
:param fixed_composition_weights: allows to set fixed isolated atom energies from
  outside. These are per target name and per (integer) atom type. For example,
  ``fixed_composition_weights: {"energy": {1: -396.0, 6: -500.0}, "mtt::U0": {1: 0.0,
  6: 0.0}}`` sets the isolated atom energies for H (``1``) and O (``6``) to different
  values for the two distinct targets.
:param per_atom_targets: specifies whether the model should be trained on a per-atom
  loss. In that case, the logger will also output per-atom metrics for that target. In
  any case, the final summary will be per-structure.
:param log_mae: Also logs MAEs in addition to RMSEs.
:param log_separate_blocks: Whether to log the errors each block of the targets
  separately.
:param loss_weights: specifies the weights to be used in the loss for each target. The
  weights should be a dictionary of floats, one for each target. All missing targets
  are assigned a weight of 1.0.
:param best_model_metric: specifies the validation set metric to use to select the best
  model, i.e. the model that will be saved as ``model.ckpt`` and ``model.pt`` both in
  the current directory and in the checkpoint directory. The default is ``rmse_prod``,
  i.e., the product of the RMSEs for each target. Other options are ``mae_prod`` and
  ``loss``.


References
----------
.. footbibliography::
