.. _architecture-soap-bpnn:

SOAP-BPNN
=========

This is a Behler-Parrinello type neural network :footcite:p:`behler_generalized_2007`,
which, instead of their original atom-centered symmetry functions, we use the Smooth
overlap of atomic positions (SOAP) :footcite:p:`bartok_representing_2013` as the atomic
descriptors, computed with `torch-spex <https://github.com/lab-cosmo/torch-spex>`_.

Installation
------------
To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install metatrain[soap-bpnn]

This will install the package with all of the SOAP-BPNN dependencies.


Default Hyperparameters
-----------------------
The full set of default hyperparameters for the SOAP-BPNN model are as follows:

.. literalinclude:: ../../../src/metatrain/soap_bpnn/default-hypers.yaml
   :language: yaml


You will note that there are mainly two sets of hyperparameters: ``model``, and
``training``. The ``training`` hyperparameters are rather general and consistent
across most of the architectures. the ``model`` hypers also contain ''general''
ones: ``add_lambda_basis``, ``heads``, ``zbl``, and ``long_range``. The rest of
the hypers are specific to SOAP-BPNN. While the above ``model`` hyperparameter
set would work OK in most cases, they may not be optimal for your specific case.
We explain below the model-specific hypers for SOAP-BPNN.

``soap``
########
- ``cutoff``: This determines the cutoff routine of the atomic environment, and has
  two internal hypers: ``radius`` and ``width``. ``radius`` should be set to a value
  after which most of interatomic is expected to be negligible. Note that the values
  should be defined in the position units of your dataset. The radial cutoff of
  atomic environments is performed smoothly, over another distance defined under
  ``width``.
- ``max_angular`` and ``max_radial``: These parameters define the maximum angular and
  radial channels of the spherical harmonics in computing the SOAP descriptors.

``bpnn``
########
- ``num_hidden_layers``: and :param num_neurons_per_layer: These hyperparameters control
  the size and depth of the neural network. Increasing these generally lead to better
  accuracy from the increased descriptivity, but comes at the cost of increased
  training and evaluation time.
- ``layernorm``: Whether to use layer normalization before the neural network. Setting
  this hyperparameter to ``false`` will lead to slower convergence of training, but
  might lead to better generalization outside of the training set distribution.
- ``loss``: This section describes the loss function to be used. See the
  :doc:`dedicated documentation page <../advanced-concepts/loss-functions>` for more
  details.

In addition to these model-specific hypers, we re-highlight that the following additive
models (``zbl`` and ``long_range``) may be needed to achieve better description at the
close-contact, repulsive regime, or to describe important long-range effects not
captured by the short-range SOAP-BPNN model.


All Hyperparameters
-------------------
For completeness, rest of the hyperparameters, which are non-specific to SOAP-BPNN, are
briefly explained below.

``model``
#########
- ``add_lambda_basis``: This boolean parameter controls whether or not to add a
  spherical expansion term of the same angular order as the targets, when they are tensorial.
- ``heads``: The type of head ("linear" or "mlp") to use for each target (e.g.
  ``heads: {"energy": "linear", "mtt::dipole": "mlp"}``). All omitted targets will use a
  MLP (multi-layer perceptron) head. MLP heads consists of one hidden layer with as
  many neurons as the SOAP-BPNN (i.e. ``num_neurons_per_layer`` below).
- ``zbl``: Whether to use the ZBL short-range repulsion as the baseline for the model
- ``long_range``: Parameters related to long-range interactions. ``enable``: whether
  to use long-range interactions; ``use_ewald``: whether to use an Ewald calculator
  (faster for smaller systems); ``smearing``: the width of the Gaussian function used
  to approximate the charge distribution in Fourier space; ``kspace_resolution``: the
  spatial resolution of the Fourier-space used for calculating long-range interactions;
  ``interpolation_nodes``: the number of grid points used in spline
  interpolation for the P3M method.

``training``
############
- ``distributed``: this boolean determines whether or not to distribute the learning.
- ``distributed_port``: this integer defines the port to be used in the distributed
  learning exercise.
- ``batch_size``: this integer defines to which number of structures the workflow
  divides up the training set into batches during model training.
- ``num_epochs``: this integer defines the number of epochs to perform in training.
- ``learning_rate``: this float defines the initial learning rate of the scheduler.
- ``warmup_fraction``: this float defines the fraction of the total number of epochs
  to be used for linearly increasing the learning rate from zero to the initial
  learning rate defined by ``learning_rate`` at the start of training.
- ``log_interval``: this integer defines the epoch interval of metric logging.
- ``checkpoint_interval``: this integer defines the epoch interval of checkpoint
  saving.
- ``scale_targets``: this boolean determines whether or not to scale the targets
  with the internal scalers before the targets are exposed to the models for learning.
- ``fixed_composition_weights``: this nested dictionary allows one to set fixed
  composition values in the composition model being used as a baseline for the model.
  These are per target name and per (integer) atom type. For example,
  ``fixed_composition_weights: {"energy": {1: -396.0, 6: -500.0}, "mtt::U0": {1: 0.0,
  6: 0.0}}`` sets the isolated atom energies for H (``1``) and O (``6``) to different
  values for the two distinct targets.
- ``per_structure_targets``: this list of strings defines the global targets for
  which the target value should _not_ be normalized w.r.t. number of atoms.
- ``log_mae``: this boolean controls the additional logging of MAEs along with RMSEs
- ``log_separate_blocks``: this boolean logs the metrics for the separate blocks of
  the target tensor map.
- ``best_model_metric``: specifies the validation set metric to use to select the best
  model, i.e. the model that will be saved as ``model.ckpt`` and ``model.pt`` both in
  the current directory and in the checkpoint directory. The default is ``rmse_prod``,
  i.e., the product of the RMSEs for each target. Other options are ``mae_prod`` and
  ``loss``.
- ``num_workers``: Number of workers for data loading. If not provided, it is set
  automatically.
- ``loss``: this string parameter defines the type of loss to be used. It only takes
  one of the losses implemented within metatrain as valid parameters.


References
----------
.. footbibliography::
