.. _architecture-alchemical-model:

Alchemical Model
================

.. warning::

  This is an **experimental model**.  You should not use it for anything important.

This is an implementation of Alchemical Model: a Behler-Parrinello neural network
:footcite:p:`behler_generalized_2007` with Smooth overlab of atomic positions (SOAP)
features :footcite:p:`bartok_representing_2013` and Alchemical Compression of the
composition space :footcite:p:`willatt_feature_2018, lopanitsyna_modeling_2023,
mazitov_surface_2024`. This model is extremely useful for simulating systems with
large amount of chemical elements.


Installation
------------
To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install .[alchemical-model]

This will install the package with the Alchemical Model dependencies.


Default Hyperparameters
-----------------------
The default hyperparameters for the Alchemical Model model are:

.. literalinclude:: ../../../src/metatrain/experimental/alchemical_model/default-hypers.yaml
   :language: yaml


Tuning Hyperparameters
----------------------
The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. In general, the most important
hyperparameters to tune are (in decreasing order of importance):

- ``cutoff``: This should be set to a value after which most of the
  interactions between atoms is expected to be negligible.
- ``num_pseudo_species``: This number determines the number of pseudo species
  to use in the Alchemical Compression of the composition space. This value should
  be adjusted based on the prior knowledge of the size of original chemical space
  size.
- ``learning_rate``: The learning rate for the neural network. This hyperparameter
  controls how much the weights of the network are updated at each step of the
  optimization. A larger learning rate will lead to faster training, but might cause
  instability and/or divergence.
- ``batch_size``: The number of samples to use in each batch of training. This
  hyperparameter controls the tradeoff between training speed and memory usage. In
  general, larger batch sizes will lead to faster training, but might require more
  memory.
- ``hidden_sizes``:
  This hyperparameter controls the size and depth of the descriptors and the neural
  network. In general, increasing this might lead to better accuracy,
  especially on larger datasets, at the cost of increased training and evaluation time.
- ``loss_weights``: This controls the weighting of different contributions to the loss
  (e.g., energy, forces, virial, etc.). The default values work well for most datasets,
  but they might need to be adjusted. For example, to set a weight of 1.0 for the energy
  and 0.1 for the forces, you can set the following in the ``options.yaml`` file:
  ``loss_weights: {"energy": 1.0, "forces": 0.1}``.


Architecture Hyperparameters
----------------------------
:param name: ``experimental.alchemical_model``

model
#####
soap
^^^^
:param num_pseudo_species: Number of pseudo species to use in the Alchemical Compression
    of the composition space.
:param cutoff_radius: Spherical cutoff (Å) to use for atomic environments.
:param basis_cutoff: The maximal eigenvalue of the Laplacian Eigenstates (LE) basis
    functions used as radial basis :footcite:p:`bigi_smooth_2022`. This controls how
    large the radial-angular basis is.
:param radial_basis_type: A type of the LE basis functions used as radial basis. The
    supported radial basis functions are

    - ``LE``: Original Laplacian Eigenstates raidal basis. These radial basis functions
      can be set in the ``.yaml`` file as:

      .. code-block:: yaml

        radial_basis_type: "le"

    - ``Physical``: Physically-motivated basis functions. These radial basis functions
      can be set in

      .. code-block:: yaml

        radial_basis_type: "physical"

:param basis_scale: Scaling parameter of the radial basis functions, representing the
    characteristic width (in Å) of the basis functions.
:param trainable_basis: If :py:obj:`True`, the radial basis functions will be
    accompanied by the trainable multi-layer perceptron (MLP). If :py:obj:`False`, the
    radial basis functions will be fixed.
:param normalize: Whether to use normalizations such as LayerNorm in the model.
:param contract_center_species: If ``True``, the Alchemcial Compression will be applied
    on center species as well. If ``False``, the Alchemical Compression will be applied
    only on the neighbor species.


bpnn
^^^^
:param hidden_sizes: number of neurons in each hidden layer
:param output_size: number of neurons in the output layer

training
########
The hyperparameters for training are

:param batch_size: batch size
:param num_epochs: number of training epochs
:param learning_rate: learning rate
:param log_interval: number of epochs that elapse between reporting new training results
:param checkpoint_interval: Interval to save a checkpoint to disk.
:param per_atom_targets: Specifies whether the model should be trained on a per-atom
    loss. In that case, the logger will also output per-atom metrics for that target. In
    any case, the final summary will be per-structure.

References
----------
.. footbibliography::


