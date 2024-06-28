.. _architecture-soap-bpnn:

SOAP-BPNN
=========

.. warning::

  This is an **experimental model**.  You should not use it for anything important.

This is a Behler-Parrinello neural network :footcite:p:`behler_generalized_2007` with
using features based on the Smooth overlab of atomic positions (SOAP)
:footcite:p:`bartok_representing_2013`. The SOAP features are calculated wit `rascaline
<https://luthaf.fr/rascaline/latest/index.html>`_.

Installation
------------
To install the package, you can run the following command in the root
directory of the repository:

.. code-block:: bash

    pip install .[soap-bpnn]

This will install the package with the SOAP-BPNN dependencies.


Default Hyperparameters
-----------------------
The default hyperparameters for the SOAP-BPNN model are:

.. literalinclude:: ../../../src/metatrain/experimental/soap_bpnn/default-hypers.yaml
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
- ``loss_weights``: This controls the weighting of different contributions to the loss
  (e.g., energy, forces, virial, etc.). The default values work well for most datasets,
  but they might need to be adjusted. For example, to set a weight of 1.0 for the energy
  and 0.1 for the forces, you can set the following in the ``options.yaml`` file:
  ``loss_weights: {"energy": 1.0, "forces": 0.1}``.
- ``layernorm``: Whether to use layer normalization before the neural network. Setting
  this hyperparameter to ``false`` will lead to slower convergence of training, but
  might lead to better generalization outside of the training set distribution.


All Hyperparameters
-------------------
:param name: ``experimental.soap_bpnn``

model
#####
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
:param per_atom_targets: Specifies whether the model should be trained on a per-atom
    loss. In that case, the logger will also output per-atom metrics for that target. In
    any case, the final summary will be per-structure.
:param loss_weights: Specifies the weights to be used in the loss for each target. The
    weights should be a dictionary of floats, one for each target. All missing targets
    are assigned a weight of 1.0.


References
----------
.. footbibliography::
