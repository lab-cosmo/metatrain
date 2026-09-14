.. _loss-functions:

Loss functions
==============

``metatrain`` supports a variety of loss functions, which can be configured in the ``loss`` subsection of the ``training`` section for each ``architecture`` in the options file.
The loss functions are designed to be flexible and can be tailored to the specific needs of the dataset and of the predicted targets.


Loss function configurations in the ``options.yaml`` file
---------------------------------------------------------

A common use case is the training of machine-learning interatomic potentials (MLIPs), where the training targets include energies, forces, and stress/virial.

The loss terms for energy, forces, and stress can be specified as:

.. code-block:: yaml

  loss:
    energy:
      type: mse
      weight: 1.0
      reduction: mean
      forces:
        type: mse
        weight: 1.0
        reduction: mean
      stress:
        type: mse
        weight: 1.0
        reduction: mean

Here, ``forces`` and ``stress`` refer to the gradients of the ``energy`` target with respect to atomic positions and strain, respectively, assuming these gradients have been requested in the training set configuration.

Another common scenario is when only the loss function type needs to be specified, while default values are acceptable for the other parameters. In that case, the configuration can be further simplified to:

.. code-block:: yaml

  loss:
    energy:
      type: mse
      forces: mae
      stress: huber

where, for example, different types of losses are requested for different targets.
This is equivalent to the more detailed configuration:

.. code-block:: yaml

  loss:
    energy:
      type: mse
      weight: 1.0
      reduction: mean
      forces:
        type: mae
        weight: 1.0
        reduction: mean
      stress:
        type: huber
        weight: 1.0
        reduction: mean
        delta: 1.0

When all targets and their gradients should use the same loss function with equal weights and reductions, it is also possible to use the global shorthand

.. code-block:: yaml

  loss: mse

which sets the loss type to mean squared error (MSE) for all targets and, if present, for all their gradients.

This example assumes that the training set contains a target named ``energy``, and that gradients with respect to both atomic positions (forces) and strain (stress/virial) have been requested.
If the energy target has a custom name (e.g., ``mtt::etot``), the loss configuration should use that name instead:

.. code-block:: yaml

  loss:
    mtt::etot:
      type: mse
      weight: 1.0
      reduction: mean
      forces:
        type: mse
        weight: 1.0
        reduction: mean
      stress:
        type: mse
        weight: 1.0
        reduction: mean
  ...
  training_set:
    systems:
    ...
    targets:
      mtt::etot:
        quantity: energy
        forces: true  # or some other allowed configuration
        stress: true  # or some other allowed configuration
    ...

Mind that, in the case the target name is not ``energy``, the key ``quantity: energy`` in the target definition must be present to specify that this target corresponds to energies.
This allows ``metatrain`` to associate the correct gradients (forces and stress/virial) when requested.
Both the explicit MLIP configuration (with separate ``energy``, ``forces``, and ``stress`` entries) and the global shorthand ``loss: mse`` are thus mapped to the same internal representation, where loss terms are specified explicitly per target and per gradient.


Internal configuration format
-----------------------------

The internal configuration used by ``metatrain`` during training is a more detailed version of the examples shown above, where each target has its own loss configuration and an optional ``gradients`` subsection.

The example above where the loss function is MSE for energy, forces, and stress is thus represented internally as:

.. code-block:: yaml

  loss:
    energy:
      type: mse
      weight: 1.0
      reduction: mean
      gradients:
        positions:
          type: mse
          weight: 1.0
          reduction: mean
        strain:
          type: mse
          weight: 1.0
          reduction: mean

This internal format is also available to users in the options file. It can be used to handle general targets and their "non-standard" gradients, those that are not simply forces or stress (for example, custom derivatives with respect to user-defined quantities).

Generally, each loss-function term accepts the following parameters:

:param type: This controls the type of loss to be used. The default value is ``mse``, and other standard options are ``mae`` and ``huber``, which implement the equivalent PyTorch loss functions `MSELoss <https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_, `L1Loss <https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html>`_, and `HuberLoss <https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_, respectively.
   There are also "masked" versions of these losses, which are useful when using padded targets with values that should be masked before computing the loss. The masked losses are named ``masked_mse``, ``masked_mae``, and ``masked_huber``.
:param ``weight``: This controls the weighting of different contributions to the loss (e.g., energy, forces, virial, etc.). The default value of 1.0 for all targets works well for most datasets, but can be adjusted if required.
:param ``reduction``: This controls how the overall loss is computed across batches. The default for this is to use the ``mean`` of the batch losses. The ``sum`` function is also supported.

Some losses, like ``huber``, require additional parameters to be specified:

:param delta: This parameter is specific to the Huber loss functions (``huber`` and ``masked_huber``) and defines the threshold at which the loss function transitions from quadratic to linear behavior. The default value is 1.0.


Masked loss functions
---------------------

Masked loss functions are particularly useful when dealing with datasets that contain padded targets. In such cases, the loss function can be configured to ignore the padded values during the loss computation.
This is done by using the ``masked_`` prefix in the loss type. For example, if the target contains padded values, you can use ``masked_mse`` or ``masked_mae`` to ensure that the loss is computed only on the valid (non-padded) values.
The values of the masks must be passed as ``extra_data`` in the training set, and the loss function will automatically apply the mask to the target values. An example configuration for a masked loss is as follows:

 .. code-block:: yaml

  loss:
    energy:
      type: masked_mse
      weight: 1.0
      reduction: sum
      forces:
        type: masked_mae
        weight: 0.1
        reduction: sum
  ...

  training_set:
    systems:
    ...
    targets:
      mtt::my_target:
        ...
    ...
    extra_data:
      mtt::my_target_mask:
        read_from: my_target_mask.mts


.. _dos-loss:

DOS Loss Function
^^^^^^^^^^^^^^^^^

The masked DOS loss function is a specialized loss designed for training on the electronic density of states (DOS), typically represented on an energy grid. Structures in a dataset can (and usually do) have eigenvalues spanning different energy ranges, and DOS calculations do not share a common absolute energy reference.
To handle this, the loss uses a user-specified number of extra predicted targets to dynamically shift the energy grid for each structure, aligning the predicted DOS with the reference DOS before computing the loss.

After this alignment step, the loss function consists of three components:

- an integrated loss on the masked DOS values

.. code-block:: python

    masked_DOS_loss = torch.trapezoid((aligned_predictions - targets)**2 * mask, x_axis = energy_grid)

- an integrated loss on the gradient of the *unmasked* DOS values, to ensure that values outside the masked region are also learned smoothly

.. code-block:: python

    unmasked_gradient_loss = torch.trapezoid(aligned_predictions_gradient**2 * (~mask), x_axis = energy_grid)

- an integrated loss on the cumulative DOS values in the masked region

.. code-block:: python

    cumulative_aligned_predictions = torch.cumulative_trapezoid(aligned_predictions, x = energy_grid)
    cumulative_targets = torch.cumulative_trapezoid(targets, x = energy_grid)
    masked_cumulative_DOS_loss = torch.trapezoid((cumulative_aligned_predictions - cumulative_targets)**2 * mask, x_axis = energy_grid[1:])

Each component can be weighted independently to tailor the loss function to specific training needs.

.. code-block:: python

    loss = (masked_DOS_loss +
            grad_weight * unmasked_gradient_loss +
            int_weight * masked_cumulative_DOS_loss)

To use this loss function, you can refer to this code snippet for the ``loss`` section in your YAML configuration file:

.. code-block:: yaml

    loss:
      mtt::dos:
        type: "masked_dos"
        grad_weight: 1e-4
        int_weight: 2.0
        extra_targets: 200
        reduction: "mean"

:param name: key for the dos in the prediction/target dictionary. (mtt::dos in this case)
:param grad_weight: Multiplier for the gradient of the unmasked DOS component.
:param int_weight: Multiplier for the cumulative DOS component.
:param extra_targets: Number of extra targets predicted by the model.
:param reduction: reduction mode for torch loss. Options are "mean", "sum", or "none".

The values used in the above example are the ones used for PETMADDOS training and can be a reasonable starting point for other applications.
