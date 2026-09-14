.. _architecture-flashmd:

FlashMD
=======

FlashMD is a method for the direct prediction of positions and momenta in a molecular
dynamics simulation, presented in :footcite:p:`bigi_flashmd_2025`. When compared to
traditional molecular dynamics methods, it predicts the positions and momenta of atoms
after a long time interval, allowing the use of much larger time steps. Therefore, it
achieves a significant speedup (10-30x) compared to molecular dynamics using MLIPs.
The FlashMD architecture implemented in metatrain is based on The
:ref:`PET architecture <architecture-pet>`.

Installation
------------

To install FlashMD and its dependencies, you can run the following from the root of the
repository:

.. code-block:: bash

    pip install metatrain[flashmd]

Default Hyperparameters
-----------------------

The default hyperparameters for the FlashMD architecture are:

.. literalinclude:: ../../../src/metatrain/experimental/flashmd/default-hypers.yaml
   :language: yaml

Hyperparameters
---------------

In order to choose and tune hyperparameters for FlashMD, you can refer to the
:ref:`PET architecture documentation <architecture-pet>`, as most of the hyperparameters
are shared between the two architectures. Here, we will only discuss the hyperparameters
that are specific to FlashMD.

model
#####

- ``predict_momenta_as_difference``: This parameter controls whether the model will
  predict future momenta directly or as a difference between the future and the current
  momenta. Setting it to true will help when predicting relatively small timesteps
  (when compared to the momentum autocorrelation time), while setting it to false is
  beneficial when predicting large timesteps. Defaults to ``false``.

training
########

- ``timestep``: The time interval (in fs) between the current and the future positions
  and momenta that the model must predict. This option is not used in the training, but
  it is registered in the model and it will be used to validate that the timestep used
  during inference in MD engines is the same as the one used during training. This
  hyperparameter must be provided by the user.
- ``masses``: A dictionary mapping atomic species to their masses (in atomic mass
  units). Indeed, it should be noted that FlashMD models, as implemented in metatrain,
  are not transferable across different isotopes. The masses are not used during
  training, but they are registered in the model and they will be used during inference
  to validate that the masses used in MD engines are the same as the ones used during
  training. If not provided, masses from the ``ase.data`` module will be used. These
  correspond to masses averaged over the natural isotopic abundance of each element.

References
----------

.. footbibliography::
