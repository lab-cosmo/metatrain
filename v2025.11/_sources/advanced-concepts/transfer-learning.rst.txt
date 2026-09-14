.. _transfer-learning:

Transfer Learning (experimental)
====================================

.. warning::

  This section of the documentation is only relevant for PET model so far.

.. warning::

  Features described in this section are experimental and not yet
  extensively tested. Please use them at your own risk and report any
  issues you encounter to the developers. The transfer learned models
  cannot be used in MD engines such as ASE or LAMMPS yet.


This section describes the process of transfer learning, which is a
common technique used in machine learning, where a model is pre-trained on
the dataset with one level of theory and/or one set of properties and then
fine-tuned on a different dataset with a different level of theory and/or
different set of properties. This approach to use the learned representations
from the pre-trained model and adapt them to the targets, which can be
expensive to compute and/or not available in the pre-trained dataset.

In the following sections we assume that the pre-trained model is trained on the
conventional DFT dataset with energies, forces and stresses, which are provided
as ``energy`` targets (and its derivatives) in the ``options.yaml`` file.


Fitting to a new level of theory
--------------------------------

Training on a new level of theory is a common use case for transfer learning. It
requires using a pre-trained model checkpoint with the ``mtt train`` command and setting the
new targets corresponding to the new level of theory in the ``options.yaml`` file. Let's
assume that the training is done on the dataset computed with the hybrid DFT functional
(e.g. PBE0) stored in the ``new_train_dataset.xyz`` file, where the corresponsing
energies and forces are written in the ``energy`` and ``forces`` key of the ``info`` dictionary
of the ``ase.Atoms`` object. Then, the ``options.yaml`` file should look like this:

.. code-block:: yaml

  architecture:
    name: pet
    training:
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt

  training_set:
    systems:
      read_from: dataset.xyz
      reader: ase
      length_unit: angstrom
    targets:
      mtt::energy_pbe0: # name of the new target
        key: energy # key of the target in the atoms.info dictionary
        unit: eV # unit of the target value
        forces:
          key: forces

  test_set: 0.1
  validation_set: 0.1

The validation and test sets can be set in the same way. The training
process will then create a new composition model and new heads for the
target ``mtt::energy_pbe0``. The rest of the model weights will be
initialized from the pre-trained model checkpoint.

Fitting to a new set of properties
----------------------------------

Training on a new set of properties is another common use case for
transfer learning. It can be done in a similar way as training on a new
level of theory. The only difference is that the new targets need to be
properly set in the ``options.yaml`` file. More information about fitting the
generic targets can be found in the :ref:`Fitting generic targets <fitting-generic-targets>`
section of the documentation.


