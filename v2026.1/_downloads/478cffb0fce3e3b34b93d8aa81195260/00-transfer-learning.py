"""
.. _transfer-learning:

Transfer Learning (experimental)
====================================

.. warning::

  This section of the documentation is only relevant for PET model so far.

.. warning::

  Features described in this section are experimental and not yet
  extensively tested. Please use them at your own risk and report any
  issues you encounter to the developers. The transfer learned models
  cannot be directly used in MD engines such as ASE or LAMMPS yet.
  If you still want to use them, please follow the instructions below.


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
requires using a pre-trained model checkpoint with the ``mtt train`` command and setting
the new targets corresponding to the new level of theory in the ``options.yaml`` file.
Let's assume that the training is done on the dataset computed with the hybrid DFT
functional (e.g. PBE0) stored in the ``new_train_dataset.xyz`` file, where the
corresponsing energies and forces are written in the ``energy`` and ``forces`` key of
the ``info`` dictionary of the ``ase.Atoms`` object. Then, the ``options.yaml`` file
should look like this:

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

Inheriting weights from existing heads
--------------------------------------

In some cases, the new targets might be similar to the existing targets
in the pre-trained model. For example, if the pre-trained model is trained
on energies and forces computed with the PBE functional, and the new targets
are energies and forces coming from the PBE0 calculations, it might be beneficial
to initialize the new PBE0 heads and last layers with the weights of the PBE
heads and last layers. This can be done by specifying the ``inherit_heads``
parameter in the ``options.yaml`` file:

.. code-block:: yaml

  architecture:
    training:
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt
        inherit_heads:
          mtt::energy_pbe0: energy # inherit weights from the "energy" head

The ``inherit_heads`` parameter is a dictionary mapping the new trainable
targets specified in the ``training_set/targets`` section to the existing
targets in the pre-trained model. The weights of the corresponding heads and
last layers will be copied from the source heads to the destination heads
instead of random initialization. These weights are still trainable and
will be adapted to the new dataset during the training process.


Using the transfer-learned model in simulation engines
-------------------------------------------------------

The default target name expected by the ``metatomic`` package in order
to use the model in ASE and LAMMPS calculations is ``energy``. If the new
transfer-learned target has a different name, e.g. ``mtt::energy_pbe0``,
the ``metatomic`` model interface will still try to access the ``energy``
target name while evaluating energies and forces. Currently, there is no
automatic way to set the target name in the ``metatomic`` model interface
(this feature is under development). Therefore, in order to use the
transfer-learned model in simulation engines, the new target needs to be renamed
to ``energy`` in the trained model checkpoint ``.ckpt`` file. This can be done
using a relatively simple python script:

.. code-block:: python

  def set_output_head(checkpoint, head_name):
      for state_dict_name in ["model_state_dict", "best_model_state_dict"]:
          state_dict = checkpoint.get(state_dict_name)
          if state_dict is not None:
              new_state_dict = {}
              for key, value in state_dict.items():
                  if ".energy." in key:
                      continue
                  if "scaler.scales" in key:
                      value = value[:1]
                  if head_name in key:
                      new_key = key.replace(head_name, "energy")
                  else:
                      new_key = key
                  new_state_dict[new_key] = value
              checkpoint[state_dict_name] = new_state_dict
      dataset_info = checkpoint["model_data"]["dataset_info"]
      if dataset_info is not None:
          new_target = dataset_info.targets.pop(head_name)
          if new_target is not None:
              dataset_info.targets["energy"] = new_target
              checkpoint["model_data"]["dataset_info"] = dataset_info
      return checkpoint


  checkpoint = torch.load(
      "your_path_to_checkpoint/model.ckpt", map_location="cpu", weights_only=False
  )
  new_target_name = "mtt::energy_pbe0"  # specify the name of the new target here
  checkpoint = set_output_head(checkpoint, new_target_name)
  torch.save(checkpoint, "new_checkpoint.ckpt")


You need to specify the path to the trained model checkpoint and the name of the new
target in the script. This name should match the new target name in the
``options.yaml`` file. The modified checkpoint will be saved as ``new_checkpoint.ckpt``.
Finally, you can run the ``mtt export new_checkpoint.ckpt`` command to convert the
model to the ``metatomic`` format and use it in ASE and LAMMPS calculations, as
described in the `metatomic documentation`_.

.. _metatomic documentation: https://docs.metatensor.org/metatomic/latest/engines/index.html


Fitting to a new set of properties
----------------------------------

Training on a new set of properties is another common use case for transfer learning. It
can be done in a similar way as training on a new level of theory. The only difference
is that the new targets need to be properly set in the ``options.yaml`` file. More
information about fitting the generic targets can be found in the :ref:`Fitting generic
targets <fitting-generic-targets>` section of the documentation.
"""
