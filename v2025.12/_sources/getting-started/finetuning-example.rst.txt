.. _fine-tuning-example:

Finetuning example
==================

.. warning::

  Finetuning is currently only available for the PET architecture.


This is a simple example for fine-tuning PET-MAD (or a general PET model), that
can be used as a template for general fine-tuning with metatrain.
Fine-tuning a pretrained model allows you to obtain a model better suited for
your specific system. You need to provide a dataset of structures that have
been evaluated at a higher reference level of theory, usually DFT. Fine-tuning
a universal model such as PET-MAD allows for reasonable model performance even if little training
data is available.
It requires using a pre-trained model checkpoint with the ``mtt train`` command and setting the
new targets corresponding to the new level of theory in the ``options.yaml`` file.


In order to obtain a pretrained model, you can use a PET-MAD checkpoint from huggingface

.. code-block:: bash

  wget https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt

Next, we set up the ``options.yaml`` file. We can specify the fine-tuning method
in the ``finetune`` block in the ``training`` options of the ``architecture``.
Here, the basic ``full`` option is chosen, which finetunes all weights of the model.
All available fine-tuning methods are found in the advanced concepts
:ref:`Fine-tuning <fine-tuning>`. This section discusses implementation details,
options and recommended use cases. Other fine-tuning options can be simply substituted in this script,
by changing the ``finetune`` block.

Furthermore, you need to specify the checkpoint, that you want to fine-tune in
the ``read_from`` option.

A simple ``options.yaml`` file for this task could look like this:

Training on a new level of theory is a common use case for transfer learning. Let's

.. code-block:: yaml

  architecture:
    name: pet
    training:
      num_epochs: 1000
      learning_rate: 1e-5
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt
  training_set:
    systems:
        read_from: dataset.xyz
        reader: ase
        length_unit: angstrom
    targets:
        energy:
            quantity: energy
            read_from: dataset.xyz
            reader: ase
            key: energy
            unit: eV
            forces:
                read_from: dataset.xyz
                reader: ase
                key: forces
            stress:
                read_from: dataset.xyz
                reader: ase
                key: stress

  test_set: 0.1
  validation_set: 0.1

In this example, we specified generic but reasonable ``num_epochs`` and ``learning_rate``
parameters. The ``learning_rate`` is chosen to be relatively low to stabilise
training.

.. warning::

  Note that in ``targets`` we use the PET-MAD ``energy`` head. This means, that there won't be a new head
  for the new reference energies provided in your dataset. This can lead to bad performance, if the reference
  energies differ from the ones used in pretraining (different levels of theory, or different electronic structure
  software used). In future it is recommended to create a new ``energy`` target for the new level of theory.
  Find more about this in :ref:`Transfer-Learning <transfer-learning>`



We assumed that the pre-trained model is trained on the dataset ``dataset.xyz`` in which
energies are written in the ``energy`` key of the ``info`` dictionary of the
energies. Additionally, forces and stresses should be provided with corresponding keys
which you can specify in the ``options.yaml`` file under ``targets``.
Further information on specifying targets can be found in the :ref:`data section of the Training YAML Reference
<data-section>`.

.. note::

  It is important that the ``length_unit`` is set to ``angstrom`` and the ``energy`` ``unit`` is ``eV`` in order
  to match the units PET-MAD was trained on. If your dataset has different energy units, it is
  necessary to convert it to ``eV`` before fine-tuning.


After setting up your ``options.yaml`` file, finetuning can then simply be run
via ``mtt train options.yaml``.


Further fine-tuning examples can be found in the
`AtomisticCookbook <https://atomistic-cookbook.org/examples/pet-finetuning/pet-ft.html>`_
