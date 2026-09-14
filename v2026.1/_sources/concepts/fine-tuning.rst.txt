.. _label_fine_tuning_concept:

Fine-tune a pre-trained model
=============================

.. warning::

  Finetuning may not be supported by every architecture and if supported the syntax to
  start a finetuning may be different from how it is explained here.
  This section describes the process of fine-tuning a pre-trained model to adapt it to
  new tasks or datasets. Fine-tuning is a common technique used in machine learning,
  where a model is trained on a large dataset and then fine-tuned on a smaller dataset
  to improve its performance on specific tasks. So far the fine-tuning capabilities are
  only available for PET model.

There is a complete example in the tutorial section
:ref:`sphx_glr_generated_examples_0-beginner_02-fine-tuning.py`.

.. note::

  Please note that the fine-tuning recommendations in this section are not universal
  and require testing on your specific dataset to achieve the best results. You might
  need to experiment with different fine-tuning strategies depending on your needs.


Basic Fine-tuning
-----------------

The basic way to fine-tune a model is to use the ``mtt train`` command with the
available pre-trained model defined in an ``options.yaml`` file. In this case, all the
weights of the model will be adapted to the new dataset. In contrast to the
training continuation, the optimizer and scheduler state will be reset. You can still
adjust the training hyperparameters in the ``options.yaml`` file, but the model
architecture will be taken from the checkpoint.
To set the path to the pre-trained model checkpoint, you need to specify the
``read_from`` parameter in the ``options.yaml`` file:

.. code-block:: yaml

  architecture:
    training:
      finetune:
        method: "full" # This stands for the full fine-tuning
        read_from: path/to/checkpoint.ckpt

We recommend to use a lower learning rate than the one used for the original training,
as this will help stabilizing the training process. I.e. if the default learning rate is
``1e-4``, you can set it to ``1e-5`` or even lower, using the following in the
``options.yaml`` file:

.. code-block:: yaml

  architecture:
    training:
      learning_rate: 1e-5

Please note, that in most use cases you should invoke a new energy head by specifying
a new energy variant. A variant is a version of a target quantity, such as ``energy``.
A model can have multiple variants, that can be selected during training and inference.
More on variants can be found in `metatomic`_

.. _metatomic: https://docs.metatensor.org/metatomic/latest/engines/index.html

Variant names follow the simple pattern ``energy/{variantname}``, where we used
``energy`` as the target quantity. A reasonable name could be the energy functional or
level of theory your finetuning dataset was trained on, e.g. ``energy/pbe``,
``energy/SCAN`` or even ``energy/dataset1``. Further we recommend adding a short
:attr:`description` for the new variant, that you can specify in ``description`` of
your ``options.yaml`` file.


.. code-block:: yaml

  training_set:
      systems:
        read_from: path/to/dataset.xyz
        length_unit: angstrom
      targets:
        energy/<variantname>:
          quantity: energy
          key: <energy-key>
          unit: <energy-unit>
          description: "description of your variant"


The new energy variant can be selected for evaluation either with ``mtt eval`` by
specifying it in the options.yaml for evaluation:

.. code-block:: yaml

   systems: path/to/dataset.xyz
   targets:
     energy/<variantname>:
       key: <energy-key>
       unit: <energy-unit>
       forces:
         key: forces

When using the finetuned model in simulation engines the default target name expected
by the ``metatomic`` package in order to use the model in ASE and LAMMPS calculations
is ``energy``. When loading the model in ``metatomic`` you have to specify which
variant should be used for energy and force prediction. You can find an example for how
to do this in the tutorial
(see :ref:`sphx_glr_generated_examples_0-beginner_02-fine-tuning.py`) and more in the
`metatomic documentation`_.

.. _metatomic documentation: https://docs.metatensor.org/metatomic/latest/engines/index.html


Until here, our model would train on all weights of the model, create a new energy head
and a new composition model.

The basic fine-tuning strategy is a good choice for most use cases. Below, we present
a few more advanced topics.

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
          energy/<variantname>: energy # inherit weights from the "energy" head

The ``inherit_heads`` parameter is a dictionary mapping the new trainable
targets specified in the ``training_set/targets`` section to the existing
targets in the pre-trained model. The weights of the corresponding heads and
last layers will be copied from the source heads to the destination heads
instead of random initialization. These weights are still trainable and
will be adapted to the new dataset during the training process.


Multi-fidelity training
-----------------------
So far the old head is left untouched, but it is rendered useless, due to changing
deeper weights of the model. If you want to fine-tune and retain multiple functional
heads, the recommended way is to do full fine-tuning on a new target, but keep
training the old energy head as well. This will leave you with a model capable of
using different variants for energy and force prediction. Again, you are able to select
the preferred head in ``LAMMPS`` or when creating a ``metatomic`` calculator object.
Thus, you should specify both variants in the ``targets`` section of your
``options.yaml``. In the code snippet, we additionally assume that the energy labels
come from different datasets. Please note, if you have both references in one file,
they can be selected by selecting the corresponding keys from the same system.
the same dataset.

.. code-block:: yaml

  training_set:
      - systems:
            read_from: dataset_1.xyz
            length_unit: angstrom
        targets:
            energy/<variant1>:
                quantity: energy
                key: my_energy_label1
                unit: eV
                description: 'my variant1 description'
      - systems:
            read_from: dataset_2.xyz
            length_unit: angstrom
        targets:
            energy/<variant2>:
                quantity: energy
                key: my_energy_label2
                unit: eV
                description: 'my variant2 description'



You can find more about setting up training with multiple files in the
:ref:`Training YAML reference <train_yaml_config>`.


Training only the head weights can be an alternative, if one wants to keep the old energy
head, but the reference data it was trained are not available. In that case, the
internal model weights are frozen, and only the weights of the new target are trained.


Fine-tuning model Heads only
----------------------------

Adapting all the model weights to a new dataset is not always the best approach. If the
new dataset consist of the same or similar data computed with a slightly different level
of theory compared to the pre-trained models' dataset, you might want to keep the
learned representations of the crystal structures and only adapt the readout layers
(i.e. the model heads) to the new dataset.
In this case, the ``mtt train`` command needs to be accompanied by the specific training
options in the ``options.yaml`` file. The following options need to be set:

.. code-block:: yaml

  architecture:
    training:
      finetune:
        method: "heads"
        read_from: path/to/checkpoint.ckpt
        config:
          head_modules: ['node_heads', 'edge_heads']
          last_layer_modules: ['node_last_layers', 'edge_last_layers']


The ``method`` parameter specifies the fine-tuning method to be used and the
``read_from`` parameter specifies the path to the pre-trained model checkpoint. The
``head_modules`` and ``last_layer_modules`` parameters specify the modules to be
fine-tuned. Here, the ``node_*`` and ``edge_*`` modules represent different parts of the
model readout layers related to the atom-based and bond-based features. The
``*_last_layer`` modules are the last layers of the corresponding heads, implemented as
multi-layer perceptron (MLPs). You can select different combinations of the node and
edge heads and last layers to be fine-tuned.

We recommend to first start the fine-tuning including all the modules listed above and
experiment with their different combinations if needed. You might also consider using a
lower learning rate, e.g. ``1e-5`` or even lower, to stabilize the training process.
