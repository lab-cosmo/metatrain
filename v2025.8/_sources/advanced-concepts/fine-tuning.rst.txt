Fine-tuning
===========

.. warning::

  This section of the documentation is only relevant for PET model so far.

This section describes the process of fine-tuning a pre-trained model to
adapt it to new tasks or datasets. Fine-tuning is a common technique used
in machine learning, where a model is trained on a large dataset and then
fine-tuned on a smaller dataset to improve its performance on specific tasks.
So far the fine-tuning capabilities are only available for PET model.


.. note::

  Please note that the fine-tuning recommendations in this section are not universal
  and require testing on your specific dataset to achieve the best results. You might
  need to experiment with different fine-tuning strategies depending on your needs.


Basic Fine-tuning
-----------------

The basic way to fine-tune a model is to use the ``mtt train`` command with the
available pre-trained model defined in an ``options.yaml`` file. In this case, all the
weights of the model will be adapted to the new dataset. In contrast to to the
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

We recommend to use a lower learning rate than the one used for the original training, as
this will help stabilizing the training process. I.e. if the default learning rate is
``1e-4``, you can set it to ``1e-5`` or even lower, using the following in the
``options.yaml`` file:

.. code-block:: yaml

  architecture:
    training:
      learning_rate: 1e-5

Please note, that in the case of the basic fine-tuning, the composition model weights
will be taken from the checkpoint and not adapted to the new dataset.

The basic fine-tuning startgy is a good choice in the case when the level of theory
which is used for the original training is the same, or at least similar to the one used for
the new dataset. However, since this is not always the case, we also provide more advanced
fine-tuning strategies described below.


Fine-tuning model Heads
-----------------------

Adapting all the model weights to a new dataset is not always the best approach. If the new
dataset consist of the same or similar data computed with a slightly different level of theory
compared to the pre-trained models' dataset, you might want to keep the learned representations
of the crystal structures and only adapt the readout layers (i.e. the model heads) to the new
dataset.

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
experiment with their different combinations if needed. You might also consider using a lower
learning rate, e.g. ``1e-5`` or even lower, to stabilize the training process.


LoRA Fine-tuning
----------------

If the conceptually new type of structures is introduced in the new dataset, tuning only the
model heads might not be sufficient. In this case, you might need to adapt the internal
representations of the crystal structures. This can be done using the LoRA technique. However,
in this case the model heads will be not adapted to the new dataset, so conceptually the
level of theory should be consistent with the one used for the pre-trained model.

What is LoRA?
^^^^^^^^^^^^^

LoRA (Low-Rank Adaptation) stands for a Parameter-Efficient Fine-Tuning (PEFT)
technique used to adapt pre-trained models to new tasks by introducing low-rank
matrices into the model's architecture.

Given a pre-trained model with the weights matrix :math:`W_0`, LoRA introduces
low-rank matrices :math:`A` and :math:`B` of a rank :math:`r` such that the
new weights matrix :math:`W` is computed as:

.. math::

  W = W_0 + \frac{\alpha}{r} A B

where :math:`\alpha` is a regularization factor that controls the influence
of the low-rank matrices on the model's weights. By adjusting the rank :math:`r`
and the regularization factor :math:`\alpha`, you can fine-tune the model
to achieve better performance on specific tasks.

To use LoRA for fine-tuning, you need to provide the pre-trained model checkpoint with
the ``mtt train`` command and specify the LoRA parameters in the ``options.yaml`` file:

.. code-block:: yaml

  architecture:
    training:
      finetune:
        method: "lora"
        read_from: path/to/pre-trained-model.ckpt
        config:
          alpha: 0.1
          rank: 4

These parameters control the rank of the low-rank matrices introduced by LoRA
(``rank``), and the regularization factor for the low-rank matrices (``alpha``).
By selecting the LoRA rank and the regularization factor, you can control the
amount of adaptation to the new dataset. Using lower values of the rank and
the regularization factor will lead to a more conservative adaptation, which can help
balancing the performance of the model on the original and new datasets.

We recommend to start with the LoRA parameters listed above and experiment with
different values if needed. You might also consider using a lower learning rate,
e.g. ``1e-5`` or even lower, to stabilize the training process.


Fine-tuning on a new level of theory
------------------------------------

If the new dataset is computed with a totally different level of theory compared to the
pre-trained model, which includes, for instance, the different composition energies,
or you want to fine-tune the model on a completely new target, you might need to consider
the transfer learning approach and introduce a new target in the
``options.yaml`` file. More details about this approach can be found in the
:ref:`Transfer Learning <transfer-learning>` section of the documentation.

