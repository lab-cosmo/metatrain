Fine-tuning
===========

.. warning::

  This section of the documentation is only relevant for PET model so far.

This section describes the process of fine-tuning a pre-trained model to
adapt it to new tasks or datasets. Fine-tuning is a common technique used
in transfer learning, where a model is trained on a large dataset and then
fine-tuned on a smaller dataset to improve its performance on specific tasks.
So far the fine-tuning capabilities are only available for PET model.


Fine-Tuning PET Model with LoRA
-------------------------------

Fine-tuning a PET model using LoRA (Low-Rank Adaptation) can significantly
enhance the model's performance on specific tasks while reducing the
computational cost. Below are the steps to fine-tune a PET model from
``metatrain.pet`` using LoRA.

What is LoRA?
^^^^^^^^^^^^^

LoRA (Low-Rank Adaptation) stands for a Parameter-Efficient Fine-Tuning (PEFT)
technique used to adapt pre-trained models to new tasks by introducing low-rank
matrices into the model's architecture. This approach reduces the number of
trainable parameters, making the fine-tuning process more efficient and less
resource-intensive. LoRA is particularly useful in scenarios where computational
resources are limited or when quick adaptation to new tasks is required.

Given a pre-trained model with the weights matrix :math:`W_0`, LoRA introduces
low-rank matrices :math:`A` and :math:`B` of a rank :math:`r` such that the
new weights matrix :math:`W` is computed as:

.. math::

  W = W_0 + \frac{\alpha}{r} A B

where :math:`\alpha` is a regularization factor that controls the influence
of the low-rank matrices on the model's weights. By adjusting the rank :math:`r`
and the regularization factor :math:`\alpha`, you can fine-tune the model
to achieve better performance on specific tasks.

Prerequisites
^^^^^^^^^^^^^

1. Train the Base Model. You can train the base model using the command:
``mtt train options.yaml``. Alternatively, you can use a pre-trained
foundational model, if you have access to its checkpoint. After this training,
you will find the checkpoint file called ``best_model.ckpt`` in the training
directory.

1. Set the LoRA parameters in the ``architecture.training``
section of the ``options.yaml``:

.. code-block:: yaml

  architecture:
    training:
      LORA_RANK: <desired_rank>
      LORA_ALPHA: <desired_alpha>
      USE_LORA_PEFT: True

These parameters control whether to use LoRA for pre-trained model fine-tuning
(``USE_LORA_PEFT``), the rank of the low-rank matrices introduced by LoRA
(``LORA_RANK``), and the regularization factor for the low-rank matrices
(``LORA_ALPHA``).

4. Run ``mtt train options.yaml -c best_model.ckpt`` to fine-tune the model.
The ``-c`` flag specifies the path to the pre-trained model checkpoint.

Fine-Tuning Options
^^^^^^^^^^^^^^^^^^^

When ``USE_LORA_PEFT`` is set to ``True``, the original model's weights will be
frozen, and only the adapter layers introduced by LoRA will be trained. This
allows for efficient fine-tuning with fewer parameters. If ``USE_LORA_PEFT`` is
set to ``False``, all the weights of the model will be trained. This can lead to
better performance on the specific task but may require more computational
resources, and the model may be prone to overfitting (i.e. loosing accuracy on
the original training set).

