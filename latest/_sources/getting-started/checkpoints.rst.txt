Checkpoints
###########

During their training process, models will produce checkpoints. These have the ``.ckpt``
extension, as opposed to the ``.pt`` extension of exported models. A final checkpoint
will always be saved together with its corresponding exported model at the end of
training. For example, if the final model is saved as ``model.pt``, a ``model.ckpt``
will also be saved. In addition, checkpoints are saved at regular intervals during
training. These can be found in the ``outputs`` directory.

While exported models are used for inference, the main use of checkpoints is to resume
training from a certain point. This is useful if you want to continue training a model
after it has been interrupted, or if you want to fine-tune a model on a new dataset.

The sub-command to continue training from a checkpoint is

.. code-block:: bash

    mtt train options.yaml --continue model.ckpt

or

.. code-block:: bash

    mtt train options.yaml -c model.ckpt

Checkpoints can also be turned into exported models using the ``export`` sub-command.

.. code-block:: bash

    mtt export model.ckpt -o model.pt

or

.. code-block:: bash

    mtt export model.ckpt --output model.pt

Keep in mind that a checkpoint (``.ckpt``) is only a temporary file, which can have
several dependencies and may become unusable if the corresponding architecture is
updated. In constrast, exported models (``.pt``) act as standalone files.
For long-term usage, you should export your model! Exporting a model is also necessary
if you want to use it in other frameworks, especially in molecular simulations
(see the :ref:`tutorials`).
