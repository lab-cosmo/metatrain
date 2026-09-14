.. _checkpoints:

Restarting and Checkpoints
##########################

During their training process, models will produce checkpoints. These have the ``.ckpt``
extension, as opposed to the ``.pt`` extension of exported models. A final checkpoint
will always be saved together with its corresponding exported model at the end of
training. For example, if the final model is saved as ``model.pt``, a ``model.ckpt``
will also be saved. In addition, checkpoints are saved at regular intervals during
training. These can be found in the ``outputs`` directory.

While exported models are used for inference, the main use of checkpoints is to resume
training from a certain point. This is useful if you want to continue training a model
after it has been interrupted, or if you want to fine-tune a model on a new dataset.

Restarting a training
---------------------

The sub-command to continue training from a checkpoint is

.. code-block:: bash

    mtt train options.yaml --restart model.ckpt

Automatic restarting
^^^^^^^^^^^^^^^^^^^^

When restarting multiple times (for example, when training an expensive model
or running on an HPC cluster with short time limits), it is useful to be able
to train and restart multiple times with the same command.

In ``metatrain``, this functionality is provided via the ``--restart auto``
flag of ``mtt train``. This flag will automatically restart
the training from the last checkpoint, if one is found in the ``outputs/``
of the current directory. If no checkpoint is found, the training will start
from scratch.


Exporting models
----------------

Checkpoints can also be turned into exported models using the ``export`` sub-command.
The command requires the *architecture name* and the saved checkpoint *path* as
positional arguments

.. code-block:: bash

    mtt export model.ckpt -o model.pt

or

.. code-block:: bash

    mtt export model.ckpt --output model.pt

Exporting remote models
^^^^^^^^^^^^^^^^^^^^^^^

For a export of distribution of models the ``export`` command also supports parsing
models from remote locations. To export a remote model you can provide a URL instead of
a file path.

.. code-block:: bash

    mtt export https://my.url.com/model.ckpt --output model.pt

Downloading private HuggingFace models is also supported, by specifying the
corresponding API token with the ``--token`` flag or the ``HF_TOKEN`` environment
variable.

Keep in mind that a checkpoint (``.ckpt``) is only a temporary file, which can have
several dependencies and may become unusable if the corresponding architecture is
updated. In constrast, exported models (``.pt``) act as standalone files. For long-term
usage, you should export your model! Exporting a model is also necessary if you want to
use it in other frameworks, especially in molecular simulations (see the
:ref:`tutorials`).

Adding information about models to checkpoints
----------------------------------------------

You can insert the model name, a description, the list of authors and references
into the model. This information will be saved either in the existing checkpoint or in
the exported model. In the first case, a new checkpoint file with attached metadata
will be created. In the second case, the model will be exported with the metadata attached.
This metadata will be displayed to users when the model is used, for example, in molecular
dynamics simulations.

.. code-block:: bash

    mtt export model.ckpt --metadata metadata.yaml

This will export the model with the metadata attached. Alternatively, if the intermediate
checkpoint with the metadata attached is needed, you can run

.. code-block:: bash

    mtt export model.ckpt -o model-with-metadata.ckpt --metadata metadata.yaml

The ``metadata.yaml`` file should have the following structure:

.. code-block:: yaml

    name: My model
    description: This model was trained on the QM9 dataset.
    authors:
      - John Doe
      - Jane Doe
    references:
      model:
        - https://arxiv.org/abs/1234.5678

You can also add additional keywords like additional references to the metadata
file. The fields are the same for :class:`ModelMetadata
<metatomic.torch.ModelMetadata>` class from metatomic.
