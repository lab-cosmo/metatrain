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
The command requires the *architecture name* and the saved checkpoint *path* as
positional arguments

.. code-block:: bash

    mtt export model.ckpt -o model.pt

or

.. code-block:: bash

    mtt export model.ckpt --output model.pt

Adding information about models
-------------------------------

You can also insert the model name, a description, the list of authors and references
into the model. This information will be saved in the exported model and can will be
displayed to users when the model is used, for example, in molecular dynamics
simulations.

.. code-block:: bash

    mtt export model.ckpt --metadata metadata.yaml

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

You can also add additional keywords like additional references to the metadata file.
The fields are the same for :class:`ModelMetadata
<metatensor.torch.atomistic.ModelMetadata>` class from metatensor.

Exporting remote models
-----------------------

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
