.. _adding-new-architecture:

Adding a new architecture
=========================

This page describes the required classes and files necessary for adding a new
architecture to ``metatrain`` as experimental or stable architecture as
described on the :ref:`architecture-life-cycle` page.

What is a ``metatrain`` architecture?
-------------------------------------

To work with ``metatrain`` any architecture has to follow the same public API to
be called correctly within the :py:func:`metatrain.cli.train` function to
process the user's options. In brief, the core of the ``train`` function looks
similar to these lines

.. code-block:: python

    from architecture import __model__ as Model
    from architecture import __trainer__ as Trainer

    hypers = {...}
    dataset_info = DatasetInfo()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)

        trainer = Trainer.load_checkpoint(
            checkpoint, hypers=hypers["training"], context="restart")
        model = Model.load_checkpoint(checkpoint, context="restart")
        model = model.restart(dataset_info)
    else:
        trainer = Trainer(hypers["training"])

        if hasattr(hypers["training"], "finetune"):
            checkpoint = hypers["training"]["finetune"]["read_from"]
            model = Model.load_checkpoint(path=checkpoint, context="finetune")
        else:
            model = Model(hypers["model"], dataset_info)

    trainer.train(
        model=model,
        dtype=dtype,
        devices=[],
        train_datasets=[],
        val_datasets=[],
        checkpoint_dir="path",
    )

    model.save_checkpoint("model.ckpt")

    mts_atomistic_model = model.export()
    mts_atomistic_model.export("model.pt", collect_extensions="extensions/")

General code structure
----------------------

To follow this, a new architecture has to define two classes

- A ``Model`` class, defining the core of the architecture. This class must
  implement the interface documented in
  :py:class:`metatrain.utils.abc.ModelInterface`
- A ``Trainer`` class, used to train an architecture and produce a model that
  can be evaluated and exported. This class must implement the interface
  documented below in :py:class:`metatrain.utils.abc.TrainerInterface`.

.. note::

    ``metatrain`` does not know the types and numbers of targets/datasets an
    architecture can handle. As a result, it cannot generate useful error
    messages when a user attempts to train an architecture with unsupported
    target and dataset combinations. Therefore, it is the responsibility of the
    architecture developer to verify if the model and the trainer support the
    provided train_datasets and val_datasets passed to the Trainer, as well as
    the dataset_info passed to the model.

The architecture must also define a documentation file which contains the
default hyperparameters, along with their types and descriptions.

To comply with this design each architecture has to implement four files
inside a new architecture directory, either inside the ``experimental``
subdirectory or in the ``root`` of the Python source if the new architecture
already complies with all requirements to be stable. The usual structure of
architecture looks as

.. code-block:: text

    myarchitecture
        ├── __init__.py
        ├── documentation.py
        ├── model.py
        └── trainer.py

.. note::
    Because achitectures can live in either ``src/metatrain/<architecture>``,
    ``src/metatrain/experimental/<architecture>``, or
    ``src/metatrain/deprecated/<architecture>``; the code inside should use
    absolute imports use the tools provided by metatrain.

    .. code-block:: python

        # do not do this
        from ..utils.dtype import dtype_to_str

        # Do this instead
        from metatrain.utils.dtype import dtype_to_str

Model class (``model.py``)
--------------------------

A model class has to follow the interface defined in
:py:class:`~metatrain.utils.abc.ModelInterface`. That is, all the
methods that are marked as abstract in the interface must be implemented
with the indicated API (same arguments and same return). At first sight,
the interface might feel overwhelming, therefore here is a summary of the
steps to take to implement a new model class:


- Implement the ``__init__`` method, which takes as input the model hyperparameters
  and the dataset information. This should initialize your model.
- Implement the ``forward`` method, which defines the forward pass of the model.
- Add some class attributes with ``__names_like_this__`` that will help metatrain
  understand how to treat your model. They are listed and described in the
  :py:class:`~metatrain.utils.abc.ModelInterface` documentation.
- Implement the rest of abstract methods, which in general deal with handling
  checkpoints, exporting the model, and restarting training from a checkpoint.

Here is an incomplete example of what a model implementation looks like:

.. code-block:: python

    import torch
    from metatomic.torch import DatasetInfo, ModelMetadata

    from metatrain.utils.abc import ModelInterface

    class MyModel(ModelInterface):
        __checkpoint_version__ = 1
        __supported_devices__ = ["cuda", "cpu"]
        __supported_dtypes__ = [torch.float64, torch.float32]
        __default_metadata__ = ModelMetadata(
            references={"implementation": ["ref1"], "architecture": ["ref2"]}
        )

        def __init__(self, hypers: dict, dataset_info: DatasetInfo):
            super().__init__(hypers, dataset_info)

            # To access hyperparameters, one can use self.hypers, whose
            # defaults are defined in the documentation.py file.
            self.hypers["size"]
            ...

        # Here one would implement the rest of the abstract methods

Trainer class (``trainer.py``)
------------------------------

A trainer class has to follow the interface defined in
:py:class:`~metatrain.utils.abc.TrainerInterface`. That is, all the
methods that are marked as abstract in the interface must be implemented
with the indicated API (same arguments and same return). We recommend
looking at existing implementations of trainers for inspiration. They
will look something like this:

.. code-block:: python

    from metatrain.utils.abc import TrainerInterface

    class MyTrainer(TrainerInterface):
        __checkpoint_version__ = 1

        def __init__(self, hypers: dict):
            super().__init__(hypers)
            # To access hyperparameters, one can use self.hypers, whose
            # defaults are defined in the documentation.py file.
            self.hypers["learning_rate"]
            ...

        # Here one would implement the rest of the abstract methods

Init file (``__init__.py``)
---------------------------

You are free to name the ``Model`` and ``Trainer`` classes as you want. These
classes should then be made available in the ``__init__.py`` under the names
``__model__`` and ``__trainer__`` so metatrain knows where to find them.
``__init__.py`` must also contain definition for the original ``__authors__``
and current ``__maintainers__`` of the architecture.

.. code-block:: python

    from .model import ModelInterface
    from .trainer import TrainerInterface

    # class to use as the architecture's model
    __model__ = ModelInterface
    # class to use as the architecture's trainer
    __trainer__ = TrainerInterface

    # List of the original authors of the architecture, each with an email
    # address and GitHub handle.
    #
    # These authors are not necessarily currently in charge of maintaining the code
    __authors__ = [
        ("Jane Roe <jane.roe@myuniversity.org>", "@janeroe"),
        ("John Doe <john.doe@otheruniversity.edu>", "@johndoe"),
    ]

    # Current maintainers of the architecture code, using the same
    # style as ``__authors__``
    __maintainers__ = [("Joe Bloggs <joe.bloggs@sotacompany.com>", "@joebloggs")]

.. _newarchitecture-documentation:

Documentation (``documentation.py``)
------------------------------------

The documentation file is used to define:

- The hyperparameters for the model class.
- The hyperparameters for the trainer class.
- The text that will go to the online documentation for the architecture.

.. warning::

    This file is meant to be imported separately to generate the
    documentation page for the architecture without needing the
    extra dependencies that the architecture might require.

    Therefore, all imports in this file should be absolute and this
    file should not import the rest of the architecture code unless
    the architecture has no extra dependencies.

Bare minimum
^^^^^^^^^^^^
We understand that during development of a new architecture expecting full
documentation for all hyperparameters is unreasonable. Therefore, ``metatrain``
will work with a very minimal ``documentation.py`` file containing only the
default hyperparameters for both the model and the trainer. One just
needs to define a ``ModelHypers`` and a ``TrainerHypers``, for the hypers of the
model and the trainer respectively.

.. code-block:: python

    # This is the most minimal documentation.py file possible.
    # Something like this should only be used during development.

    # Default hyperparameters for the model
    class ModelHypers:
        size = 150
        mode = "strict"

    # Default hyperparameters for the trainer
    class TrainerHypers:
        learning_rate = 1e-3
        lr_scheduler = "CosineAnnealing"

.. note::

    The name of these classes (``ModelHypers`` and ``TrainerHypers``), as well
    as the file they are in (``documentation.py``) are **mandatory**.
    ``metatrain`` will look for these specific names when loading the
    architecture.

    This rigidity allows ``metatrain`` to easily generate documentation pages
    and maintain a consistent experience across all architectures.

For an experimental architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an architecture to be considered accepted as "experimental" into the
main ``metatrain`` distribution, ``documentation.py`` should at least contain:

- A minimal docstring at the top of the file with at least a short description
  of the architecture. It should contain as a title the name of the architecture,
  underlined with equal signs (``=``).
- Some documentation for each hyperparameter.

For example, this would be a valid ``documentation.py`` file for an
experimental architecture:

.. code-block:: python

    """
    My architecture
    ===============

    This is an architecture that does amazing things.
    """

    class ModelHypers:

        size = 150
        """Size of the model's hidden layers."""
        mode = "strict"
        """Mode of operation for the model."""

    class TrainerHypers:
        learning_rate = 1e-3
        """Initial learning rate for the optimizer."""
        lr_scheduler = "CosineAnnealing"
        """Type of learning rate scheduler to use."""

You can check :ref:`this section <newarchitecture-documentation-page>` to
understand how the module docstring will be used to generate the documentation
page for the architecture.

For a stable architecture
^^^^^^^^^^^^^^^^^^^^^^^^^

Going from experimental to stable architecture requires one last step:
documentation of the hyperparameters types. This is done using ``TypedDict``
and Python's type hinting system, and it allows ``metatrain`` to automatically
validate user inputs. By doing validation, ``metatrain`` can give users
meaningful error messages when the provided hyperparameters are invalid,
avoiding errors deep inside the architecture that would be harder to understand.

Here is the example of the previous ``documentation.py`` file, now ready for
the architecture to be considered stable:

.. code-block:: python

    """
    My architecture
    ===============

    This is an architecture that does amazing things.
    """
    from typing_extensions import TypedDict
    from typing import Literal

    class ModelHypers(TypedDict):

        size: int = 150
        """Size of the model's hidden layers."""
        mode: Literal["strict", "lenient"] = "strict"
        """Mode of operation for the model."""

    class TrainerHypers(TypedDict):
        learning_rate: float = 1e-3
        """Initial learning rate for the optimizer."""
        lr_scheduler: Literal["CosineAnnealing", "StepLR"] = "CosineAnnealing"
        """Type of learning rate scheduler to use."""

.. note::

    It is important to use ``typing_extensions.TypedDict`` instead of
    ``typing.TypedDict`` for compatibility with ``python <= 3.12`` in pydantic's
    validation system.

With this, you will be almost ready to have your architecture accepted as stable.
The last step is to update the ``Model`` and ``Trainer`` classes so that they are
aware of the hyperparameter types. This will help static type checkers like mypy
catch bugs in your code, as well as improving the development experience in IDE's
like VSCode or PyCharm. To do this, you just have to:

- Make your model and trainer classes inherit from ``ModelInterface[ModelHypers]``
  and ``TrainerInterface[TrainerHypers]`` respectively, instead of just
  ``ModelInterface`` and ``TrainerInterface``.
- Add the hypers type annotation to the ``hypers`` argument of the ``__init__``
  method of both classes, as well as any other method that takes hyperparameters
  as input (like ``Trainer.load_checkpoint``).

For example, for the model:

.. code-block:: python

    import torch
    from metatomic.torch import DatasetInfo, ModelMetadata

    from metatrain.utils.abc import ModelInterface

    # New import to get the ModelHypers type
    from .documentation import ModelHypers

    class MyModel(ModelInterface[ModelHypers]): # Add the hypers type here
        __checkpoint_version__ = 1
        __supported_devices__ = ["cuda", "cpu"]
        __supported_dtypes__ = [torch.float64, torch.float32]
        __default_metadata__ = ModelMetadata(
            references={"implementation": ["ref1"], "architecture": ["ref2"]}
        )

        # Type hint the hypers argument of __init__
        def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo):
            super().__init__(hypers, dataset_info)
            ...

.. _newarchitecture-documentation-page:

Documentation page
^^^^^^^^^^^^^^^^^^

By following the guidelines for documenting hyperparameters, ``metatrain``
**will automatically generate a documentation page for the new architecture**.
This documentation page will contain information about how to install your
architecture, the default hyperparameters, and the descriptions of all
the hyperparameters for both the model and the trainer.

The documentation page will be generated from the docstring at the top of the
``documentation.py`` file, as well as the ``ModelHypers`` and ``TrainerHypers``
classes defined there. Here is the description of how the docstring will
be generated:

.. autoclass:: src.architectures.generate.ArchitectureDocVariables
   :no-index:
   :members:
   :undoc-members:

.. _ckpt_version:

Checkpoint versioning
----------------------

Checkpoints are used to save the weights of a models and the state of the
trainer to disk, enabling to restart interupted training runs, to fine-tune
existing models on new dataset, and to export standalone models based on
TorchScript.

A checkpoint created for one version might need to be read again
by a later version of the architecture, where the internal structure might have
changed. To enable this, all ``Model`` classes are required to have a
``__checkpoint_version__`` class attribute containing the version of the
checkoint, as a strictly inreasing integer. Additionally, architectures should
provide an ``upgrade_checkpoint(checkpoint: Dict) -> Dict`` function, that will
be called when a user is trying to load some outdated checkpoint. This function
is responsible for updating the checkpoint data and returning a checkpoint
compatible with the current version.

Similarly, the ``Trainer`` state is also saved in checkpoint and used to restart
training. All trainer must thus have a ``__checkpoint_version__`` class
attribute as well as a ``upgrade_checkpoint(checkpoint: Dict) -> Dict`` function
to updgrade from previous checkpoints.
