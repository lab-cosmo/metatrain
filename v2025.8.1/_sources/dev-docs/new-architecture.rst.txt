.. _adding-new-architecture:

Adding a new architecture
=========================

This page describes the required classes and files necessary for adding a new
architecture to ``metatrain`` as experimental or stable architecture as
described on the :ref:`architecture-life-cycle` page.

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


To follow this, a new architecture has to define two classes

- a ``Model`` class, defining the core of the architecture. This class must
  implement the interface documented below in
  :py:class:`metatrain.utils.abc.ModelInterface`
- a ``Trainer`` class, used to train an architecture and produce a model that
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

To comply with this design each architecture has to implement a couple of files
inside a new architecture directory, either inside the ``experimental``
subdirectory or in the ``root`` of the Python source if the new architecture
already complies with all requirements to be stable. The usual structure of
architecture looks as

.. code-block:: text

    myarchitecture
        ├── model.py
        ├── trainer.py
        ├── __init__.py
        ├── default-hypers.yaml
        └── schema-hypers.json

.. note::
    A new architecture doesn't have to be registered somewhere in the file tree
    of ``metatrain``. Once a new architecture folder with the required files is
    created ``metatrain`` will include the architecture automatically.

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

.. autoclass:: metatrain.utils.abc.ModelInterface
    :members:

Defining a new model can then be done as follow;

.. code-block:: python

    from metatomic.torch import ModelMetadata
    from metatrain.utils.abc import ModelInterface

    class MyModel(ModelInterface):

        __supported_devices__ = ["cuda", "cpu"]
        __supported_dtypes__ = [torch.float64, torch.float32]
        __default_metadata__ = ModelMetadata(
            references = {"implementation": ["ref1"], "architecture": ["ref2"]}
        )

        def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
            ...

        ... # implementation of all the functions from ModelInterface


In addition to subclassing ``ModelInterface``, the model class should have the
following class attributes:

- ``__supported_devices__`` list of the suported torch devices for running the
  model;
- ``__supported_dtypes__`` list of the supported dtype for this model;

Both lists should be sorted in order of preference since ``metatrain`` will use
these to determine, based on the user request and machines' availability, the
optimal ``dtype`` and ``device`` for training.

- ``__default_metadata__`` can be used to provide references that will be
  stored in the exported model. The references are stored in a dictionary with
  keys ``implementation`` and ``architecture``. The ``implementation`` key
  should contain references to the software used in the implementation of the
  architecture, while the ``architecture`` key should contain references about
  the general architecture.


Trainer class (``trainer.py``)
------------------------------

.. autoclass:: metatrain.utils.abc.TrainerInterface
    :members:

Defining a new model can then be done as like this;

.. code-block:: python


    from metatrain.utils.abc import TrainerInterface

    class MyTrainer(TrainerInterface):

        def __init__(self, train_hypers):
            ...

        ... # implementation of all the functions from TrainerInterface

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

Default Hyperparamers (``default-hypers.yaml``)
-----------------------------------------------

The default hyperparameters for each architecture should be stored in a YAML
file ``default-hypers.yaml`` inside the architecture directory. Reasonable
default hypers are required to improve usability. The default hypers must follow
the structure

.. code-block:: yaml

    name: myarchitecture

    model:
        ...

    training:
        ...

``metatrain`` will parse this file and overwrite these default hypers with the
user-provided parameters and pass the merged ``model`` section as a Python
dictionary to the ``ModelInterface`` and the ``training`` section to the
``TrainerInterface``.

Finetuning
^^^^^^^^^^

If your architecture is supporting finetuning you have to add a ``finetune`` subsection
in the ``training`` section. The subsection must contain a ``read_from`` key that points
to the checkpoint file the finetuning is started from. Any additional hyperparameters
can be architecture specific.

.. code-block:: yaml

    training:
        finetune:
            read_from: path/to/checkpoint.ckpt
            # other architecture finetune hyperparameters

JSON schema (``schema-hypers.yaml``)
------------------------------------

To validate the user's input hyperparameters we are using `JSON schemas
<https://json-schema.org/>`_ stored in a schema file called
``schema-hypers.json``. For an :ref:`experimental architecture
<architecture-life-cycle>` it is not required to provide such a schema along
with its default hypers but it is highly recommended to reduce possible errors
of user input like typos in parameter names or wrong sections. If no
``schema-hypers.json`` is provided no validation is performed and user hypers
are passed to the architecture model and trainer as is.

To create such a schema you can try using `online tools
<https://jsonformatter.org>`_ that convert the ``default-hypers.yaml`` into a
JSON schema. Besides online tools, we also had success using ChatGPT/LLM for
this for conversion.

Documentation
-------------

Each new architecture should be added to ``metatrain``'s documentation. A short
page describing the architecture and its default hyperparameters will be
sufficient. You can take inspiration from existing architectures. The various
targets that the architecture can fit should be added to the table in the
"Fitting generic targets" section.
