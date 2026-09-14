.. _train_yaml_config:

Training YAML Reference
***********************

Overview
========

``metatrain`` uses a YAML file to specify the parameters for model training, accessed
via ``mtt train options.yaml``. In this section, we provide a complete reference for the
parameters provided by the training YAML input. For a minimal example of a YAML input
file, suitable to start a first training, we refer the viewer to the sample YAML file in
the :ref:`Quickstart <label_quickstart>` section.

The YAML input file can be divided into five sections:

- :ref:`computational-parameters-section`
- :ref:`architecture-section`
- :ref:`loss-section`
- :ref:`data-section`
- :ref:`wandb-integration-section`

.. _computational-parameters-section:

Computational Parameters
========================

The computational parameters define the computational ``device``, ``base_precision`` and
``seed``. These parameters are optional.

.. code-block:: yaml

    device: cuda
    base_precision: 32
    seed: 0

.. container:: mtt-hypers-remove-classname

    .. autoattribute:: metatrain.share.base_hypers.BaseHypers.device
        :no-index:

    .. autoattribute:: metatrain.share.base_hypers.BaseHypers.base_precision
        :no-index:

    .. autoattribute:: metatrain.share.base_hypers.BaseHypers.seed
        :no-index:

.. _architecture-section:

Architecture
============

The next section of the YAML file would focus on options pertaining to the architecture.
The main skeleton is as follows:

.. code-block:: yaml

    architecture:
        name: architecture_name
        model:
            ...
        training:
            ...

The options for the ``architecture.model`` and ``architecture.training`` sections are
highly specific to the architecture used. You can refer to the :ref:`architecture
documentation <available-architectures>` page to find the options for your desired
architecture.

.. _loss-section:

Loss
====

A special parameter that you will find in the ``architecture.training`` section is
the one dedicated to the loss. There is a plethora of loss functions used in different
ML workflows, and you can refer to :ref:`the loss functions documentation <loss-functions>`
to understand the support of ``metatrain`` for all these different cases.

.. _data-section:

Data
====

The final section of the YAML file focuses on options regarding the data used in model
training. This secion can be broken down into three subsections:

- ``training_set``
- ``validation_set``
- ``test_set`` (optional)

The training set is the data that will be used for model training, the validation set is
the data that will be used to track the generalizability of the model during training
and is usually used to decide on the best model. The test set is only used after training
and it is used to evaluate the model's performance on an unseen dataset after training.
If not specified, no test set will be created.
Each subsection has the same parameter configuration. As an example, the configuration
of the training set is usually divided into three main sections:

.. code-block:: yaml

    training_set:
        systems:
            ...
        targets:
            ...
        extra_data:
            ...

with the three sections being:

- ``systems``: defines the molecular/crystal structures, which are the inputs to the model.
- ``targets``: defines the outputs to be predicted by the model.
- ``extra_data``: defines any additional data required by the loss function during
  training.

The validation and test set sections can also be fully specified in the same way as the
training set section, but they can also be simply a fraction of the training set. For
example:

.. code-block:: yaml

    training_set:
        ... # Training set specification
    validation_set: 0.1
    test_set: 0.2

will randomly select 10% of the training set for validation and 20% for testing.
The selected indices for the training, validation and test subset will be
available in the ``outputs`` directory.

.. note::

   If you don't need a test set, you can simply omit the ``test_set`` parameter entirely.

Systems
-------

The systems section can be defined as simply as:

.. code-block:: yaml

    training_set:
        systems: dataset.xyz
        ... # Rest of training set specification

which would instruct ``metatrain`` to read the systems from the file
``dataset.xyz`` using the default reader inferred from the file extension. If one
requires more control over the way the systems are read, one can provide a
specification that is defined by the following parameters:

.. autoclass:: metatrain.share.base_hypers.SystemsHypers
    :members:
    :undoc-members:
    :no-index:


As an example, the simple configuration that we saw previously is equivalent to:

.. code-block:: yaml

    training_set:
        systems:
            read_from: dataset.xyz
            reader: null
            length_unit: null
        ... # Rest of training set specification

Targets
-------

In the ``targets`` category, one can define any number of target sections, each with a
unique name, i.e. something like:

.. code-block:: yaml

    training_set:
        targets:
            energy:
                ... # Energy target specification
            mtt:dipole:
                ... # Dipole target specification
        ... # Rest of training set specification

The name of the target should either be a standard output of ``metatomic``
(see `metatomic outputs documentation <https://docs.metatensor.org/metatomic/latest/outputs/index.html>`_)
or begin with ``mtt::``, see :ref:`example below <datayaml-full-example>` for a fully fledged
version of a training set specification.

Each target can be specified with the following parameters:

.. autoclass:: metatrain.share.base_hypers.TargetHypers
    :members:
    :undoc-members:
    :no-index:

A single string in a target section automatically expands, using the string as the
``read_from`` parameter.

.. _gradient-subsection:

Gradient Subsection
^^^^^^^^^^^^^^^^^^^

Each gradient subsection (like ``forces`` or ``stress``) has similar parameters:

.. autoclass:: metatrain.share.base_hypers.GradientDict
    :members:
    :undoc-members:
    :no-index:

A single string in a gradient section automatically expands, using the string as the
``read_from`` parameter.

Sections set to ``true`` or ``on`` automatically expand with default parameters. A
warning is raised if requisite data for a gradient is missing, but training proceeds
without them. For instance,

.. code-block:: yaml

        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                reader: ase
                key: energy
                unit: null
                forces:
                    read_from: dataset.xyz
                    reader: ase
                    key: forces
                stress:
                    read_from: dataset.xyz
                    reader: ase
                    key: stress

can be condensed into

.. code-block:: yaml

        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                reader: ase
                key: energy
                unit: null
                forces: on
                stress: on

.. note::

   Unknown keys are ignored and not deleted in all sections during dataset parsing.

Datasets requiring additional data
----------------------------------

Some targets require additional data to be passed to the loss function for training. In
the example above, we included the mask for the density of states, which defines the
regions of the DOS that are well-defined based on the eigenvalues of the underlying
electronic structure calculation. This is important when the DOS is computed over a
finite energy range, as the DOS near the edges of this range may be inaccurate due to
the lack of states computed beyond this range. ``metatrain`` supports passing additional
data in the ``options.yaml`` file. This can be seen in the ``extra_data`` section of the
full example above.

As another example, training a model to predict the polarization for extended systems
under periodic boundary conditions might require the quantum of polarization to be
provided for each system in the dataset. For this, you can add the following section to
your ``options.yaml`` file:

.. code-block:: yaml

    training_set:
        systems:
            read_from: dataset_0.xyz
            length_unit: angstrom
        targets:
            mtt::polarization:
                read_from: polarization.mts
        extra_data:
            polarization_quantum:
                read_from: polarization_quantum.mts

.. warning::

   While the ``extra_data`` section can always be present, it will typically be ignored
   unless using specific loss functions. If the loss function you picked does not
   support the extra data, it will be ignored.

The ``extra_data`` section supports the same parameters as the target sections. In this
case, we have also read the targets and extra data from files other than the systems
file.

.. _datayaml-full-example:

Full data example
-----------------

Here is a full fledged example of a training set specification, in this case for
learning the electronic density of states (DOS) along with forces and stresses:

.. code-block:: yaml

    training_set:
        systems:
            read_from: dataset.xyz
            reader: ase
            length_unit: null
        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                reader: ase
                key: energy
                unit: null
                per_atom: True
                type: scalar
                num_subtargets: 1
                forces:
                    read_from: dataset.xyz
                    reader: ase
                    key: forces
                stress:
                    read_from: dataset.xyz
                    reader: ase
                    key: stress
            non_conservative_forces:
                quantity: null
                read_from: nonconservative_force.mts
                reader: metatensor
                key: forces
                unit: null
                per_atom: True
                type:
                    cartesian:
                        rank: 1
                num_subtargets: 1
            mtt::dos:
                quantity: null
                read_from: DOS.mts
                reader: metatensor
                key: dos
                unit: null
                per_atom: False
                type: scalar
                num_subtargets: 4000
        extra_data:
            mtt::dos_mask:
                quantity: null
                read_from: dataset.xyz
                reader: ase
                key: dos_mask
                unit: null
                per_atom: False
                type: scalar
                num_subtargets: 4000

Using Multiple Files for Training
---------------------------------

For some applications, it is simpler to provide more than one dataset for model
training. ``metatrain`` supports stacking several datasets together using the ``YAML``
list syntax, which consists of lines beginning at the same indentation level starting
with a ``"- "`` (a dash and a space)

.. code-block:: yaml

    training_set:
        - systems:
              read_from: dataset_0.xyz
              length_unit: angstrom
          targets:
              energy:
                  quantity: energy
                  key: my_energy_label0
                  unit: eV
        - systems:
              read_from: dataset_1.xyz
              length_unit: angstrom
          targets:
              energy:
                  quantity: energy
                  key: my_energy_label1
                  unit: eV
              free-energy:
                  quantity: energy
                  key: my_free_energy
                  unit: hartree
    test_set: 0.1
    validation_set: 0.1

The required test and validation splits are performed consistently for each element
element in ``training_set`` .

The ``length_unit`` has to be the same for each element of the list. If target section
names are the same for different elements of the list, their unit also has to be the
same. In the the example above the target section ``energy`` exists in both list
elements and therefore has the the same unit ``eV``. The target section ``free-energy``
only exists in the second element and its unit does not have to be the same as in the
first element of the list.

Typically the global atomic types the the model is defined for are inferred from the
training and validation datasets. Sometimes, due to shuffling of datasets with low
representation of some types, these datasets may not contain all atomic types that you
want to use in your model. To explicitly control the atomic types the model is defined
for, specify the ``atomic_types`` key in the ``architecture`` section of the options
file:

.. code-block:: yaml

    architecture:
        name: pet
        model:
            cutoff: 5.0
        training:
            batch_size: 32
            epochs: 100
        atomic_types: [1, 6, 7, 8, 16]  # i.e. for H, C, N, O, S

.. warning::

   Even though parsing several datasets is supported by the library, it may not work
   with every architecture. Check your :ref:`desired architecture
   <available-architectures>` if they **support multiple datasets**.

.. _wandb-integration-section:

WandB Integration
=================

Optional section dealing with integration with `Weights and Biases (wandb)
<https://wandb.ai>`_ logging. Leaving this section blank will simply disable wandb
integration. The parameters for this section is the same as that in `wandb.init
<https://docs.wandb.ai/models/ref/python/functions/init>`_. Here we provide a minimal example for the
YAML input

.. code-block:: yaml

    wandb:
        project: my_project
        name: my_run_name
        tags:
          - tag1
          - tag2
        notes: This is a test run

All parameters of your ``options.yaml`` file will be automatically added to the wandb
run so you don't have to set the ``config`` parameter.

.. important::

    You need to install wandb with ``pip install wandb`` if you want to use this logger.
    **Before** running also set up your credentials with ``wandb login`` from the
    command line. See `wandb login documentation
    <https://docs.wandb.ai/ref/cli/wandb-login>`_ for details on the setup.
