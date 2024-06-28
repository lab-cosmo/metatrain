.. _dataset_conf:

Customize a Dataset Configuration
=================================
Overview
--------
The main task in setting up a training procedure with `metatrain` is to provide
files for training, validation, and testing datasets. Our system allows flexibility in
parsing data for training. Mandatory sections in the `options.yaml` file include:

- ``training_set``
- ``test_set``
- ``validation_set``

Each section can follow a similar system, with shorthand methods available to
simplify dataset definitions.

Minimal Configuration Example
-----------------------------
Below is the simplest form of these sections:

.. code-block:: yaml

    training_set: "dataset.xyz"
    test_set: 0.1
    validation_set: 0.1

This configuration parses all information from ``dataset.xyz``, with 20% of the training
set randomly selected for testing and validation (10% each).

Expanded Configuration Format
-----------------------------
The train script automatically expands the ``training_set`` section into the following
format, which is also valid for initial input:

.. code-block:: yaml

    training_set:
        systems:
            read_from: dataset.xyz
            file_format: .xyz
            length_unit: null
        targets:
            energy:
                quantity: energy
                read_from: dataset.xyz
                file_format: .xyz
                key: energy
                unit: null
                forces:
                    read_from: dataset.xyz
                    file_format: .xyz
                    key: forces
                stress:
                    read_from: dataset.xyz
                    file_format: .xyz
                    key: stress
                virial: false
    test_set: 0.1
    validation_set: 0.1

Understanding the YAML Block
----------------------------
The ``training_set`` is divided into sections ``systems`` and ``targets``:

Systems Section
^^^^^^^^^^^^^^^
Describes the system data like positions and cell information.

:param read_from: The file containing system data.
:param file_format: The file format, guessed from the suffix if ``null`` or not
    provided.
:param length_unit: The unit of lengths, optional but highly recommended for running
    simulations.

A single string in this section automatically expands, using the string as the
``read_from`` parameter.

.. note::

   `metatrain` does not convert units during training or evaluation. Units are
   only required if model should be used to run MD simulations.

Targets Section
^^^^^^^^^^^^^^^
Allows defining multiple target sections, each with a unique name.

- Commonly, a section named ``energy`` should be defined, which is essential for running
  molecular dynamics simulations. For the ``energy`` section gradients like `forces` and
  `stress` are enabled by default.
- Other target sections can also be defined, as long as they are prefixed by ``mtt::``.
  For example, ``mtt::free_energy``. In general, all targets that are not standard
  outputs of ``metatensor.torch.atomistic`` (see
  https://docs.metatensor.org/latest/atomistic/outputs.html) should be prefixed by
  ``mtt::``.

Target section parameters include:

:param quantity: The target's quantity (e.g., ``energy``, ``dipole``). Currently only
    ``energy`` is supported.
:param read_from: The file for target data, defaults to the ``systems.read_from``
  file if not provided.
:param file_format: The file format, guessed from the suffix if not provided.
:param key: The key for reading from the file, defaulting to the target section's name
  if not provided.
:param unit: The unit of the target, optional but highly recommended for running
    simulations.
:param forces: Gradient sections. See :ref:`gradient-section` for parameters.
:param stress: Gradient sections. See :ref:`gradient-section` for parameters.
:param virial: Gradient sections. See :ref:`gradient-section` for parameters.

A single string in a target section automatically expands, using the string as the
``read_from`` parameter.

.. _gradient-section:

Gradient Section
^^^^^^^^^^^^^^^^
Each gradient section (like ``forces`` or ``stress``) has similar parameters:

:param read_from: The file for gradient data.
:param file_format: The file format, guessed from the suffix if not provided.
:param key: The key for reading from the file.

A single string in a gradient section automatically expands, using the string as the
``read_from`` parameter.

Sections set to ``true`` or ``on`` automatically expand with default parameters. A
warning is raised if requisite data for a gradient is missing, but training proceeds
without them.

.. note::

   Unknown keys are ignored and not deleted in all sections during dataset parsing.

Multiple Datasets
-----------------
For some applications, it is required to provide more than one dataset for model
training. `metatrain` supports stacking several datasets together using the
``YAML`` list syntax, which consists of lines beginning at the same indentation level
starting with a ``"- "`` (a dash and a space)


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
element in ``training_set``

The ``length_unit`` has to be the same for each element of the list. If target section
names are the same for different elements of the list, their unit also has to be the
same. In the the example above the target section ``energy`` exists in both list
elements and therefore has the the same unit ``eV``. The target section ``free-energy``
only exists in the second element and its unit does not have to be the same as in the
first element of the list.

.. warning::

   Even though parsing several datasets is supported by the library, it may not
   work with every architecture. Check your :ref:`desired architecture
   <available-architectures>` if they **support multiple datasets**.

In the next tutorials we explain and show how to set some advanced global training
parameters.
