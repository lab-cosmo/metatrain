Readers
#######

Parsers for obtaining *system* and *target* information from files. Currently,
``metatrain`` support the following libraries for reading data

.. list-table::
  :header-rows: 1

  * - Library
    - Supported targets
    - Linked file formats
  * - ``ase``
    - system, energy, forces, stress, virials
    - ``.xyz``, ``.extxyz``


If the ``reader`` parameter is not set the library is determined from the file
extension. Overriding this behavior is in particular useful, if a file format is not
listed here but might be supported by a library.

Below the synopsis of the reader functions in details.

System and target data readers
==============================

The main entry point for reading system and target information are the reader functions

.. autofunction:: metatrain.utils.data.read_systems
.. autofunction:: metatrain.utils.data.read_targets

Target type specific readers
----------------------------

:func:`metatrain.utils.data.read_targets` uses sub-functions to parse supported
target properties like the ``energy`` or ``forces``. Currently we support reading the
following target properties via

.. autofunction:: metatrain.utils.data.read_energy
.. autofunction:: metatrain.utils.data.read_forces
.. autofunction:: metatrain.utils.data.read_virial
.. autofunction:: metatrain.utils.data.read_stress
