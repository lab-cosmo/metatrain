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
  * - ``metatensor``
    - system, energy, forces, stress, virials
    - ``.mts``


If the ``reader`` parameter is not set, the library is determined from the file
extension. Overriding this behavior is in particular useful if a file format is not
listed here but might be supported by a library.

Below the synopsis of the reader functions in details.

System and target data readers
==============================

The main entry point for reading system and target information are the reader functions.

.. autofunction:: metatrain.utils.data.read_systems
.. autofunction:: metatrain.utils.data.read_targets

These functions dispatch the reading of the system and target information to the
appropriate readers, based on the file extension or the user-provided library.

In addition, the read_targets function uses the user-provided information about the
targets to call the appropriate target reader function (for energy targets or generic
targets).

ASE
---

This section describes the parsers for the ASE library.

.. autofunction:: metatrain.utils.data.readers.ase.read
.. autofunction:: metatrain.utils.data.readers.ase.read_systems
.. autofunction:: metatrain.utils.data.readers.ase.read_energy
.. autofunction:: metatrain.utils.data.readers.ase.read_generic

It should be noted that :func:`metatrain.utils.data.readers.ase.read_energy` currently
uses sub-functions to parse the energy and its gradients like ``forces``, ``virial``
and ``stress``.

Metatensor
----------

This section describes the parsers for the ``metatensor`` library. As the systems and/or
targets are already stored in the ``metatensor`` format, these reader functions mainly
perform checks and return the data.

.. autofunction:: metatrain.utils.data.readers.metatensor.read_systems
.. autofunction:: metatrain.utils.data.readers.metatensor.read_energy
.. autofunction:: metatrain.utils.data.readers.metatensor.read_generic
