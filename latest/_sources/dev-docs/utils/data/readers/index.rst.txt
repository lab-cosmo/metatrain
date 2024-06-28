system and Target data Readers
=================================

The main entry point for reading system and target information are the two reader
functions

.. autofunction:: metatrain.utils.data.read_systems
.. autofunction:: metatrain.utils.data.read_targets

Target type specific readers
----------------------------

:func:`metatrain.utils.data.read_targets` uses sub-functions to parse supported
target properties like the `energy` or `forces`. Currently we support reading the
following target properties via

.. autofunction:: metatrain.utils.data.read_energy
.. autofunction:: metatrain.utils.data.read_forces
.. autofunction:: metatrain.utils.data.read_virial
.. autofunction:: metatrain.utils.data.read_stress

File type specific readers
--------------------------

Based on the provided `file_format` they chose which sub-reader they use. For details on
these refer to their documentation

.. toctree::
   :maxdepth: 1

   systems
   targets
