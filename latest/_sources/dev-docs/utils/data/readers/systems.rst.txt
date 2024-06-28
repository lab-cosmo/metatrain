system Readers
#################

Parsers for obtaining information from systems. All readers return a :py:class:`list`
of :py:class:`metatensor.torch.atomistic.System`. The mapping which reader is used for
which file type is stored in

.. autodata:: metatrain.utils.data.readers.systems.SYSTEM_READERS

Implemented Readers
-------------------

.. autofunction:: metatrain.utils.data.readers.systems.read_systems_ase
