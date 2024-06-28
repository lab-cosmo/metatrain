Target data Readers
###################

Parsers for obtaining target informations from target files. All readers return a
:py:class:`metatensor.torch.TensorBlock`. Currently we support the following target
properties

- :ref:`energy`
- :ref:`forces`
- :ref:`stress`
- :ref:`virial`

The mapping which reader is used for which file type is stored in a dictionary.

.. _energy:

Energy
======

.. autodata:: metatrain.utils.data.readers.targets.ENERGY_READERS

Implemented Readers
-------------------

.. autofunction:: metatrain.utils.data.readers.targets.read_energy_ase


.. _forces:

Forces
======

.. autodata:: metatrain.utils.data.readers.targets.FORCES_READERS

Implemented Readers
-------------------

.. autofunction:: metatrain.utils.data.readers.targets.read_forces_ase

.. _stress:

Stress
======

.. autodata:: metatrain.utils.data.readers.targets.STRESS_READERS

Implemented Readers
-------------------

.. autofunction:: metatrain.utils.data.readers.targets.read_stress_ase

.. _virial:

Virial
======

.. autodata:: metatrain.utils.data.readers.targets.VIRIAL_READERS

Implemented Readers
-------------------

.. autofunction:: metatrain.utils.data.readers.targets.read_virial_ase
