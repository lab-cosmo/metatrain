Units
=====

``metatrain`` will always work with the units as provided by the user, and all logs will
be in the same units. In other terms, ``metatrain`` does not perform any unit
conversion. The only exception is the logging of energies in ``meV`` if the energies are
declared to be in ``eV``, for consistency with common practice and other codes.

Although not mandatory, the user is encouraged to specify the units of their datasets
in the input files, so that the logs can be more informative and, more importantly, in
order to make the resulting exported models usable in simulation engines (which instead
require the units to be specified) without unpleasant surprises.
