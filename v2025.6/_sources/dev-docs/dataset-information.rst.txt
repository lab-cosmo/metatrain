Dataset Information
===================

When working with ``metatrain``, you will most likely need to interact with some core
classes which are responsible for storing some information about datasets. All these
classes belong to the ``metatrain.utils.data`` module which can be found in the
:ref:`data` section of the developer documentation.

These classes are:

- :py:class:`metatrain.utils.data.DatasetInfo`: This class is responsible for storing
  information about a dataset. It contains the length unit used in the dataset, the
  atomic types present, as well as information about the dataset's targets as a
  ``Dict[str, TargetInfo]`` object. The keys of this dictionary are the names of the
  targets in the datasets (e.g., ``energy``, ``mtt::dipole``, etc.).

- :py:class:`metatrain.utils.data.TargetInfo`: This class is responsible for storing
    information about a target in a dataset. It contains the target's physical quantity,
    the unit in which the target is expressed, and the ``layout`` of the target. The
    ``layout`` is ``TensorMap`` object with zero samples which is used to exemplify
    the metadata of each target.

At the moment, only three types of layouts are supported:

- scalar: This type of layout is used when the target is a scalar quantity. The
    ``layout`` ``TensorMap`` object corresponding to a scalar must have one
    ``TensorBlock`` and no ``components``.
- Cartesian tensor: This type of layout is used when the target is a Cartesian tensor.
    The ``layout`` ``TensorMap`` object corresponding to a Cartesian tensor must have
    one ``TensorBlock`` and as many ``components`` as the tensor's rank. These
    components are named ``xyz`` for a tensor of rank 1 and ``xyz_1``, ``xyz_2``, and
    so on for higher ranks.
- Spherical tensor: This type of layout is used when the target is a spherical tensor.
    The ``layout`` ``TensorMap`` object corresponding to a spherical tensor can have
    multiple blocks corresponding to different irreps (irreducible representations) of
    the target. The ``keys`` of the ``TensorMap`` object must have the ``o3_lambda``
    and ``o3_sigma`` names, and each ``TensorBlock`` must have a single component named
    ``o3_mu``.
