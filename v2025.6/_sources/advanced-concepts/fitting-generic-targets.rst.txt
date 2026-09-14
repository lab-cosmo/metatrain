.. _fitting-generic-targets:

Fitting generic targets
=======================

Besides energy-like targets, the library also supports reading (and training on)
more generic targets.

Support for generic targets
---------------------------

Not all architectures can train on all types of target. Here you can find the
capabilities of the architectures in metatrain.

.. list-table:: Sample Table
   :header-rows: 1

   * - Target type
     - Energy and its gradients
     - Scalars
     - Spherical tensors
     - Cartesian tensors
   * - SOAP-BPNN
     - Energy, forces, stress/virial
     - Yes
     - Yes
     - No
   * - GAP
     - Energy, forces
     - No
     - No
     - No
   * - PET
     - Energy, forces, stress/virial
     - Yes
     - Yes
     - Only with ``rank=1`` (vectors) and ``rank=2`` (2D tensors)
   * - NanoPET
     - Energy, forces, stress/virial
     - Yes
     - Yes
     - Only with ``rank=1`` (vectors) and ``rank=2`` (2D tensors)


Preparing generic targets for reading by metatrain
--------------------------------------------------

Only a few steps are required to fit arbitrary targets in metatrain.

Input file
##########

In order to read a generic target, you will have to specify its layout in the input
file. Suppose you want to learn a target named ``mtt::my_target``, which is
represented as a set of 10 independent per-atom 3D Cartesian vector (we need to
learn 3x10 values for each atom). The ``target`` section in the input file
should look
like this:

.. code-block:: yaml

    targets:
      mtt::my_target:
        read_from: dataset.xyz
        key: my_target
        quantity: ""
        unit: ""
        per_atom: True
        type:
          cartesian:
            rank: 1
        num_subtargets: 10

The crucial fields here are:

- ``per_atom``: This field should be set to ``True`` if the target is a per-atom
    property. Otherwise, it should be set to ``False``.
- ``type``: This field specifies the type of the target. In this case, the target is
    a Cartesian vector. The ``rank`` field specifies the rank of the target. For
    Cartesian vectors, the rank is 1. Other possibilities for the ``type`` are
    ``scalar`` (for a scalar target) and ``spherical`` (for a spherical tensor).
- ``num_subtargets``: This field specifies the number of sub-targets that need to be
    learned as part of this target. They are treated as entirely equivalent by models in
    metatrain and will often be represented as outputs of the same neural network layer.
    A common use case for this field is when you are learning a discretization of a
    continuous target, such as the grid points of a band structure. In the example
    above, there are 10 sub-targets. In ``metatensor``, these correspond to the number
    of ``properties`` of the target.

A few more words should be spent on ``spherical`` targets. These should be made of a
certain number of irreducible spherical tensors. For example, if you are learning a
property that can be decomposed into two proper spherical tensors with L=0 and L=2,
the target section should would look like this:

.. code-block:: yaml

    targets:
      mtt::my_target:
        quantity: ""
        read_from: dataset.xyz
        key: energy
        unit: ""
        per_atom: True
        type:
          spherical:
            irreps:
                - {o3_lambda: 0, o3_sigma: 1}
                - {o3_lambda: 2, o3_sigma: 1}
        num_subtargets: 10

where ``o3_lambda`` specifies the L value of the spherical tensor and ``o3_sigma`` its
parity with respect to inversion (1 for proper tensors, -1 for pseudo-tensors).

Preparing your targets -- ASE
#############################

If you are using the ASE readers to read your targets, you will have to save them
either in the ``.info`` (if the target is per structure, i.e. not per atom) or in the
``.arrays`` (if the target is per atom) attributes of the ASE atoms object. Then you can
dump the atoms object to a file using ``ase.io.write``.

The ASE reader will automatically try to reshape the target data to the format expected
given the target section in the input file. In case your target data is invalid, an
error will be raised.

Reading targets with more than one spherical tensor is not supported by the ASE reader.
In that case, you should use the metatensor reader.

Preparing your targets -- metatensor
####################################

If you are using the metatensor readers to read your targets, you will have to save them
as a ``metatensor.torch.TensorMap`` object with ``metatensor.torch.TensorMap.save()``
into a file with the ``.mts`` extension.

The metatensor reader will verify that the target data in the input files corresponds to
the metadata in the provided ``TensorMap`` objects. In case of a mismatch, errors will
be raised.

In particular:

- if the target is per atom, the samples should have the [``system``, ``atom``] names,
  otherwise the [``system``] name. In order to produce correct results, per-atom targets
  should be prepared in the same order as the atoms in the system. This means that you
  should use e.g. ``metatensor.torch.sort`` to sort the atoms in your target.
- if the target is a ``scalar``, only one ``TensorBlock`` should be present, the keys
  of the ``TensorMap`` should be a ``Labels.single()`` object, and there should be no
  components.
- if the target is a ``cartesian`` tensor, only one ``TensorBlock`` should be present,
  the keys of the ``TensorMap`` should be a ``Labels.single()`` object, and there should
  be one components, with names [``xyz``] for a rank-1 tensor,
  [``xyz_1``, ``xyz_2``, etc.] for higher rank tensors.
- if the target is a ``spherical`` tensor, the ``TensorMap`` can contain multiple
  ``TensorBlock``, each corresponding to one irreducible spherical tensor. The keys of
  the ``TensorMap`` should have the ``o3_lambda`` and ``o3_sigma`` names, corresponding
  to the values provided in the input file, and each ``TensorBlock`` should be one
  component label, with name ``o3_mu`` and values going from -L to L.
