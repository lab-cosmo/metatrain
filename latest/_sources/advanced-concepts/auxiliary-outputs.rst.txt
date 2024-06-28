Auxiliary outputs
=================

These outputs, which are idenfified by the ``mtt::aux::`` prefix,
represent additional information that the model may provide. They are not
conventional trainable outputs, and they often correspond to internal
information that the model is capable of providing, such as its internal
representation.

The following auxiliary outputs that are currently supported
by one or more architectures in the library:

- ``mtt::aux::last_layer_features``: The internal representation
   of the model at the last layer, before the final linear transformation.

The following table shows the architectures that support each of the
auxiliary outputs:

+------------------------------------------+-----------+------------------+-----+
| Auxiliary output                         | SOAP-BPNN | Alchemical Model | PET |
+------------------------------------------+-----------+------------------+-----+
| ``mtt::aux::last_layer_features``        | Yes       |       No         | No  |
+------------------------------------------+-----------+------------------+-----+

The following tables show the metadata that is expected for each of the
auxiliary outputs:

mtt::aux::last_layer_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Metadata for last-layer features
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"`` if invariant or non-symmetrized, ``["o3_lambda", "o3_sigma"]`` if
      equivariant
    - if invariant or non-symmetrized, a single dimension named ``"_"``, with a
      single entry set to ``0``, or, if equivariant, two dimensions named
      ``"o3_lambda"`` and ``"o3_sigma"`` with the corresponding O3 equivariant
      indices.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - if a ``per_atom`` output is requested, the sample names will be
      ``["system", "atom"]``, otherwise they will be ``["system"]``.

      ``"system"`` ranges from 0 to the number of systems given as input to
      the model. ``"atom"`` ranges between 0 and the number of
      atoms/particles in the corresponding system.

  * - components
    - No components if invariant or non-symmetrized, ``["o3_mu"]`` if equivariant
    - Nothing if invariant or non-symmetrized, the O3 equivariant ``mu`` number if
      equivariant.

  * - properties
    - ``"properties"``
    - the last-layer features have a single property dimension named
      ``"property"``, with entries ranging from 0 to the number of features
      in the last layer.
