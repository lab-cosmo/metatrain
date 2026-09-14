Auxiliary outputs
=================

These outputs, which are generally idenfified by the ``mtt::aux::`` prefix,
represent additional information that the model may provide. They are not
conventional trainable outputs, in the sense that they do not correspond to
training targets. For example, such quantities might be the internal
representation of the model, uncertainty estimates, or non-trainable
quantities.

The following auxiliary outputs are currently supported
by one or more architectures in the library:

- ``mtt::aux::{target}_last_layer_features``: The representation
   of the model at the last layer, before the final linear transformation
   to produce target ``target``. If the model produces multiple targets,
   the corresponding representations might be different. This output
   differs from the ``features`` output which is the same for all targets
   of a model.
- ``features``: A common representation of the model for all targets.
  Generally, this will correspond to the last representation before the
  decoder(s), or heads, of the model.
- ``mtt::aux::{target}_uncertainty`` and ``mtt::aux::{target}_ensemble``:
  Auxiliary outputs related to uncertainty estimation. For the energy
  output, ``mtt::aux::energy_ensemble`` is instead named
  ``energy_uncertainty``. For the moment, these are only accessible
  through the LLPR module, which itself requires the use of the
  ``mtt::aux::{target}_last_layer_features`` output.


The following table shows the architectures that support each of the
auxiliary outputs:

+--------------------------------------------+-----------+------+-----+---------+
| Auxiliary output                           | SOAP-BPNN | PET  | GAP | NanoPET |
+--------------------------------------------+-----------+------+-----+---------+
| ``mtt::aux::{target}_last_layer_features`` |    Yes    | Yes  | No  |   Yes   |
+--------------------------------------------+-----------+------+-----+---------+
| ``features``                               |    Yes    | Yes  | No  |   Yes   |
+--------------------------------------------+-----------+------+-----+---------+

The following tables show the metadata that will be provided for each of the
auxiliary outputs:

mtt::aux::{target}_last_layer_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
      ``"properties"``, with entries ranging from 0 to the number of features
      in the last layer.

features
^^^^^^^^

See the
`feature output <https://docs.metatensor.org/latest/atomistic/outputs/features.html>`_
in ``metatensor.torch.atomistic``.
