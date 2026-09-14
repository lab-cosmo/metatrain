Auxiliary outputs
=================

These outputs, which are generally idenfified by the ``mtt::aux::`` prefix,
represent additional information that the model may provide. They are not
conventional trainable outputs, in the sense that they do not correspond to
training targets. For example, such quantities might be the internal
representation of the model, uncertainty estimates, or non-trainable quantities.

The following auxiliary outputs are currently supported by one or more
architectures in ``metatrain``:

- :ref:`mtt-aux-target-last-layer-features`: The representation of the model
  at the last layer, before the final linear transformation to produce
  target ``target``.
- :ref:`mtt-aux-target-uncertainty`: An uncertainty estimate for target
  ``target``, as computed by the LLPR module.
- :ref:`mtt-aux-target-ensemble`: An ensemble prediction for target
  ``target``, as computed by the LLPR module.


.. _mtt-aux-target-last-layer-features:

``mtt::aux::{target}_last_layer_features``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output contains the representation of the model at the last layer, before
the final linear transformation to produce target ``target``. If the model
produces multiple targets, the corresponding representations might be different.

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
    - ``"feature"``
    - the last-layer features have a single property dimension named
      ``"feature"``, with entries ranging from 0 to the number of features
      in the last layer.

Last-layer features are supported by the following architectures:

+-----------------+-----------------------+------------------+---------------------+
| :ref:`arch-pet` | :ref:`arch-soap_bpnn` | :ref:`arch-mace` | :ref:`arch-flashmd` |
+-----------------+-----------------------+------------------+---------------------+

.. _mtt-aux-target-uncertainty:

``mtt::aux::{target}_uncertainty``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output contains an uncertainty estimate for target ``target``, as computed
by the LLPR module. The metadata for this output is the same as for the
corresponding target, except that there is a single property named
``"uncertainty"``.

When ``target`` is ``energy``, this output is instead named
``energy_uncertainty``, creating the `corresponding standard metatomic
output <mta-energy-uncertainty_>`_.

.. _mta-energy-uncertainty: https://docs.metatensor.org/metatomic/latest/outputs/energy.html#energy-uncertainty

This output is only supported by the :ref:`LLPR architecture <arch-llpr>`.

.. _mtt-aux-target-ensemble:

``mtt::aux::{target}_ensemble``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output contains an ensemble prediction for target ``target``, as computed
by the LLPR module. The metadata for this output is the same as for the
corresponding target, except that there is a single property named
``"ensemble_prediction"``, with entries ranging from 0 to the number of
ensemble members.

When ``target`` is ``energy``, this output is instead named
``energy_ensemble``, creating the `corresponding standard metatomic
output <mta-energy-ensemble_>`_.

.. _mta-energy-ensemble: https://docs.metatensor.org/metatomic/latest/outputs/energy.html#energy-ensemble

This output is only supported by the :ref:`LLPR architecture <arch-llpr>`.
