.. _scale-targets:

Target scaling
==============

When ``scale_targets: true`` is set in the ``training`` section of the options file,
``metatrain`` automatically fits a :py:class:`~metatrain.utils.scaler.Scaler` to the
training data and uses it to normalise targets before computing the loss.
This page explains what scales are computed, how they are derived from the training
data, and how they are applied during training and inference.


Why scale targets?
------------------

Scaling serves two distinct purposes that are worth keeping separate:

1. **Global per-target scaling.** Normalising each target to a comparable magnitude
   so that multi-target training is balanced. Without this, a target with values in the
   thousands (e.g. total energy in eV for a large cell) would dominate the loss over a
   target with values near zero (e.g. dipole components).

2. **Per-property scaling.** Capturing the relative difference in magnitude between the
   individual properties of a multi-property target. For example, the :math:`\lambda=0` and
   :math:`\lambda=2` irreps of a polarizability tensor, or the individual radial channels of
   an electron density decomposition, naturally have very different scales that the
   model should not need to learn from scratch.


While scales are computed over different indices depending on specifically what the
scale is being computed on (explained below), in all cases the un-centered standard
deviations, without Bessel's correction, is computed. This is because the ``Scaler``
module is usually paired with a ``CompositionModel`` module that removes the mean of the
targets, and scales must be uncentered in order to preserve the equivariance of the
targets.


Single-property targets
-----------------------

A target is considered *single-property* if it has exactly one block and that block has
exactly one property.  Energy, forces, and stress are all in this category.

Before scales are computed, it is assumed that the composition contributions have first
been subtracted, and that per-structure targets have been normalized by their number of
atoms. Then, the per-target scale is computed as the root-mean-square (RMS) of these
values, pooled across all samples. Consider a generic single-block target with
components :math:`c`.

For **per-structure targets** (i.e. the total energy) with samples indexed by structure
index, :math:`A`, the per-target scales are computed as:

.. math::

   \sigma = \sqrt{
       \frac{\displaystyle\sum_{A c}
             y_{A c}^{2}}
            {\displaystyle\sum_{A c} 1}
   },

For **per-atom targets** (i.e. atomic forces) with samples indexed by system and atom
indices :math:`A` and :math:`i`, the scales are conditioned on atomic type, meaning only
samples that correspond to the given atomic type are pooled when computing the scale,
:math:`\sigma_Z`:

.. math::

   \sigma_Z = \sqrt{
       \frac{\displaystyle\sum_{A c}
             \delta_{Z_i Z} \ y_{A i c}^{2}}
            {\displaystyle\sum_{A c} \delta_{Z_i Z} \ 1}
   },

For single-property targets, the per-target scale is equivalent to the full
scale. In the relevant base class, computed scale(s) :math:`\sigma_{(Z)}` are
stored in ``BaseScaler.scales`` and equivalently in
``BaseScaler.per_target_scales``. ``BaseScaler.per_property_scales``, which
track the relative scales of different properties within each target, are all by
definition set to ``1.0``.


Multi-property targets
----------------------

A target is considered *multi-property* if it has more than one block, or if it is a
single block with more than one property.  Examples include polarizability tensors
(multiple blocks corresponding to irreps indexed by :math:`\lambda`), electron density
decompositions (multiple blocks corresponding to atom-typed irreps, each with multiple
radial channels per block).

Scaling for these targets proceeds in two stages.

Step 1 — per-target scales
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-target scale(s) :math:`\sigma_{(Z)}` are computed in the same way as for
single-property targets: compositional contributions are removed and atom-normalization
is performed (for per-structure targets), then scales are computed by this time pooling
across all blocks, labelled by a composite index :math:`\alpha`, and all properties,
:math:`p`. For per-structure targets:

.. math::

   \sigma = \sqrt{
       \frac{\displaystyle \sum_{\alpha A c p}
             y_{\alpha A c p}^{2}}
            {\displaystyle\sum_{\alpha A c p} 1}
   },

and for per-atom targets:

.. math::

   \sigma_Z = \sqrt{
       \frac{\displaystyle\sum_{\alpha A c p}
             \delta_{Z_i, Z} \ y_{\alpha A i c p}^{2}}
            {\displaystyle\sum_{\alpha A c p} \delta_{Z_i Z} \ 1}
   },

and the result is stored in ``BaseScaler.per_target_scales``.


Step 2 — per-property scales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the per-target scale(s) are computed, these are removed from the targets (after
composition contribution subtracted and atom-normalization) per-property scales are
computed for each block and each property independently.  For per-structure targets, the
per-property scale is:

.. math::

   \sigma_{\alpha p} = \frac{1}{\sigma}
                  \sqrt{
                      \frac{\displaystyle\sum_{A c}
                            y_{\alpha A c p}^{2}}
                           {\displaystyle\sum_{\alpha A c} 1}
                  },

and for per-atom targets:

.. math::

   \sigma_{\alpha p Z} = \frac{1}{\sigma_Z}
                  \sqrt{
                      \frac{\displaystyle\sum_{A i c}
                            \delta_{Z_i Z} y_{\alpha A i c p}^{2}}
                           {\displaystyle\sum_{\alpha A i c p} \delta_{Z_i Z} 1}
                  },


Dividing by :math:`\sigma_{(Z)}` means that :math:`\sigma_{(Z)} \cdot \sigma_{b p (Z)}`
equals the raw per-property RMS — in other words, the per-property scale is a
*correction factor relative to the global scale*, and is close to 1 for blocks whose
magnitude matches the target average.  These are stored in
``BaseScaler.per_property_scales``, and are only defined (and computed) for
multi-property targets.

Full scales
^^^^^^^^^^^

The full scale for block :math:`\alpha`, property :math:`p` is the product:

.. math::

   \sigma_{\alpha p (Z)}^{\mathrm{full}}
   = \sigma_{(Z)} \cdot \sigma_{\alpha p (Z)}

which is simply the per-property RMS of the atom-normalised values, exactly as was
computed in previous versions of ``metatrain``.  These are stored in
``BaseScaler.scales``.


How scales are applied
----------------------

The per-target and per-property scales are used differently depending on the stage
of the training loop.

**During training: dataloader / loss computation.**  All targets in a batch are divided
by the per-target scales.  The model therefore learns to predict per-target-scaled
quantities.  After the model makes its predictions:

- For *single-property* targets: the loss is computed directly between the
  (per-target-scaled) predictions and the (per-target-scaled) targets.
- For *multi-property* targets: the predictions are first multiplied by the
  per-property scales (i.e. un-scaled back towards physical units relative to the
  per-target scale) before the loss is computed against the per-target-scaled targets.
  This allows the model to focus on learning the physically meaningful relative
  differences between properties, while the loss is still computed in a normalised
  space.

**Metric reporting.**  When computing training and validation MAEs/RMSEs that are
reported to the user, both predictions and targets are un-scaled by the full per-target
scales so that the reported metrics are in physical units.

**Inference.**  At inference time, model predictions are multiplied by the full scales
(``BaseScaler.scales``, which equals ``per_target_scales × per_property_scales``)
before the additive model contribution is added and the final predictions are returned.
This ensures that predictions are always in physical units regardless of whether the
target is single- or multi-property.

