Training with atom-count-based batching
========================================

This section describes how to use ``max_atoms_per_batch`` (and the accompanying
``min_atoms_per_batch``) to control the number of atoms contained in each training
batch. This is useful when working with datasets that contain systems of highly
variable sizes and when computational resources are limited.

.. note::

   ``max_atoms_per_batch``/``min_atoms_per_batch`` limit the **total number of atoms
   across all systems in a batch**, not the number of systems themselves.


Motivation and best practices
------------------------------

Packing batches by atom count can help address several practical challenges:

1. **Memory safety**
   Prevent out-of-memory errors by limiting the maximum number of atoms per batch.

2. **Computational efficiency**
   Avoid inefficient batches with too few atoms by enforcing a minimum atom count.

3. **Training stability**
   Create more consistent computational load per batch when system sizes vary
   significantly.

Therefore:

- If you're here for **training efficiency**, a good starting point is to identify the
  average system size (by inspecting your dataset), multiply it by the batch size you
  would otherwise use, and set ``max_atoms_per_batch`` to that value (``min_atoms_per_batch``
  can usually be left at its default).

- If you're here because you ran **out of memory**, start with a large
  ``max_atoms_per_batch`` and lower it until training succeeds without running out of
  memory.


Setting ``max_atoms_per_batch``
--------------------------------

Below is an example configuration using the SOAP-BPNN architecture with atom-count
packing enabled:

.. code-block:: yaml

   device: cpu
   base_precision: 64
   seed: 42

   architecture:
     name: soap_bpnn
     training:
       num_epochs: 10
       learning_rate: 0.01
       max_atoms_per_batch: 100
       min_atoms_per_batch: 10

   training_set:
     systems: qm9_reduced_100.xyz
     targets:
       energy:
         key: U0
         unit: hartree

   validation_set: 0.1
   test_set: 0.0


Understanding ``max_atoms_per_batch``/``min_atoms_per_batch``
-----------------------------------------------------------------

When ``max_atoms_per_batch`` is set, structures are greedily accumulated into a batch,
in dataset order, until adding the next one would exceed ``max_atoms_per_batch``; the
batch is then closed and a new one started. This produces a variable number of
structures per batch, and ``batch_size`` is ignored when constructing training and
validation batches (it is still used internally for composition model and scaler
fitting).

- A single structure whose own atom count exceeds ``max_atoms_per_batch`` cannot be
  packed into any batch; it is skipped for the epoch, with a warning.
- ``min_atoms_per_batch`` (default ``0``, i.e. no minimum) discards any packed batch
  whose total atom count falls below it — this avoids spending a training step on an
  unusually small, inefficient batch.

.. note::

   Unlike a fixed ``batch_size``, the number of *structures* per batch varies with
   ``max_atoms_per_batch``: e.g. with ``max_atoms_per_batch: 100``, a batch could
   contain five 20-atom structures or one single 95-atom structure.
