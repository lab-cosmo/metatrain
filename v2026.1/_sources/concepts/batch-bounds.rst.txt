Training with Batch Bounds
==========================

This section describes how to use batch bounds to control the number of atoms
contained in each training batch. Batch bounds are useful when working with datasets
that contain systems of highly variable sizes and when computational resources are
limited.

.. note::

   Batch bounds limit the **total number of atoms across all systems in a batch**, not
   the number of systems themselves.


Motivation and best practices
-----------------------------

Using batch atom bounds can help address several practical challenges:

1. **Memory safety**
   Prevent out-of-memory errors by limiting the maximum number of atoms per batch.

2. **Computational efficiency**
   Avoid inefficient batches with too few atoms by enforcing a minimum atom count.

3. **Training stability**
   Create more consistent batch sizes when system sizes vary significantly.

Therefore:

- If you're here for **training efficiency**, a good starting point is to identify the
  average system size (by inspecting your dataset), multiply it by the batch size you
  want to set, and set the minimum and maximum bounds to 50% and 150% of that value,
  respectively.

- If you're here because you ran **out of memory**, you can ignore the minimum bound and
  only set the maximum bound to a value that fits your hardware. For example, start with
  a large threshold (e.g. ``[None, 1000]``) and lower the maximum bound until training
  succeeds without running out of memory.


Setting batch bounds
--------------------

Below is an example configuration using the SOAP-BPNN architecture with batch bounds
enabled:

.. code-block:: yaml

   device: cpu
   base_precision: 64
   seed: 42

   architecture:
     name: soap_bpnn
     training:
       batch_size: 5
       num_epochs: 10
       learning_rate: 0.01
       batch_atom_bounds: [10, 100]  # minimum 10 atoms, maximum 100 atoms per batch

   training_set:
     systems: qm9_reduced_100.xyz
     targets:
       energy:
         key: U0
         unit: hartree

   validation_set: 0.1
   test_set: 0.0


Understanding ``batch_atom_bounds``
-----------------------------------

The ``batch_atom_bounds`` option specifies a minimum and maximum number of atoms allowed
per batch, according to the syntax ``batch_atom_bounds: [min_atoms, max_atoms]``.
Batches with a total atom count outside these bounds are skipped during training.

The following types of configurations are supported:

- ``[10, None]``
  Enforces a minimum number of atoms per batch.

- ``[None, 100]``
  Enforces a maximum number of atoms per batch.

- ``[None, None]``
  Disables batch bounds (default behavior).

.. note::

   For example, with ``batch_size: 5``, if each system contains 20 atoms, the batch will
   contain 100 atoms in total.
