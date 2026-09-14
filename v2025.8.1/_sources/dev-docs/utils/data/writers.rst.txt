Target data Writers
===================

The main entry point for writing target information is

.. autofunction:: metatrain.utils.data.writers.write_predictions


Based on the provided filename the writer choses which child writer to use. The mapping
which reader is used for which file type is stored in

.. autodata:: metatrain.utils.data.writers.PREDICTIONS_WRITERS

Implemented Writers
-------------------

.. autofunction:: metatrain.utils.data.writers.write_xyz
.. autofunction:: metatrain.utils.data.writers.write_mts
