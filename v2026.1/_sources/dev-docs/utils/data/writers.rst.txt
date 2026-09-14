Target data Writers
===================

The main entry point for writing target information is

.. autofunction:: metatrain.utils.data.writers.get_writer


Based on the provided filename the writer choses which child writer to use. The mapping
which writer is used for which file type is stored in

.. autodata:: metatrain.utils.data.writers.PREDICTIONS_WRITERS

Implemented Writers
-------------------

Writer Abstract Class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: metatrain.utils.data.writers.Writer
    :members:
    :undoc-members:
    :show-inheritance:


Available Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^

The available implementations listed below represent concrete writers that inherit from
the ``Writer`` abstract class.

.. autoclass:: metatrain.utils.data.writers.ASEWriter
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: metatrain.utils.data.writers.DiskDatasetWriter
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: metatrain.utils.data.writers.MetatensorWriter
    :members:
    :undoc-members:
    :show-inheritance:
