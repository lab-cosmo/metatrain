.. _architecture-life-cycle:

Life Cycle of an Architecture
=============================

.. TODO: Maybe add a flowchart later

Architectures in ``metatrain`` undergo different stages based on their
development/functionality level and maintenance status. We distinguish three distinct
stages: **experimental**, **stable**, and **deprecated**. Typically, an architecture
starts as experimental, advances to stable, and eventually becomes deprecated before
removal if maintenance is no longer feasible.

.. note::
    The development and maintenance of an architecture must be fully undertaken by the
    architecture's authors or maintainers. The core developers of ``metatrain``
    provide infrastructure and implementation support but are not responsible for the
    architecture's internal functionality or any issues that may arise therein.

Experimental Architectures
--------------------------

New architectures added to the library will initially be classified as experimental.
These architectures are stored in the ``experimental`` subdirectory within the
repository. To qualify as an experimental architecture, certain criteria must be met:

1. Capability to fit at least a single quantity and predict it, verified through CI
   tests.
2. Compatibility with JIT compilation using `TorchScript
   <https://pytorch.org/docs/stable/jit.html>`_.
3. Provision of reasonable default hyperparameters.
4. Minimal code quality, ensured by passing linting tests invoked with ``tox -e lint``.
5. A contact person designated as the maintainer.
6. All external dependencies must be pip-installable. While not required to be on PyPI,
   a public git repository or another public URL with a repository is acceptable.

For detailed instructions on adding a new architecture, refer to
:ref:`adding-new-architecture`.

Stable Architectures
--------------------

Transitioning from an experimental to a stable model requires additional criteria to be
satisfied:

1. Provision of regression prediction tests with a small (not exported) checkpoint file.
2. Comprehensive architecture documentation including a schema for verifying the
   architecture's hyperparameters.
3. If an architecture has external dependencies, all must be publicly available on PyPI.
4. Adherence to the standard output infrastructure of ``metatrain``, including
   logging and model save locations.

Deprecated Architectures
------------------------

An architecture will be deemed deprecated if its maintainer becomes irresponsive
any of its CI jobs fail. Such an architecture will be **removed after 6 months** unless
a new maintainer is found who can address the issues. If rectified within this 6-month
period, the model may revert to its previous stable or experimental status.
