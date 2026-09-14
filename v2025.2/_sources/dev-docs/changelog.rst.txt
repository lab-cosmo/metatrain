.. _changelog:

Changelog
=========

All notable changes to ``metatrain`` are documented here, following the `keep a
changelog <https://keepachangelog.com/en/1.1.0/>`_ format. This project follows
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

.. Possible sections for each release:

.. Fixed
.. #####

.. Added
.. #####

.. Changed
.. #######

.. Removed
.. #######

Version 2025.2 - 2025-03-11
---------------------------

Added
#####

- Implement a long-range featurizer as a utility for all models
- Speed up system preparation

Changed
#######

- Remove biases in SOAP-BPNN's linear layers

Fixed
#####

- Fix NanoPET multi-GPU error message
- Fix ``device`` for fixed composition weights


Version 2025.1 - 2025-02-20
---------------------------

Added
#####

- Support for Python 3.13 and ``ase`` >= 3.23

Fixed
#####

- Some irrelevant autograd warnings

Version 2025.0 - 2025-02-19
---------------------------

Added
#####

* First release outside of the lab
