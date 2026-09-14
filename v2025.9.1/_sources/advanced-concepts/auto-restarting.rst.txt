Automatic restarting
====================

When restarting multiple times (for example, when training an expensive model
or running on an HPC cluster with short time limits), it is useful to be able
to train and restart multiple times with the same command.

In ``metatrain``, this functionality is provided via the ``--restart auto``
(or ``-c auto``) flag of ``mtt train``. This flag will automatically restart
the training from the last checkpoint, if one is found in the ``outputs/``
of the current directory. If no checkpoint is found, the training will start
from scratch.
