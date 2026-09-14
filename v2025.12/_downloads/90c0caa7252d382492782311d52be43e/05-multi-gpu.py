"""
Multi-GPU training
==================

``metatrain`` supports training a model with several GPUs, which can accelerate the
training, especially when the training dataset is large / there are many training
epochs. This feature is enabled by the :py:mod:`torch.distributed` module, and thus can
do multiprocess parallelism across several nodes.

In multi-GPU training, every batch of samples is split into smaller mini-batches and the
computation is run for each of the smaller mini-batches in parallel on different GPUs.
The different gradients obtained on each device are then summed. This approach allows
the user to reduce the time it takes to train models.

To know if the model supports multi-GPU training, please check
:ref:`available-architectures` and see if the default hyperparameters have the
``distributed`` option.

Input file
----------

To do this, you only need to switch on the ``distributed`` option in the ``.yaml`` file
for the training. Let's take the
:ref:`sphx_glr_generated_examples_0-beginner_03-train_from_scratch.py` example and
adjust the ``options.yaml`` file.

To know if the model supports multi-GPU training, please check
:ref:`available-architectures` and see if the default hyperparameters have the
``distributed`` option.

Input file
----------

To do this, you only need to switch on the ``distributed`` option in the ``.yaml`` file
for the training. Let's take the
:ref:`sphx_glr_generated_examples_0-beginner_03-train_from_scratch.py` example and
adjust the ``options.yaml`` file.

.. literalinclude:: options-distributed.yaml
   :language: yaml
   :linenos:

Slurm script
------------

Below is an example Slurm script for submitting the job. Please be aware that the actual
configurations vary from clusters to clusters, so you have to modify it. Different
scheduler will require similar options. ``metatrain`` will automatically use all the
GPUs that you have asked for. You should make a single GPU visible for each process
(setting ``--gpus-per-node`` equal to the number of GPUs, or setting
``--gpus-per-task=1``, depending on your cluster configuration).

.. code-block:: bash

    #!/bin/bash
    #SBATCH --nodes 1
    #SBATCH --ntasks 2  # must equal to the number of GPUs
    #SBATCH --ntasks-per-node 2
    #SBATCH --gpus-per-node 2  # use 2 GPUs
    #SBATCH --cpus-per-task 8
    #SBATCH --exclusive
    #SBATCH --partition=h100  # adapt this to your cluster
    #SBATCH --time=1:00:00

    # load modules and/or virtual environments and/or containers here

    srun mtt train options-distributed.yaml

Performance
-----------

If the multi-GPU training runs successfully, you should see this in the training log:

.. code-block:: bash

    [2025-10-08 11:34:22][INFO] - Distributed environment set up with MASTER_ADDR=kh080,
    MASTER_PORT=39591, WORLD_SIZE=2, RANK=0, LOCAL_RANK=0
    [2025-10-08 11:34:23][INFO] - Training on 2 devices with dtype torch.float32

This 100-epoch training takes 23 seconds.

.. code-block:: bash

    [2025-10-08 11:34:22][INFO] - Starting training from scratch
    ...
    [2025-10-08 11:34:45][INFO] - Training finished!

Now let's switch off the multi-GPU training by writing ``distributed: false``, and
submit this job again. The training takes 69 seconds.

.. code-block:: bash

    [2025-10-08 11:37:38][INFO] - Setting up model
    ...
    [2025-10-08 11:38:47][INFO] - Training finished!

Multi-GPU fine-tuning
---------------------

You can use multi-GPU for fine-tuning too, by writing ``distributed: True`` in the
``.yaml`` input. For information about fine-tuning, please refer to the
:ref:`sphx_glr_generated_examples_0-beginner_02-fine-tuning.py` example.
"""
