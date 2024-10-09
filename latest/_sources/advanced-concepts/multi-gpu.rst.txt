Multi-GPU training
==================

Some of the architectures in metatensor-models support multi-GPU training.
In multi-GPU training, every batch of samples is split into smaller
mini-batches and the computation is run for each of the smaller mini-batches
in parallel on different GPUs. The different gradients obtained on each
device are then summed. This approach allows the user to reduce the time
it takes to train models.

Here is a list of architectures supporting multi-GPU training:


SOAP-BPNN
---------

SOAP-BPNN supports distributed multi-GPU training on SLURM environments.
The options file to run distributed training with the SOAP-BPNN model looks
like this:

.. literalinclude:: ../../../examples/multi-gpu/soap-bpnn/options-distributed.yaml
   :language: yaml

and the slurm submission script would look like this:

.. literalinclude:: ../../../examples/multi-gpu/soap-bpnn/submit-distributed.sh
   :language: shell
