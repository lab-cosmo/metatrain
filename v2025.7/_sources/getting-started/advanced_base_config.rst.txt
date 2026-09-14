.. _advanced_base_conf:

Advanced Base Configuration
===========================

Here, we show how some advanced base properties in the ``options.yaml`` can
be adjusted. They should be written without indentation in the ``options.yaml`` file.

:param device: The device in which the training should be run. Takes two possible
    values: ``cpu`` and ``gpu``. Default: ``cpu``
:param base_precision: Override the base precision of all floats during training. By
    default an optimal precision is obtained from the architecture. Changing this will
    have an effect on the memory consumption during training and maybe also on the
    accuracy of the model. Possible values: ``64``, ``32`` or ``16``.
:param seed: Seed used to start the training. Set all the seeds of ``numpy.random``,
    ``random``, ``torch`` and ``torch.cuda`` (if available) to the same value ``seed``.
    If ``seed`` is not the initial seed will be set to a random number. This initial
    seed will be reported in the output folder
:param wandb: If you want to use Weights and Biases (wandb) for logging, create a new
    section with this name. The parameters of section are the same as of the `wandb.init
    <https://docs.wandb.ai/ref/python/init/>`_ method and a minimal example of the
    section is:

    .. code-block:: yaml

        wandb:
          project: my_project
          name: my_run_name
          tags:
            - tag1
            - tag2
          notes: This is a test run

    All parameters of your options file will be automatically added to the wandb run so
    you don't have to set the ``config`` parameter.

    .. important::

        You need to install wandb with ``pip install wandb`` if you want to use this
        logger. **Before** running also set up your credentials with ``wandb login``
        from the command line. See `wandb login
        documentation <https://docs.wandb.ai/ref/cli/wandb-login/>`_ for details on the
        setup.

In the next tutorials we show how to override the default parameters of an architecture.
