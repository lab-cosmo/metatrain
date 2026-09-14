Override Architecture's Default Parameters
==========================================

In our initial tutorial, we used default parameters to train a model employing the
SOAP-BPNN architecture, as shown in the following config:

.. literalinclude:: ../../../examples/basic_usage/options.yaml
   :language: yaml

While default parameters often serve as a good starting point, depending on your
training target and dataset, it might be necessary to adjust the architecture's
parameters.

First, familiarize yourself with the specific parameters of the architecture you intend
to use. We provide a list of all architectures and their parameters in the
:ref:`available-architectures` section. For example, the parameters of the SOAP-BPNN
models are detailed at :ref:`architecture-soap-bpnn`.

Modifying Parameters (yaml)
---------------------------

As an example, let's increase the number of epochs (``num_epochs``) and the ``cutoff``
radius of the SOAP descriptor. To do this, create a new section in the ``options.yaml``
named ``architecture``. Within this section, you can override the architecture's
hyperparameters. The adjustments for ``num_epochs`` and ``cutoff`` look like this:

.. code-block:: yaml

   architecture:
      name: "soap_bpnn"
      model:
         soap:
            cutoff: 7.0
      training:
         num_epochs: 200

   training_set:
   systems: "qm9_reduced_100.xyz"
   targets:
      energy:
         key: "U0"

   test_set: 0.1
   validation_set: 0.1

Modifying Parameters (Command Line Overrides)
---------------------------------------------

For quick adjustments or additions to an options file, command-line overrides are also
possibility. The changes above can be achieved by typing:

.. code-block:: bash

   mtt train options.yaml \
      -r architecture.model.soap.cutoff=7.0 -r architecture.training.num_epochs=200

Here, the ``-r`` or equivalent ``--override`` flag is used to parse the override flags.
The syntax follows a dotlist-style string format where each level of the options is
seperated by a ``.``. As a further example, to use single precision for your training
you can add ``-r base_precision=32``.

.. note::
   Command line overrides allow adding new values to your training parameters and
   override the architectures as well as the parameters of your provided options file.
