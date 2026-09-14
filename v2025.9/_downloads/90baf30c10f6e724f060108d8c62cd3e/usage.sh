# .. _label_basic_usage:
#
# Basic Usage
# ===========
#
# ``metatrain`` is designed for a direct usage from the command line (cli). The program
# is registered via the abbreviation ``mtt`` to your command line. The general help of
# ``metatrain`` can be accessed using
#

mtt --help

# %%
#
# We now demonstrate how to ``train`` and ``evaluate`` a model from the command line.
# For this example we use the :ref:`architecture-soap-bpnn` architecture and a subset of
# the `QM9 dataset <https://paperswithcode.com/dataset/qm9>`_. You can obtain the
# reduced dataset from our :download:`website <../../../static/qm9/qm9_reduced_100.xyz>`.
#
# Training
# --------
#
# To train models, ``metatrain`` uses a dynamic override strategy for your training
# options. We allow a dynamical composition and override of the default architecture
# with either your custom ``options.yaml`` and even command line override grammar. For
# reference and reproducibility purposes ``metatrain`` always writes the fully expanded,
# including the overwritten option to ``options_restart.yaml``. The restart options file
# is written into a subfolder named with the current *date* and *time* inside the
# ``output`` directory of your current training run.
#
# The sub-command to start a model training is
#
# .. code-block:: bash
#
#     mtt train
#
# To train a model you have to define your options. This includes the specific
# architecture you want to use and the data including the training systems and target
# values
#
# The default model and training hyperparameter for each model are listed in their
# corresponding documentation page. We will use these minimal options to run an example
# training using the default hyperparameters of an SOAP BPNN model
#
# .. literalinclude:: ../../../static/qm9/options.yaml
#    :language: yaml
#
# For each training run a new output directory in the format
# ``outputs/YYYY-MM-DD/HH-MM-SS`` based on the current *date* and *time* is created. We
# use this output directory to store checkpoints, the restart ``options_restart.yaml``
# file as well as the log files. To start the training, create an ``options.yaml`` file
# in the current directory and type


mtt train options.yaml

# %%
#
# The functions saves the final model `model.pt` to the current output folder for later
# evaluation. An `extensions/` folder, which contains the compiled extensions for the
# model, might also be saved depending on the architecture. All command line flags of
# the train sub-command can be listed via
#

mtt train --help

# %%
#
# After the training has finished, the ``mtt train`` command generates the
# ``model.ckpt`` (final checkpoint) and ``model.pt`` (exported model) files in the
# current directory, as well as in the ``output/YYYY-MM-DD/HH-MM-SS`` directory.
#
# Evaluation
# ----------
#
# The sub-command to evaluate an already trained model is
#
# .. code-block:: bash
#
#     mtt eval
#
# Besides the trained ``model``, you will also have to provide a file containing the
# system and possible target values for evaluation. The system section of this
# ``eval.yaml`` is exactly the same as for a dataset in the ``options.yaml`` file.
#
# .. literalinclude:: ../../../static/qm9/eval.yaml
#    :language: yaml
#
# Note that the ``targets`` section is optional. If the ``targets`` section is present,
# the function will calculate and report RMSE values of the predictions with respect to
# the real values as loaded from the ``targets`` section. You can run an evaluation by
# typing
#
# We now evaluate the model on the training dataset, where the first arguments specifies
# trained model and the second an option file containing the path of the dataset for
# evaulation. The extensions of the model, if any, can be specified via the ``-e`` flag.

mtt eval model.pt eval.yaml -e extensions/

# %%
#
# The evaluation command predicts those properties the model was trained against; here
# ``"U0"``. The predictions together with the systems have been written in a file named
# ``output.xyz`` in the current directory. The written file starts with the following
# lines

head -n 20 output.xyz

# %%
#
# All command line flags of the eval sub-command can be listed via

mtt eval --help

# %%
#
# An important parameter of ``mtt eval`` is the ``-b`` (or ``--batch-size``) option,
# which allows you to specify the batch size for the evaluation.
#
# Molecular simulations
# ---------------------
#
# The trained model can also be used to run molecular simulations.
# You can find how in the :ref:`tutorials` section.
