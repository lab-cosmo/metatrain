Output naming
=============

The name and format of the outputs in ``metatensor-models`` are based on
those of the :py:mod:`metatensor.torch.atomistic` package. An immediate example
is given by the ``energy`` output.

Any additional outputs present within the library are denoted by the
``mts-models::`` prefix. For example, some models can output their last-layer
features, which are named as ``mts-models::aux::last_layer_features``, where
``aux`` denotes an auxiliary output.

Outputs that are specific to a particular model should be named as
``mts-models::<model_name>::<output_name>``.