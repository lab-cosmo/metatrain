Output naming
=============

The name and format of the outputs in ``metatrain`` are based on
those of the `<metatensor.torch.atomistic
https://lab-cosmo.github.io/metatensor/latest/atomistic/outputs.html>`_
package. An immediate example is given by the ``energy`` output.

Any additional outputs present within the library are denoted by the
``mtt::`` prefix. For example, some models can output their last-layer
features, which are named as ``mtt::aux::last_layer_features``, where
``aux`` denotes an auxiliary output.

Outputs that are specific to a particular model should be named as
``mtt::<model_name>::<output_name>``.
