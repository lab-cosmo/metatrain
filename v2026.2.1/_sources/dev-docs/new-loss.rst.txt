.. _adding-new-loss:

Adding a new loss function
==========================

This page describes the required classes and files necessary for adding a new
loss function to ``metatrain``. Defining a new loss can be useful in case some extra
data has to be used to compute the loss.

Loss functions in ``metatrain`` are implemented as subclasses of
:py:class:`metatrain.utils.loss.LossInterface`. This interface defines the
required method :py:meth:`compute`, which takes the model predictions and
the ground truth values as input and returns the computed loss value. The
:py:meth:`compute` method accepts an additional argument ``extra_data`` on top of
``predictions`` and ``targets``, that can be used to pass any extra information needed
for the loss computation.

.. code-block:: python

    from typing import Dict, Optional
    import torch
    from metatrain.utils.loss import LossInterface
    from metatensor.torch import TensorMap

    class NewLoss(LossInterface):
        def __init__(
            self,
            name: str,
            gradient: Optional[str],
            weight: float,
            reduction: str,
        ) -> None:
            ...

        def compute(
            self,
            predictions: Dict[str, TensorMap],
            targets: Dict[str, TensorMap],
            extra_data: Dict[str, TensorMap]
        ) -> torch.Tensor:
            ...


Examples of loss functions already implemented in ``metatrain`` are
:py:class:`metatrain.utils.loss.TensorMapMSELoss` and
:py:class:`metatrain.utils.loss.TensorMapMAELoss`. They both inherit from the
:py:class:`metatrain.utils.loss.BaseTensorMapLoss` class, which implements pointwise
losses for :py:class:`metatensor.torch.TensorMap` objects.
