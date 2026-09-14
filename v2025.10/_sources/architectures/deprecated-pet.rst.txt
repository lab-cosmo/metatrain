.. _architecture-deprecated-pet:

PET (deprecated)
================

.. warning::

  This is an **deprecated model**. You should consider using the stable metatrain
  implementation of the PET model instead.

Metatrain training interface to the Point Edge transformer (PET)
:footcite:p:`pozdnyakov_smooth_2023` architecture.

Installation
------------

To install the package, you can run the following command in the root directory of the
repository:

.. code-block:: bash

  pip install metatrain[deprecated-pet]

This will install the package with the PET dependencies.


Default Hyperparameters
-----------------------
The default hyperparameters for the PET model are:

.. literalinclude:: ../../../src/metatrain/deprecated/pet/default-hypers.yaml
  :language: yaml

Tuning Hyperparameters
----------------------
1) Set ``R_CUT`` so that there are about 20-30 neighbors on average for your dataset.
2) Fit the model with the default values for all the other hyperparameters.
3) Ensure that you fit the model long enough for the error to converge. (If not, you can
   always continue fitting the model from the last checkpoint.)
4) [Optional, recommended for large datasets] Increase the scheduler step size, and
   refit the model from scratch until convergence. Do this for several progressively
   increased values for the scheduler step size until convergence.
5) [Optional, this step aims to create a lighter and faster model, not to increase
   accuracy.] Set ``N_TRANS_LAYERS`` to 2 instead of 3, and repeat steps 3) and 4). If
   step 4) was already done for the default ``N_TRANS_LAYERS`` value of 3, you can
   probably reuse the converged scheduler step size. The resulting model would be about
   1.5 times faster than the default one, hopefully with very little deterioration of
   the accuracy or without any at all.
6) [Optional, quite laborious, 99% you don't need this] Read sections 6 and 7 of the
   `PET paper <https://arxiv.org/abs/2305.19302>`_, which discuss the architecture, main
   hyperparameters, and an ablation study illustrating their impact on the model's
   accuracy. Design your own experiments.

More details:
~~~~~~~~~~~~~
There are two significant groups of hyperparameters controlling PET fits. The first
group consists of the hyperparameters related to the model architecture itself, such as
the number of layers, type of activation function, etc. The second group consists of
settings that control how the fitting is done, such as batch size, the total number of
epochs, learning rate, parameters of the learning rate scheduler, and so on.

Within conventional wisdom originating from *traditional* models, such as linear and
kernel regression, the second group of hyperparameters that controls the optimizer's
behavior might seem unimportant. Indeed, when fitting linear or kernel models, the exact
value of the optimum is usually achieved by linear algebra methods, and thus, the
particular choice of optimizer has little importance.

However, with deep neural networks, the situation is drastically different. The exact
minimum of the loss is typically never achieved; instead, the model asymptotically
converges to it during fitting. It is essential to ensure that the total number of
epochs is sufficient for the model to approach the optimum closely, thus achieving good
accuracy.

**In the case of PET, there is only one hyperparameter that MUST be manually adjusted
for each new dataset: the cutoff radius.** The selected cutoff significantly impacts the
model's accuracy and fitting/inference times, making it very sensitive to this
hyperparameter. All other hyperparameters can be safely set to their default values. The
next reasonable step (after fitting with default settings), especially for large
datasets is to try to increase the duration of fitting and see if it improves the
accuracy of the obtained model.

Selection of ``R_CUT``
**********************
A good starting point is to select a cutoff radius that ensures about 20-30 neighbors on
average. This can be done by analyzing the neighbor lists for different cutoffs before
launching the training script. `This
<https://ase-lib.org/ase/neighborlist.html>`_ is an example of a neighbor list
constructor in Python.

For finite configurations, such as small molecules in COLL/QM9/rmd17 datasets, it makes
sense to select ``R_CUT`` large enough to encompass the whole molecule. For instance, it
can be set to 100 Å, as there are no numerical instabilities for arbitrarily large
cutoffs.

The hyperparameter for the cutoff radius is called ``R_CUT.``

Selection of fitting duration
*****************************
The second most important group of settings is the one that adjusts the fitting duration
of the model. Unlike specifying a dataset-specific cutoff radius, this step is optional
since reasonable results can be obtained with the default fitting duration. The time
required to fit the model is a complex function of the model's size, the dataset's size,
and the complexity of the studied interatomic interactions. The default value might be
insufficient for large datasets. If the model is still underfit after the predefined
number of epochs, the fitting procedure can be continued by relaunching the fitting
script.

However, the total number of epochs is only part of the equation. Another key aspect is
the rate at which the learning rate decays. We use `StepLR
<https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`_ as a
learning rate scheduler. This scheduler reduces the learning rate by a factor of
``gamma`` (``new_learning_rate = old_learning_rate * gamma``) every ``step_size``
epochs. In the current implementation of PET, ``gamma`` is fixed at 0.5, meaning that
the learning rate is halved every ``step_size`` epochs.

If ``step_size`` is set too small, the learning rate will decrease to very low values
too quickly, hindering the convergence of PET. Prolonged fitting under these conditions
will be ineffective due to the nearly zero learning rate. Therefore, achieving complete
convergence requires not only a sufficient number of epochs but also an appropriately
large ``step_size``. For typical moderately sized datasets, the default value should
suffice. However, for particularly large datasets, increasing ``step_size`` may be
necessary to ensure complete convergence. The hyperparameter controlling the
``step_size`` of the StepLR learning rate scheduler is called ``SCHEDULER_STEP_SIZE``.

It is worth noting that the default ``step_size`` is quite large. Thus, it is normal
if, when fitting on V100, which is quite slow, there is no event of lr rate decrease
during the first day or even during a couple of days. In addition, for some datasets,
the fitting might take longer than for others (related to inhomogeneous densities),
which can further postpone the first event of lr decrease.

The discussed convergence above, especially in terms of the total duration of fitting,
should preferably be checked on log-log plots showing how the validation error depends
on the epoch number. The raw log values are typically hard to extract useful insights
from.

For hyperparameters like ``SCHEDULER_STEP_SIZE``, ``EPOCH_NUM``, ``BATCH_SIZE``, and
``EPOCHS_WARMUP``, either normal or atomic versions can be specified.
``SCHEDULER_STEP_SIZE`` was discussed above; ``EPOCH_NUM`` represents the total number
of epochs, and ``BATCH_SIZE`` is the number of structures sampled in each minibatch for
a single step of stochastic gradient descent. The atomic versions are termed
``SCHEDULER_STEP_SIZE_ATOMIC``, ``EPOCH_NUM_ATOMIC``, ``BATCH_SIZE_ATOMIC``, and
``EPOCHS_WARMUP_ATOMIC``. The motivation for the atomic versions is to improve the
transferability of default hyperparameters across heterogeneous datasets. For instance,
using the same batch size for datasets with structures of very different sizes makes
little sense. If one dataset contains molecules with 10 atoms on average and another
contains nanoparticles with 1000 atoms, it makes sense to use a 100 times larger batch
size in the first case. If ``BATCH_SIZE_ATOMIC`` is specified, the normal batch size is
computed as ``BATCH_SIZE = BATCH_SIZE_ATOMIC /
(average_number_of_atoms_in_the_training_dataset)``. Similar logic applies to
``SCHEDULER_STEP_SIZE,`` ``EPOCH_NUM,`` and ``EPOCHS_WARMUP.`` In these cases, normal
versions are obtained by division by the total number of atoms of structures in the
training dataset. All default values are given by atomic versions for better
transferability across various datasets.

To increase the step size of the learning rate scheduler by, for example, 2 times, take
the default value for ``SCHEDULER_STEP_SIZE_ATOMIC`` from the default hypers and
specify a value that's twice as large.

It is worth noting that the stopping criterion of PET is either exceeding the maximum
number of epochs (specified by ``EPOCH_NUM`` or ``EPOCH_NUM_ATOMIC``) or exceeding the
specified maximum fitting time (controlled by the hyperparameter ``MAX_TIME``). By
default, the second criterion is used, with the default number of epochs set nearly to
infinity, while the default maximum time is set to be 65 hours.

Lightweight Model
*****************
The default hyperparameters were selected with one goal in mind: to maximize the
probability of achieving the best accuracy on a typical moderate-sized dataset. As a
result, some default hyperparameters might be excessive, meaning they could be adjusted
to significantly increase the model's speed with minimal impact on accuracy. For
practical use, especially when conducting massive calculations where model speed is
crucial, it may be beneficial to set ``N_TRANS_LAYERS`` to ``2`` instead of the default
value of ``3``. The ``N_TRANS_LAYERS`` hyperparameter controls the number of transformer
layers in each message-passing block (For more details see
:footcite:t:`pozdnyakov_smooth_2023`). This adjustment would result in a model that is
about *1.5 times* more lightweight and faster, with an expected minimal deterioration in
accuracy.


Description of the Architecture
-------------------------------
This section contains a simplified description of the architecture covering
most important macro-organization without all the details and nuances.

PET is a graph neural network (GNN) architecture featuring
``N_GNN_LAYERS`` message-passing layers. At each layer, messages are exchanged
between all atoms within a distance ``R_CUT`` from each other. The functional
form of each layer is an arbitrarily deep transformer applied individually to
each atom. Atomic environments are constructed around each atom, defined by all
neighbors within ``R_CUT``. Each neighbor sends a message to the central atom,
with each message being a token of fixed size ``TRANSFORMER_D_MODEL``.

These tokens are processed by a transformer, which performs a permutationally
equivariant sequence-to-sequence transformation. The output sequence is then
treated as outbound messages from the central atom to all neighbors. Consequently,
for a model with ``N_GNN_LAYERS`` layers and a system with ``N`` atoms, there are
``N_GNN_LAYERS`` individual transformers with distinct weights, each independently
invoked ``N`` times, resulting in ``N_GNN_LAYERS * N`` transformer runs. The
number of input tokens for each transformer run is determined by the number of
neighbors of the central atom.

In addition to an input message from a neighboring atom, geometric information
about the displacement vector ``r_ij`` from the central atom to the corresponding
neighbor is incorporated into the token. After each message-passing layer, all
output messages are fed into a head (individual for each message-passing layer),
implemented as a shallow MLP, to produce a contribution to the total prediction.
The total prediction is computed
as the sum of all head outputs over all message-passing layers and all messages.

This architecture is rigorously invariant with respect to translations because it
uses displacement vectors that do not change if both the central atom and a neighbor
are rigidly shifted. It is invariant with respect to permutations of identical atoms
because the transformer defines a permutationally covariant sequence-to-sequence
transformation, and the sum over the contributions from all edges yields an overall
invariant energy prediction. However, it is not rotationally invariant since it
operates with the raw Cartesian components of displacement vectors.

Architecture Hyperparameters
----------------------------
.. warning::

  While PET supports CPU training, it is highly recommended to use a CUDA GPU for
  significantly faster training. CPU training can be very slow.

- ``RANDOM_SEED``: random seed
- ``CUDA_DETERMINISTIC``: if applying PyTorch reproducibility settings
- ``MULTI_GPU``: use multi-GPU training (on one node) using DataParallel from
  PyTorch-Geometric
- ``R_CUT``: cutoff radius
- ``CUTOFF_DELTA``: width of the transition region for a cutoff function used
  by PET to ensure smoothness with respect to the (dis)appearance of atoms at
  the cutoff sphere
- ``GLOBAL_AUG``: whether to use global augmentation or a local one, rotating
  atomic environments independently
- ``USE_ENERGIES``: whether to use energies for training
- ``USE_FORCES``: whether to use forces for training
- ``SLIDING_FACTOR``: sliding factor for exponential sliding averages of MSE in
  energies and forces in our combined loss definition
- ``ENERGY_WEIGHT``: $w_{E}$, dimensionless energy weight in our combined loss
  definition
- ``N_GNN_LAYERS``: number of message-passing blocks
- ``TRANSFORMER_D_MODEL``: was denoted as d_{pet} in the main text of
  the paper
- ``TRANSFORMER_N_HEAD``: number of heads of each transformer
- ``TRANSFORMER_DIM_FEEDFORWARD``: feedforward dimensionality of each
  transformer
- ``HEAD_N_NEURONS``: number of neurons in the intermediate layers of HEAD MLPs
- ``N_TRANS_LAYERS``: number of layers of each transformer
- ``ACTIVATION``: activation function used everywhere
- ``INITIAL_LR``: initial learning rate
- ``MAX_TIME``: maximal time to train the model in seconds

----

For parameters such as ``EPOCH_NUM`` the user can specify either normal
``EPOCH_NUM`` or ``EPOCH_NUM_ATOMIC``. If the second is specified, normal
``EPOCH_NUM`` is computed as ``EPOCH_NUM_ATOMIC / (total number of atoms in the
training dataset)``. Similarly defined are:

- ``SCHEDULER_STEP_SIZE_ATOMIC``: step size of StepLR learning rate schedule
- ``EPOCHS_WARMUP_ATOMIC``: linear warmup time

For the batch size, the normal version of batch size is computed as
``BATCH_SIZE_ATOMIC / (average number of atoms in structures in the training
dataset)``.

- ``ATOMIC_BATCH_SIZE``: batch size

----

- ``USE_LENGTH``: explicitly use length in r embedding or not
- ``USE_ONLY_LENGTH``: use only length in r embedding (used to get auxiliary
  intrinsically invariant models)
- ``USE_BOND_ENERGIES``: use bond contributions to energies or not
- ``AVERAGE_BOND_ENERGIES``: average bond contributions or sum
- ``BLEND_NEIGHBOR_SPECIES``: if True, explicitly encode embeddings of neighbor
  species to the overall embeddings in each message-passing block; if False,
  specify the very first input messages as embeddings of neighbor species
  instead
- ``R_EMBEDDING_ACTIVATION``: apply or not activation after computing r
  embedding by a linear layer
- ``COMPRESS_MODE``: if "mlp," get overall embedding either by MLP; if "linear,"
  use simple linear compression instead
- ``ADD_TOKEN_FIRST``: add or not token associated with central atom for the
  very first message-passing block
- ``ADD_TOKEN_SECOND``: add or not token associated with central atom for all
  the others (to be renamed in future)
- ``AVERAGE_POOLING``: if not using a central token, controls if summation or
  average pooling is used
- ``USE_ADDITIONAL_SCALAR_ATTRIBUTES``: if using additional scalar attributes
  such as collinear spins
- ``SCALAR_ATTRIBUTES_SIZE``: dimensionality of additional scalar attributes

References
----------
.. footbibliography::
