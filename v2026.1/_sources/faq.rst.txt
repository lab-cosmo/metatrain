==========================
Frequently Asked Questions
==========================

This page is a collection of FAQs.
If you have further questions and suggestions, please do not hesitate to add them to
our `Github discussion page about metatrain`_.

.. _Github discussion page about metatrain: https://github.com/metatensor/metatrain/discussions

Below are links to the sections:

Contents
--------

`Training troubleshooting`_

`General training concepts`_

`Citing us`_

Training troubleshooting
------------------------
.. _Training troubleshooting:

**Q: My training fails with an out of memory error, what can I do?**

**A:** This stems from loading a too large model or too many structures into memory at the same time. Try to reduce batch size or change the model size. Reducing the model size can be achieved by reducing the number of features or the cutoff radius. Refer to to the model docs for details.

**Q: My training is very slow, what can I do?**

**A:** There are several reasons why training can be slow. If possible,
try to reduce the dataset size or increase the batch size.
If available you can also try to run on a GPU, which significantly increases performance times.

**Q: My training is not converging, what can I do?**

**A:** First, please make sure that you dataset is computed consitently and converged to a reasonable accuracy.
Looking at a distribution of your energies per atom can help. Furthermore, outliers, such as large forces
complicate training, so looking at the distribution of the forces and removing structures with large forces
(e.g. all structures with forces with an absolute force > 20 eV/Å) from the dataset can help to stabilize training. For these tasks parity plots can be useful to find outliers.
See our :ref:`sphx_glr_generated_examples_0-beginner_04-parity_plot.py` for how to create them.

General training concepts
-------------------------
.. _General training concepts:

**Q: What cutoff radius should I use?**

**A:** The optimal cutoff radius depends on the type of system you are modeling.

In general, the cutoff should be large enough to include all physically relevant interactions
(e.g., chemical bonding and short-range correlations) but not so large that it adds unnecessary
computational cost. For istance around **4–6 Å** is a good value for most systems, but it can be
increased to **8-10 Å** for condensed phases you expect that correlation at larger distances are important. You can
then test convergence by gradually increasing the cutoff and monitoring whether your target quantities
(energies, forces, or other observables) change significantly.

Note that if you are using massage passing the effective cutoff extends beyond the nominal atomic cutoff.
For example, a model with a 5 Å cutoff and two message-passing layers (PET default hypers) can capture
correlations up to roughly 10 Å.

**Q: In what format should I provide my data?**

**A:** You can find everything on how to prepare your data in
:ref:`sphx_glr_generated_examples_0-beginner_01-data_preparation.py`.

**Q: How small should my errors be before I can use my model to run Molecular Dynamics simulations?**

**A:** This depends on your system, the temperature you want to run your MD and the
dataset you trained on. As rule of thumb, 5% error of the forces on the dataset can be considered good.
But please check the MD along the way if unphysical phenomena (e.g. explosion/implosion of the trajectory)
occur and continue traning and rewise your dataset if necessary.

**Q: How can I use a custom model architecture?**

**A:** You can add a new model architecture to metatrain, if you want to do so have a look at
:ref:`adding-new-architecture`. For adding a custom loss function have a look at :ref:`adding-new-loss`.

**Q: How can I visualize the results of my training?**

**A:** Every training run writes the train log into a csv file. You can use a simple python
script for e.g. parsing the losses.

**Q: How can I get uncertainties for my model?**

**A:** Have a look at the :ref:`LLPR tutorial <llprexample>`. It shows how to use models
with the last-layer prediction rigidity (`LLPR <LLPR_>`_) and local prediction rigidity (`LPR <LPR_>`_).

.. _LLPR: https://arxiv.org/html/2403.02251v1
.. _LPR: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00704

**Q: How can I save and restart my training?**

**A:** Metatrain offers a convenient and automatic way to restart models from checkpoints.
Please have a look at :ref:`checkpoints` for details.

Citing us
---------
.. _Citing us:

**Q: How do I cite metatrain?**

**A:** Please follow the instructions on :ref:`this page <citingmetatrain>`.

