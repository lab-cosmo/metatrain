<h1 align="center">
    <img src="https://raw.githubusercontent.com/metatensor/metatrain/refs/heads/main/docs/src/logo/metatrain-horizontal-dark.svg" alt="Metatensor logo" width="600"/>
</h1>

<h4 align="center">

[![tests status](https://img.shields.io/github/checks-status/metatensor/metatrain/main)](https://github.com/metatensor/metatrain/actions?query=branch%3Amain)
[![documentation](https://img.shields.io/badge/📚_documentation-latest-sucess)](https://metatensor.github.io/metatrain)
[![coverage](https://codecov.io/gh/metatensor/metatrain/branch/main/graph/badge.svg)](https://codecov.io/gh/metatensor/metatrain)
</h4>

<!-- marker-introduction -->

`metatrain` is a command line interface (CLI) to **train** and **evaluate** atomistic
models of various architectures. It features a common `yaml` option inputs to configure
training and evaluation. Trained models are exported as standalone files that can be
used directly in various molecular dynamics (MD) engines (e.g. `ASE`, `LAMMPS`, `i-PI`, 
`TorchSim`, `ESPResSo`,...) using the [metatomic](https://docs.metatensor.org/metatomic)
interface.

The idea behind `metatrain` is to have a general training hub that provides a
homogeneous environment and user interface, transforming every ML architecture into an
end-to-end model that can be connected to MD engines. Any custom architecture compatible
with [TorchScript](https://pytorch.org/docs/stable/jit.html) can be integrated into
`metatrain`, gaining automatic access to a training and evaluation interface, as well as
compatibility with various MD engines.

<!-- marker-architectures -->

# List of Implemented Architectures

Currently `metatrain` supports the following architectures for building an atomistic
model:

| Name                                     | Description                                                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| {ref}`PET <arch-pet>`                          | Point Edge Transformer (PET), interatomic machine learning potential                                                                 |
| {ref}`SOAP-BPNN <arch-soap_bpnn>`              | A Behler-Parrinello neural network with SOAP features                                                                                |
| {ref}`MACE <arch-mace>`                        | A higher order equivariant message passing neural network.                                                                           |
| {ref}`GAP <arch-gap>`                          | Sparse Gaussian Approximation Potential (GAP) using Smooth Overlap of Atomic Positions (SOAP).                                       |
| {ref}`FlashMD <arch-flashmd>`                  | An architecture for the direct prediction of molecular dynamics                                                                      |

<!-- marker-arch-links -->

<!-- links for the different architectures. To be replaced if we are building the docs locally or
on a PR, since the docs use this README file directly.-->
[arch-flashmd]: https://docs.metatensor.org/metatrain/latest/architectures/generated/flashmd.html
[arch-gap]: https://docs.metatensor.org/metatrain/latest/architectures/generated/gap.html
[arch-mace]: https://docs.metatensor.org/metatrain/latest/architectures/generated/mace.html
[arch-pet]: https://docs.metatensor.org/metatrain/latest/architectures/generated/pet.html
[arch-soap_bpnn]: https://docs.metatensor.org/metatrain/latest/architectures/generated/soap_bpnn.html

<!-- marker-documentation -->

# Documentation

For details, tutorials, and examples, please visit our
[documentation](https://metatensor.github.io/metatrain/latest/).

<!-- marker-installation -->

# Installation

Install `metatrain` with pip:

```bash
pip install metatrain
```

Install specific models by specifying the model name. For example, to install the SOAP-BPNN model:

```bash
pip install metatrain[soap-bpnn]
```

We also offer a conda installation:

```bash
conda install -c conda-forge metatrain
```

> ⚠️ The conda installation does not install model-specific dependencies and will only
> work for architectures without optional dependencies such as PET.

After installation, you can use mtt from the command line to train your models!

<!-- marker-quickstart -->

# Quickstart

To train a model, use the following command:

```bash
mtt train options.yaml
```

Where options.yaml is a configuration file specifying training options. For example, the
following configuration trains a *SOAP-BPNN* model on the QM9 dataset:

```yaml
# architecture used to train the model
architecture:
  name: soap_bpnn
training:
  num_epochs: 5  # a very short training run

# Mandatory section defining the parameters for system and target data of the training set
training_set:
  systems: "qm9_reduced_100.xyz"  # file where the positions are stored
  targets:
    energy:
      key: "U0"      # name of the target value
      unit: "eV"     # unit of the target value

test_set: 0.1        # 10% of the training_set are randomly split for test
validation_set: 0.1  # 10% of the training_set are randomly split for validation
```

<!-- marker-shell -->

# Shell Completion

`metatrain` comes with completion definitions for its commands for bash and zsh. You
must manually configure your shell to enable completion support.

To make the completions available, source the definitions in your shell’s startup file
(e.g., `~/.bash_profile`, `~/.zshrc`, or `~/.profile`):

```bash
source $(mtt --shell-completion)
```

<!-- marker-issues -->

# Having problems or ideas?

Having a problem with metatrain? Please let us know by submitting an issue.

Submit new features or bug fixes through a pull request.

<!-- marker-contributing -->

# Contributors

Thanks goes to all people who make metatrain possible:

[![Contributors](https://contrib.rocks/image?repo=metatensor/metatrain)](https://github.com/metatensor/metatrain/graphs/contributors)

# Citing metatrain

If you found ``metatrain`` useful, you can cite its pre-print
(<https://doi.org/10.48550/arXiv.2508.15704>) as

```
@misc{metatrain,
title = {Metatensor and Metatomic: Foundational Libraries for Interoperable Atomistic
Machine Learning},
shorttitle = {Metatensor and Metatomic},
author = {Bigi, Filippo and Abbott, Joseph W. and Loche, Philip and Mazitov, Arslan
and Tisi, Davide and Langer, Marcel F. and Goscinski, Alexander and Pegolo, Paolo
and Chong, Sanggyu and Goswami, Rohit and Chorna, Sofiia and Kellner, Matthias and
Ceriotti, Michele and Fraux, Guillaume},
year = {2025},
month = aug,
publisher = {arXiv},
doi = {10.48550/arXiv.2508.15704},
}
```
