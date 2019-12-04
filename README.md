# Chainer/PyTorch Migration Library

`chainer-pytorch-migration` is a provisional tool to help migrating Chainer projects to PyTorch.

## Description

chainer-pytorch-migration (CPM) is a tool to help in the migration process of a project from Chainer to PyTorch.

The main features in CPM are:

+ Use PyTorch models with Chainer training scripts
+ Use Chainer models with PyTorch training scripts
+ Use Chainer extensions with Ignite trainers
+ Use PyTorch memory allocator in CuPy

The main goal of CPM is to allow components from the two frameworks to interact together while the migration of a project is on-going.

Please refer to the migration guide for the detailed usage.

## Installation

```sh
pip install chainer-pytorch-migration

# Required only if you want to use `chainer_pytorch_migration.ingite`:
pip install pytorch-ignite

# Required only if you want to use CuPy integration:
# See: https://docs-cupy.chainer.org/en/latest/install.html#install-cupy
pip install cupy-cudaXXX
```

## Contribution Guide

You can contribute to this project by sending a pull request.
After approval, the pull request will be merged by the reviewer.

Before making a contribution, please confirm that:

- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.
