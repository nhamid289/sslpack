# sslpack

A package for semi-supervised learning.

This package is built from Microsoft's Unified SSL Benchmark (https://github.com/microsoft/Semi-supervised-learning/tree/main)
but is designed to be more transparent, more flexible and more extensible than USB.


## Getting started

`sslpack` relies on torch, torchvision and several other packages. It is recommended to use a pip or conda environment.
First, clone this repository and then install the prerequisites.
```sh
pip install -r requirements.txt
```
Then, install `sslpack` itself
```sh
pip install .
```

## Usage

`sslpack` is designed for making scripts that look like conventional Pytorch scripts. The key components such as the model, dataloader, optimiser are all modular. See the examples available in [Examples](https://github.com/nhamid289/sslpack/tree/main/examples)

