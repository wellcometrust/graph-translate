# GraphTranslate: Predicting Clinical Trial Translation using Graph Neural Networks on Biomedical Literature

## Description

This library provides functionality to train a graph neural network (GNN) model to predict "translation" of publications (defined as citation by a clinical trial).

## Setup

The translation classifier GNN is built using [Pytorch Geometric (PyG)](<https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html>). To ensure correct functioning of this library, several requirements have to be installed in addition to those specified in `environment.yml`, in the following order:

1. pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f `https://data.pyg.org/whl/torch-2.3.0+cpu.html`
2. pip install torch_geometric
3. pip install torch
4. pip install lightning torchmetrics

To make development of training and inference pipelines run as smoothly as possible, add the translation_classifier to your Python path: `export PYTHONPATH="${PYTHONPATH}:/path/to/translation/translation_classifier"` -> add this to your `~/.bashrc` and `source ~/.bashrc`.

To enable logging to Weights & Biases, run `wandb login`.

## translation_classifier

Contains GNN classifier model code, as well as functionality to load translation data from Wellcome Academic Graph.

Refer to the [data documentation](translation_classifier/data/README.md) or [model documentation](translation_classifier/models/README.md) to find out more about these components.

## training

To find out more about model training, refer to the [training documentation](training/README.md).
