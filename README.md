# GraphTranslate: Predicting Clinical Trial Translation using Graph Neural Networks on Biomedical Literature

## Description

This library provides functionality to train a graph neural network (GNN) model to predict "translation" of publications (defined as citation by a clinical trial).

## Setup

* Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

* Add dependencies for model training: `uv sync`. Models were trained on a cloud compute instance with a Nvidia A10G GPU.

* To enable logging to Weights & Biases, run `wandb login`

## Code structure

`src/data` contains functionality to load graph data from `.parquet`, and `src/models` contains GNN classifier model code.

Refer to the [data documentation](src/data/README.md) or [model documentation](src/models/README.md) to find out more about these components.

## Model training

Model training is done via a custom [LightningCLI](<https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>). Training a GNN for translation classification is as simple as:

```
cd src
uv run run_experiment.py fit --config config.yaml
```

Refer to the sample `config.yaml` for the full set of hyperparameters and other configuration options. While the sample config contains some callbacks, additional ones can be specified.

### Weights & Biases Hyperparameter Optimization

Experiments can be run sequentially over a range of parameters by setting parameters in the `wandconfig.yaml` file. Note that any parameters not set in this file will default to the `config.yaml`. Initialise agent using `wandb sweep wandbconfig.yaml`.

### Citation
If you use GraphTranslate in your research project or are looking to reference this work, please cite the accompanying paper:

[Emily Muller, Justin Boylan-Toomey, Jack Ekinsmyth, Arne Robben, María De La Paz Cardona, and Antonia Langfelder. 2025. GraphTranslate: Predicting Clinical Trial Translation using Graph Neural Networks on Biomedical Literature. In Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025), pages 31–41, Vienna, Austria. Association for Computational Linguistics.](https://aclanthology.org/2025.sdp-1.4/)
