## Training

Model training is done via a custom [LightningCLI](<https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>). Training a GNN for translation classification is as simple as: `python3 run_experiment.py fit --config config.yaml`. Refer to the sample `config.yaml` for the full set of hyperparameters and other configuration options. While the sample config contains some callbacks, additional ones can be specified.
