import wandb
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import Logger

from data import TranslationLitData
from models import NodeLevelGNN


class LoggerSaveConfigCallback(SaveConfigCallback):
    """Custom callback to log Lightning CLI config to Weights & Biases."""

    def save_config(self, trainer, pl_module, stage):
        """Save the configuration to Weights & Biases.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (LightningModule): The PyTorch Lightning module.
            stage (str): The stage of the training process (e.g., 'fit', 'validate').
        """
        if isinstance(trainer.logger, Logger):
            combined_config = {**self.config, **wandb.config}
            trainer.logger.log_hyperparams({"config": combined_config})


class TranslationLightningCLI(LightningCLI):
    """Pytorch Lightning CLI with linked arguments for directed graph data."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """Add custom arguments to the Lightning CLI parser.

        Args:
            parser (LightningArgumentParser): The parser to which arguments are added.
        """
        parser.link_arguments("data.directed", "model.directed")
        parser.add_argument(
            "--wandb_project", default="my_project", type=str, help="W&B project name"
        )
        parser.add_argument(
            "--wandb_entity", default=None, type=str, help="W&B entity name (optional)"
        )


def main():
    """Main function to run the experiment with the Lightning CLI and Weights & Biases logging."""
    wandb.init()

    TranslationLightningCLI(
        NodeLevelGNN,
        TranslationLitData,
        seed_everything_default=123,
        save_config_kwargs={"overwrite": True},
        save_config_callback=LoggerSaveConfigCallback,
    )


if __name__ == "__main__":
    main()
