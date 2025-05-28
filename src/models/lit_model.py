import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryROC,
)

from .gnn import GNN

LOSS = "binary_cross_entropy_with_logits"
HIDDEN_CHANNELS = 512
DROPOUT = 0.5
NUM_FEATURES = 768
NUM_LAYERS = 2
CONV_TYPE = "GCNConv"
DIRECTED = False
JUMPING_KNOWLEDGE = False
NORMALIZE = False
ALPHA = 0.5
WEIGHTED_LOSS = False
USE_EDGE_FEATURES = False


class NodeLevelGNN(LightningModule):
    """Pytorch Lightning module for binary classification with a graph neural network."""

    def __init__(
        self,
        num_features,
        hidden_dim,
        num_layers,
        dropout,
        conv_type,
        directed,
        jumping_knowledge,
        normalize,
        alpha,
        loss,
        weighted_loss,
        use_edge_features,
    ):
        """Initialise the NodeLevelGNN model.

        Args:
            num_features(int): Number of input features.
            hidden_dim(int): Number of hidden channels.
            num_layers(int): Number of GNN layers.
            dropout(float): Dropout rate.
            conv_type(str): Type of convolutional layer to use.
            directed(bool): Whether to use a directed graph neural network model (DirGNN).
            jumping_knowledge(str, optional): Jumping knowledge mode (e.g., "cat", "max", "lstm").
            normalize(bool): Whether to normalize the output.
            alpha(float): Alpha coefficient for directed graphs.
            loss(str, optional): Loss function to use. Defaults to binary cross-entropy with logits.
            weighted_loss(bool): Whether to use a weighted loss function.
            use_edge_features(bool): Whether to use edge features in the model.

        """
        super().__init__()
        self.save_hyperparameters(ignore="_instantiator")

        num_features = num_features or NUM_FEATURES
        hidden_dim = hidden_dim or HIDDEN_CHANNELS
        num_layers = num_layers or NUM_LAYERS
        dropout = dropout or DROPOUT
        conv_type = conv_type or CONV_TYPE
        directed = directed or DIRECTED
        jumping_knowledge = jumping_knowledge or JUMPING_KNOWLEDGE
        normalize = normalize or NORMALIZE
        alpha = alpha or ALPHA
        weighted_loss = weighted_loss or WEIGHTED_LOSS
        use_edge_features = use_edge_features or USE_EDGE_FEATURES

        self.model = GNN(
            num_features,
            hidden_dim,
            num_layers,
            dropout,
            directed,
            conv_type,
            jumping_knowledge,
            normalize,
            alpha,
            use_edge_features,
        )

        self.loss = loss or LOSS
        self.loss_fn = getattr(torch.nn.functional, self.loss)
        self.weighted_loss = weighted_loss

        metrics = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryPrecision(),
                BinaryAveragePrecision(thresholds=5),
                BinaryRecall(),
                BinaryF1Score(),
                AUROC(task="binary", num_classes=2),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.val_roc = BinaryROC()
        self.val_pr = BinaryPrecisionRecallCurve()

    def forward(self, data, mode="train"):
        """Run data through the model.

        Depending on the mode (train, val, test), the data is
        subset by the relevant node mask such that the loss and metrics are only calculated based on the subset.

        Args:
            data(torch_geometric.data.Data): Graph data batch.
            mode(str): Whether model is used in train, val, or test mode.

        Returns:
            torch.Tensor: Updated node features for masked subset.
            torch.Tensor: Binary target values for masked subset.
            torch.Tensor: Loss.

        """
        x, y, edge_index, weight = data.x, data.y, data.edge_index, data.weight

        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        x = self.model(x, edge_index, edge_attr)
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        x, y = x[mask].squeeze(), y[mask]
        if not self.weighted_loss:
            weight = None
        loss = self.loss_fn(x, y, pos_weight=weight)
        return x, y, loss

    def training_step(self, batch, batch_idx):
        """Train the model on a batch of data and log training metrics.

        Args:
            batch(torch_geometric.data.Data): Graph data batch.
            batch_idx(int): Index of the batch in the training set.

        Returns:
            torch.Tensor: Training loss.

        """
        logits, y, loss = self.forward(batch, mode="train")
        self.log("train_loss", loss, batch_size=batch.batch_size)
        output = self.train_metrics(logits, y.long())
        self.log_dict(output, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate the model on a batch of data and log validation metrics.

        Args:
            batch(torch_geometric.data.Data): Graph data batch.
            batch_idx(int): Index of the batch in the validation set.

        Returns:
            torch.Tensor: Validation loss.

        """
        logits, y, loss = self.forward(batch, mode="val")
        self.log("val_loss", loss, batch_size=batch.batch_size)
        self.valid_metrics.update(logits, y.long(), batch_size=batch.batch_size)
        self.val_roc.update(logits, y.long())
        self.val_pr.update(logits, y.long())

    def on_validation_epoch_end(self):
        """Compute metrics on the validation set after an epoch and log to Weights & Biases."""
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()
        wandb.log({"val_roc": wandb.Image(self.val_roc.plot()[0])})
        wandb.log({"val_pr": wandb.Image(self.val_pr.plot()[0])})
        plt.close()

    def test_step(self, batch, batch_idx):
        """Test the model on a batch of data and log test metrics.

        Args:
            batch(torch_geometric.data.Data): Graph data batch.
            batch_idx(int): Index of the batch in the test set.

        """
        out, y, _ = self.forward(batch, mode="test")
        self.test_metrics.update(out, y.long(), batch_size=batch.batch_size)

    def on_test_epoch_end(self):
        """Compute metrics on the full test set and log to Weights & Biases."""
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()
