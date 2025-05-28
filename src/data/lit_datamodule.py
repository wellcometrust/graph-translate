import os
from functools import partial

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils import add_self_loops, to_undirected

from . import pyg_datasets, transforms

ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data")


class TranslationLitData(LightningDataModule):
    """Lightning dataset for translational graph data."""

    def __init__(
        self,
        pyg_data="TranslationInMemoryDataset",
        root=ROOT,
        edges=None,
        nodes=None,
        embeddings=None,
        metadata=None,
        features=None,
        downsample=False,
        pre_transform=None,
        directed=False,
        self_loops=False,
        batch_size=32,
        num_neighbors=None,
        **loader_kwargs,
    ):
        """Initialize the Lightning data module for translational graph data.

        Args:
            pyg_data (str): Name of the Pytorch geometric dataset class to use.
            root (str): Root directory for the dataset.
            edges (str): S3 URI to the edges file.
            nodes (str): S3 URI to the nodes file.
            embeddings (str): S3 URI to the embeddings file.
            metadata (str, optional): S3 URI to metadata file.
            features (iterable, optional): Names of additional features.
            downsample (bool, optional): Whether to downsample the training set.
            pre_transform (str, optional): Name of the pre-transform to apply.
            directed (bool, optional): Whether the graph is directed.
            self_loops (bool, optional): Whether to add self-loops to the graph.
            batch_size (int, optional): Batch size for data loaders.
            num_neighbors (list or int, optional): Number of neighbors for NeighborLoader, otherwise will use full-batch DataLoader.
            **loader_kwargs: Additional keyword arguments for data loaders.
        """
        super().__init__()

        self.directed = directed
        self.self_loops = self_loops
        transform_cls = None
        if pre_transform is not None:
            transform_cls = getattr(transforms, pre_transform)()
        self.pyg_data = partial(
            getattr(pyg_datasets, pyg_data),
            root=root,
            edges=edges,
            nodes=nodes,
            embeddings=embeddings,
            metadata=metadata,
            features=features,
            downsample=downsample,
            pre_transform=transform_cls,
        )
        if isinstance(num_neighbors, list):
            self.loader = partial(
                NeighborLoader,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                **loader_kwargs,
            )
            self._loader_type = "neighbor"
        else:
            self.loader = partial(DataLoader, batch_size=1)
            self._loader_type = "full"

    def prepare_data(self):
        """Data preparation step (this is run on a single GPU).

        This is where the data is loaded from memory.
        If specified, the edges are converted to undirected and/or self-loops are added.
        """
        dataset = self.pyg_data()
        self.dataset = dataset
        if not self.directed:
            self.dataset[0].edge_index = to_undirected(self.dataset[0].edge_index)
        if self.self_loops:
            self.dataset[0].edge_index = add_self_loops(self.dataset[0].edge_index)

    def train_dataloader(self):
        """Training dataloader, either a Pytorch geometric NeighborLoader or a full-batch dataloader."""
        if self._loader_type == "neighbor":
            return self.loader(
                data=self.dataset[0], input_nodes=self.dataset[0].train_mask
            )
        else:
            return self.loader(dataset=self.dataset)

    def val_dataloader(self):
        """Validation dataloader, either a Pytorch geometric NeighborLoader or a full-batch dataloader."""
        if self._loader_type == "neighbor":
            return self.loader(
                data=self.dataset[0], input_nodes=self.dataset[0].val_mask
            )
        else:
            return self.loader(dataset=self.dataset)

    def test_dataloader(self):
        """Test dataloader, either a Pytorch geometric NeighborLoader or a full-batch dataloader."""
        if self._loader_type == "neighbor":
            return self.loader(
                data=self.dataset[0], input_nodes=self.dataset[0].test_mask
            )
        else:
            return self.loader(dataset=self.dataset)
