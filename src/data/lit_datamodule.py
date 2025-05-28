import os
from functools import partial

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_undirected

from . import pyg_datasets
from . import transforms

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
        pre_transform=None,
        directed=False,
        self_loops=False,
        batch_size=32,
        num_neighbors=[30, 30],
        **loader_kwargs
    ):
        super().__init__()

        self.directed = directed
        self.self_loops = self_loops
        transform_cls = None
        if pre_transform is not None:
            transform_cls = getattr(transforms, pre_transform)()
        self.pyg_data = partial(
            getattr(pyg_datasets, pyg_data),
            root=root,
            nodes=nodes,
            edges=edges,
            embeddings=embeddings,
            pre_transform=transform_cls,
        )
        if isinstance(num_neighbors, list):
            self.loader = partial(
                NeighborLoader,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                **loader_kwargs
            )
            self._loader_type = "neighbor"
        else:
            self.loader = partial(DataLoader, batch_size=1)
            self._loader_type = "full"

    def prepare_data(self):
        """Data preparation step (this is run on a single GPU). This is where the data is loaded from memory.
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
