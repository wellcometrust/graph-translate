import torch
from torch_geometric.data import InMemoryDataset

from .process import TranslationDataset


class TranslationInMemoryDataset(InMemoryDataset):
    """Pytorch geometric in-memory dataset for translational publications."""

    def __init__(
        self,
        root,
        edges=None,
        nodes=None,
        embeddings=None,
        metadata=None,
        resample_train="downsample",
        transform=None,
        features=None,
        downsample=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
    ):
        """Initialize the TranslationInMemoryDataset.

        Args:
            root (str): Root directory where the dataset should be stored.
            edges (str): S3 URI to the edges file.
            nodes (str): S3 URI to the nodes file.
            embeddings (str): S3 URI to the embeddings file.
            metadata (str): S3 URI to the metadata file.
            resample_train (str): Method for resampling training data, default is "downsample".
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
            features (iterable, optional): Names of additional features.
            downsample (float, optional): Factor by which to downsample the dataset.
            pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version before saving to disk.
            pre_filter (callable, optional): A function that takes in a Data object and returns a boolean mask indicating which data points to keep.
            log (bool, optional): Whether to log processing steps. Defaults to True.
            force_reload (bool, optional): Whether to force reloading of the dataset. Defaults to False.
        """
        self.resample_train = resample_train
        self.edges = edges
        self.nodes = nodes
        self.embeddings = embeddings
        self.metadata = metadata
        self.features = features
        self.downsample = downsample

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.data = torch.load(self.processed_paths[0])
        self.pre_transform = pre_transform

    @property
    def raw_file_names(self):
        """Return the names of the raw files."""
        return [self.edges, self.nodes, self.embeddings]

    @property
    def processed_file_names(self):
        """Return the name of the processed file."""
        return "translationdata.pt"

    def process(self):
        """Process the raw data into a Pytorch Geometric dataset."""
        data = TranslationDataset(
            self.edges, self.nodes, self.embeddings, self.metadata, self.features
        )
        data.generate_node_masks(resample_train=self.resample_train)
        data, node_mapping = data.to_pyg(self.downsample)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, self.processed_paths[0])
        node_mapping.to_csv(
            self.processed_paths[0].split("processed/translationdata.pt")[0]
            + "node_mapping.csv"
        )
