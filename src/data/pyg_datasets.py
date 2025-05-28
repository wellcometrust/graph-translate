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
        self.resample_train = resample_train
        self.edges = edges
        self.nodes = nodes
        self.embeddings = embeddings
        self.metadata = metadata
        self.features = features
        self.downsample = downsample

        super(TranslationInMemoryDataset, self).__init__(
            root, transform, pre_transform, pre_filter, log, force_reload
        )
        self.data = torch.load(self.processed_paths[0])
        self.pre_transform = pre_transform

    @property
    def raw_file_names(self):
        return [self.edges, self.nodes, self.embeddings]

    @property
    def processed_file_names(self):
        return "translationdata.pt"

    def process(self):
        data = TranslationDataset(self.edges, self.nodes, self.embeddings, self.metadata, self.features)
        data.generate_node_masks(resample_train=self.resample_train)
        data, node_mapping = data.to_pyg(self.downsample)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, self.processed_paths[0])
        node_mapping.to_csv(self.processed_paths[0].split('processed/translationdata.pt')[0] + 'node_mapping.csv')
