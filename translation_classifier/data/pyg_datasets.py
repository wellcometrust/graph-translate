import torch
from torch_geometric.data import InMemoryDataset

from translation_classifier.data.process import TranslationDataset


class TranslationInMemoryDataset(InMemoryDataset):
    """Pytorch geometric in-memory dataset for translational publications."""

    def __init__(
        self,
        root,
        edges=None,
        nodes=None,
        embeddings=None,
        resample_train="downsample",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
    ):
        self.resample_train = resample_train
        self.edges = edges
        self.nodes = nodes
        self.embeddings = embeddings

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
        data = TranslationDataset(self.edges, self.nodes, self.embeddings)
        data.generate_node_masks(resample_train=self.resample_train)
        data = data.to_pyg()
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, self.processed_paths[0])
