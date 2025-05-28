import torch
from torch_geometric.transforms import BaseTransform


class CitationCount(BaseTransform):
    """Pytorch geometric transform which adds citation counts (approximate number of citations) to node features."""

    def __init__(self, normalize=True):
        self.normalize = normalize

        super().__init__()

    def forward(self, data):
        """Transform the dataset to add normalised citation count.
        This is done by counting the number of unique cited nodes within the edge index.

        """
        feature_indices = torch.arange(data.x.size(0))
        counts = torch.bincount(
            data.edge_index[1], minlength=feature_indices.max().item() + 1
        )
        if self.normalize:
            counts = counts / counts.max()
        index_counts = counts[feature_indices]
        data.x = torch.cat((data.x, index_counts.unsqueeze(-1)), dim=-1)

        return data
