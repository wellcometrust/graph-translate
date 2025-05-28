from torch.nn import Linear
from torch_geometric.nn.conv import DirGNNConv


def get_conv(in_channels, out_channels, conv_cls, alpha, directed):
    """Initialise convolutional layer.

    Args:
        in_channels(int): Number of input channels.
        out_channels(int): Number of output channels.
        conv_cls(torch_geometric.nn.conv, GraphLinear): Conv class.
        alpha(float, optional): The alpha coefficient used in directed graphs to weight the aggregations of in- and out-edges.
        directed(bool): Whether to wrap the convolution in a DirGNNConv for use on directed graphs.

    Returns:
        torch.nn: GNN conv or linear layer.

    """
    conv = conv_cls(in_channels, out_channels)
    if directed:
        return DirGNNConv(conv=conv, alpha=alpha)
    return conv


class GraphLinear(Linear):
    """Simple linear layer which ignores the edge index."""

    def __init__(self, in_features, out_features, bias=True):
        """Initialise the GraphLinear layer.

        Args:
            in_features(int): Number of input features.
            out_features(int): Number of output features.
            bias(bool, optional): Whether to include a bias term. Defaults to True.
        """
        super().__init__(in_features, out_features, bias)

    def forward(self, x, edge_index, edge_attr=None):
        """Perform forward pass on graph data.

        Args:
            x(torch.Tensor): Input features.
            edge_index(torch.Tensor): Edge index (not used).
            edge_attr(torch.Tensor, optional): Edge attribute(s) (not used).

        Returns:
            torch.Tensor: Output features.
        """
        x = super().forward(x)
        return x
