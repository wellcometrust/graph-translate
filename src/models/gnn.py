from functools import partial

import torch
import torch.nn.functional as F
import torch_geometric.nn
from torch.nn import Linear
from torch.nn import ModuleList
from torch_geometric.nn import JumpingKnowledge

from .utils import get_conv
from .utils import GraphLinear


class GNN(torch.nn.Module):
    """Graph neural network model."""

    def __init__(
        self,
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
    ):
        super(GNN, self).__init__()

        if conv_type == "Linear":
            conv_cls = GraphLinear
            directed = False
            jumping_knowledge = None
        else:
            conv_cls = getattr(torch_geometric.nn.conv, conv_type)

        add_conv = partial(get_conv, conv_cls=conv_cls, alpha=alpha, directed=directed)

        if num_layers == 1:
            self.convs = ModuleList([add_conv(num_features, 1)])
        else:
            self.convs = ModuleList([add_conv(num_features, hidden_dim)])
            for _ in range(num_layers - 2):
                self.convs.append(add_conv(hidden_dim, hidden_dim))

        if isinstance(jumping_knowledge, str):
            self.convs.append(add_conv(hidden_dim, hidden_dim))
            self.jump = JumpingKnowledge(
                mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers
            )
            if jumping_knowledge == "cat":
                hidden_dim = hidden_dim * num_layers
            self.lin = Linear(hidden_dim, 1)
        else:
            self.convs.append(add_conv(hidden_dim, 1))

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.use_edge_features = use_edge_features

    def forward(self, x, edge_index, edge_attr):
        """Apply the graph neural network to input graph data and return logits.

        Args:
            x(torch.Tensor): Node features.
            edge_index(torch.Tensor): Edge index (citing -> cited node index).
            edge_attr(torch.Tensor, optional): Edge attribute(s). Note that not all types of
                graph convolutional layer are able to use edge attributes.

        Returns:
            torch.Tensor: Logits for binary classification.

        """
        xs = []
        for i, conv in enumerate(self.convs):
            if self.use_edge_features:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if isinstance(self.jumping_knowledge, str):
            x = self.jump(xs)
            x = self.lin(x)

        return x
