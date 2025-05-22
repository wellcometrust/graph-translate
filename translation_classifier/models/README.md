## Graph neural network (GNN)

See [here](<https://distill.pub/2021/gnn-intro/>) for a gentle introductino to GNNs. Refer to the [PyG documentation](<https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html>) for more information on different model components.

The model code can be found in `gnn.py`. The following model parameters need to be specified:

* `num_features`: Number of input featues.
* `hidden_dim`: Dimension of hidden layer(s).
* `num_layers`: Number of layers. Set this to less than or equal the number of citation levels present in the dataset.
* `dropout`: Dropout value.
* `directed`: Whether to wrap the convolution in a [DirGNNConv](<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.DirGNNConv.html>) for use on directed graphs.
* `conv_type`: Specify the name of the convolutional layer class you want to use (from torch_geometric.nn).
* `jumping_knowledge`: Optional [Jumping Knowledge](<https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.JumpingKnowledge.html>) layer aggregation module. Can be "cat", "max", "lstm", or None.
* `normalize`: Whether to normalise embeddings or not.
* `alpha`: The alpha coefficient used to weight the aggregations of in- and out-edges (optional, only relevant for directed graphs).
* `use_edge_features`: Optionally use edge features in convolution, if present (this may not always be possible depending on the type of conv layer).

## Lightning module

`NodeLevelGNN` is a PyTorch Lightning module for binary node classification, which can be used with a Lightning Trainer (source code in `lit_model.py`). Several metrics are logged to Weights & Biases. The collection of metrics can be easily exetended (see [torchmetrics](<https://lightning.ai/docs/torchmetrics/stable/>) for more information on available metrics).
