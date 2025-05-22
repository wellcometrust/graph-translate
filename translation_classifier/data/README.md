## Create PyG graph dataset

Functionality to process `.parquet` files containing raw nodes, edges, and embeddings is contained within `process.py`. Nodes and edges in these files represent distinct sub-graphs corresponding to each post-publication timedelta. `preprocess.py` includes functionality to filter nodes and edges such that only those source nodes (and associated data) remain which have text embeddings and are cited at least once within the given timedelta by publications which also have text embeddings. Train/test/val splits are provided to the PyG datasets as node masks to the source nodes. There is also an option to resample the training dataset (over- or undersampling). It is advisable to process the data as part of PyG's InMemoryDataset, which will save the tensors to your local memory. This will likely take up several GB of memory, but greatly speed up model training. A `TranslationInMemoryDataset` can be found in `pyg_datasets.py`,which can be used to save various translation datasets to memory.

Note that a custom transform `CitationCount` can be specified when creating the PyG dataset. This will add normalised citation counts to the node features.

## Lightning DataModule

You can find more about Lightning DataModules [here](<https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule>). There is a datamodule for translation classification called `TranslationLitData`, which can be found in `lit_datamodule.py`. This can be used in conjunction with a Lightning Trainer. Graph data can be loaded as a full-batch dataset or with a PyG NeighborLoader (if initialised with `num_neighbors`).
