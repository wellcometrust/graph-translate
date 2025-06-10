## Create PyG graph dataset

Functionality to process `.parquet` files containing raw nodes (publication IDs and labels), edges (cited and citing publication IDs), and embeddings (publication IDs and text embeddings) is contained within `process.py`. The edges represent distinct sub-graphs corresponding to each post-publication timedelta. `preprocess.py` includes functionality to filter nodes and edges such that only those source nodes (and associated data) remain which have text embeddings and are cited at least once within the given timedelta by publications which also have text embeddings. Train/test/val splits are provided to the PyG datasets as node masks to the source nodes. There is also an option to resample the training dataset (over- or undersampling). It is advisable to process the data as part of PyG's InMemoryDataset, which will save the tensors to your local memory. This will likely take up several GB of memory, but greatly speed up model training. A `TranslationInMemoryDataset` can be found in `pyg_datasets.py`,which can be used to save various translation datasets to memory.

Note that a custom transform `CitationCount` can be specified when creating the PyG dataset. This will add normalised citation counts to the node features.

## Lightning DataModule

You can find more about Lightning DataModules [here](<https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule>). There is a datamodule for translation classification called `TranslationLitData`, which can be found in `lit_datamodule.py`. This can be used in conjunction with a Lightning Trainer. Graph data can be loaded as a full-batch dataset or with a PyG NeighborLoader (if initialised with `num_neighbors`).

## Input Data

The input data is expected to be in the following format:

Nodes: Parquet file in the following format:

```parquet
publication_id: string
year: int
label: bool
```

Where `publication_id` is the unique identifier for each publication, `year` is the year of publication, and `label` indicates whether the publication node translated (True) or not (False).

Edges: Individual parquet files for each year containing edges (citing and cited publication IDs) in the following format:

```parquet
cited_publication_id: string
cited_year: int
citing_publication_id: string
citing_year: int
```

Embeddings: Parquet file in the following format:

```parquet
publication_id: string
embeddings: list of float
```

Where `publication_id` is the unique identifier for each publication and `embeddings` is a list of floats representing the SCIBERT text embeddings for that publication.

Metadata: Parquet file in the following format:

```parquet
publication_id: string
citation_count: int
ct_linked: bool
... (other metadata fields)
```

Where `publication_id` is the unique identifier for each publication and `citation_count` is the number of citations for that publication in the time-delta.