from pathlib import Path

import awswrangler as wr
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.decomposition import PCA


class TranslationDataset:
    """Methods to process citation data into Pytorch geometric format."""

    def __init__(self, edges_fpath, nodes_fpath, embeddings_fpath, metadata_fpath=None, features=None):
        self.nodes = self.load_source_nodes(nodes_fpath)
        self.edges = self.load_edges(edges_fpath)
        self.embeddings = self.load_embeddings(embeddings_fpath)

        self.filter_edges_by_embeddings()
        self.filter_nodes_by_edges()
        self.features = features

        if self.features:
            self.metadata = self.load_metadata(metadata_fpath, features)
            self.filter_nodes_by_metadata()
            self.filter_edges_by_metadata()
        self.node_mapping = {}

    def load_edges(self, path):
        """Load citation edges from s3.

        Args:
            path(str): s3 path to citation edges, saved by year.

        Returns:
            dict: Citation edges by year of source publications.

        """
        print("Loading edges...")
        edges_by_year = {}
        for f in tqdm(wr.s3.list_objects(path)):
            edges = wr.s3.read_parquet(f)
            edges = edges.drop_duplicates()
            
            edges_by_year[int(Path(f).stem)] = edges
        return edges_by_year

    def load_source_nodes(self, path):
        """Load source publication nodes.

        Args:
            path(str): s3 path to source publication data.

        Returns:
            pd.DataFrame: Source nodes data.

        """
        print("Loading nodes...")
        files = wr.s3.list_objects(path)
        nodes = wr.s3.read_parquet(files)
        nodes["label"] = nodes["label"].astype(int)
        nodes["year"] = nodes["year"].astype(int)

        return nodes

    def load_embeddings(self, path):
        """Load text embeddings from s3.

        Args:
            path(str): s3 path to text embeddings for all publications including citations.

        Returns:
            dict: Mapping of publication ID to embedding.

        """
        print("Loading embeddings...")
        embeddings = wr.s3.read_parquet(path)

        return embeddings.set_index("publication_id")["embeddings"].to_dict()
    
    @staticmethod
    def load_metadata(path, features):
        """Load metadata from s3.
        
        Args:
            path(str): s3 path to metadata.
            
        Returns:
            pd.DataFrame: Metadata.
        
        """
        print("Loading metadata...")
        metadata = wr.s3.read_parquet(path)
        metadata['citation_count'].fillna(-1, inplace=True)
        metadata['ct_linked'].fillna(0, inplace=True)

        if features:
            feature_cols = list(metadata.columns[metadata.columns.str.startswith(tuple(features))])
            cols = feature_cols + ['dimensions_publication_id']
            metadata = metadata[cols].dropna()

            metadata = pd.DataFrame({
                        'features': list(metadata[feature_cols].to_numpy()),
                        'dimensions_publication_id': metadata['dimensions_publication_id']
                    })
            
        return metadata.set_index("dimensions_publication_id")['features'].to_dict()

    def filter_nodes_by_edges(self):
        """Filter out any nodes that do not have citation data."""
        print("Filtering nodes by edges...")
        filtered_nodes = []
        for year, df in self.edges.items():
            print(year)
            nodes_by_year = self.nodes[
                (self.nodes["year"] == year)
                & (
                    self.nodes["dimensions_publication_id"].isin(
                        df["cited_dimensions_publication_id"].tolist()
                    )
                )
            ]
            filtered_nodes.append(nodes_by_year)
        self.nodes = pd.concat(filtered_nodes, ignore_index=True)

    def filter_edges_by_embeddings(self):
        """Filter out any citation edges where one or both publication IDs do not have text embeddings."""
        
        print("Filtering edges by embeddings...")
        for year, df in self.edges.items():
            print(year)
            for c in ["citing", "cited"]:
                df = df[
                    df[f"{c}_dimensions_publication_id"].isin(self.embeddings.keys())
                ]
            self.edges[year] = df

    def filter_edges_by_metadata(self):
        """Filter out any citation edges where one or both publications IDs do not have metadata features."""

        print ("Filtering edges by metadata...")
        for year, df in self.edges.items():
            print (year)
            for c in ["cited", "citing"]:
                df = df[
                    df[f"{c}_dimensions_publication_id"].isin(self.metadata.keys())
                ]
            self.edges[year] = df

    def filter_nodes_by_metadata(self):
        """Filter out any nodes where one or both publications IDs do not have metadata features."""

        self.nodes = self.nodes[self.nodes["dimensions_publication_id"].isin(self.metadata.keys())]

    def filter_edges_by_nodes(self):
        """Filter out obsolete edges which are not attached to any source nodes."""
        for year in self.nodes["year"].unique():
            node_ids = self.nodes.loc[
                self.nodes["year"] == year, "dimensions_publication_id"
            ].tolist()
            l1 = self.edges[year][
                self.edges[year]["cited_dimensions_publication_id"].isin(node_ids)
            ]

            l1_ids = l1["citing_dimensions_publication_id"].tolist()
            l2 = self.edges[year][
                self.edges[year]["cited_dimensions_publication_id"].isin(l1_ids)
            ]

            self.edges[year] = pd.concat([l1, l2]).drop_duplicates()

    def split_data(self, val_size, test_size, stratify_by, random_state):
        """Perform train, validation, test splits.

        Args:
            val_size(float, int): Size of validation set to be passed to sklearn train-test-split.
            test_size(float, int): Size of test set to be passed to sklearn train-test-split.
            stratify_by(str, list, optional): Column(s) to stratify data splits by.
            random_state(int): Random seed for sklearn train-test-split.

        Returns:
            pd.DataFrame: Training features
            pd.DataFrame: Test features
            pd.DataFrame: Validation features
            pd.Series: Training label
            pd.Series: Test label
            pd.Series: Validation label

        """
        if isinstance(stratify_by, list):
            composite_colname = "_".join(stratify_by)
            self.nodes[composite_colname] = ""
            for c in stratify_by:
                self.nodes[composite_colname] = (
                    self.nodes[composite_colname] + "_" + self.nodes[c].astype(str)
                )
            stratify_by = composite_colname

        y = self.nodes["label"]
        X = self.nodes.drop("label", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=X[stratify_by],
            random_state=random_state,
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=val_size / (val_size + test_size),
            stratify=X_test[stratify_by],
            random_state=random_state,
        )
        return X_train, X_test, X_val, y_train, y_test, y_val

    @staticmethod
    def resample(X, y, stratify_by_col, mode, random_state):
        """Perform random over- or undersampling on a given data split.

        Args:
            X(pd.DataFrame): Features.
            y(pd.Series): Label.
            stratify_by(str, optional): Column to stratify data splits by (optional).
            mode(str): Sampling mode, either "upsample" or "downsample".
            random_state(int): Random seed for sampler.

        Returns:
            pd.DataFrame: Resampled features.
            pd.Series: Resampled labels.

        """
        if mode == "upsample":
            sampler = RandomOverSampler(random_state=random_state)
        elif mode == "downsample":
            sampler = RandomUnderSampler(random_state=random_state)
        if stratify_by_col is not None:
            resampled_x = []
            resampled_y = []
            for _, data_x in X.groupby(stratify_by_col):
                data_y = y[data_x.index]
                X_resampled, y_resampled = sampler.fit_resample(data_x, data_y)
                resampled_x.append(X_resampled)
                resampled_y.append(y_resampled)
            resampled_x, resampled_y = pd.concat(resampled_x), pd.concat(resampled_y)
        else:
            resampled_x, resampled_y = sampler.fit_resample(X, y)

        return resampled_x, resampled_y

    def generate_node_masks(
        self,
        val_size=0.1,
        test_size=0.1,
        stratify_by=["label", "year"],
        resample_train="downsample",
        random_state=123,
    ):
        X_train, X_test, X_val, y_train, y_test, y_val = self.split_data(
            val_size=val_size,
            test_size=test_size,
            stratify_by=stratify_by,
            random_state=random_state,
        )
        if resample_train is not None:
            if isinstance(stratify_by, list):
                stratify_by.remove("label")
                if len(stratify_by) == 0:
                    stratify_by = None
            elif stratify_by == "label":
                stratify_by = None
            X_train, y_train = self.resample(
                X_train,
                y_train,
                stratify_by_col=stratify_by,
                mode=resample_train,
                random_state=random_state,
            )
            """Generate node masks for training, validation, and testing.

            Args:
                val_size(float, int): Size of validation set to be passed to sklearn train-test-split.
                test_size(float, int): Size of test set to be passed to sklearn train-test-split.
                stratify_by(str, list, optional): Column(s) to stratify data splits by.
                resample_train(str, optional): Whether to balance the training set (optional). Can be "downsample", "upsample", or None.
                random_state(int): Random seed for sklearn train-test-split.

            Returns:
                pd.DataFrame: Training features
                pd.DataFrame: Test features
                pd.DataFrame: Validation features
                pd.Series: Training label
                pd.Series: Test label
                pd.Series: Validation label

            """

        df_train = pd.concat([X_train, y_train], axis=1)
        df_train["mask"] = "train"
        df_test = pd.concat([X_test, y_test], axis=1)
        df_test["mask"] = "test"
        df_val = pd.concat([X_val, y_val], axis=1)
        df_val["mask"] = "val"

        self.nodes = pd.concat([df_train, df_test, df_val])
        self.filter_edges_by_nodes()

    def collate_nodes_from_edges(self):
        """Add neighbour nodes which only appear in the edgelist to the overall nodes dataset. Each node is given a new ID
        before merging, which constists of the publication year of the seed nodes and the publication ID.

        """
        self.nodes["node_id"] = (
            self.nodes["dimensions_publication_id"]
            + "_"
            + self.nodes["year"].astype(str)
        )
        nodes_from_edges = []
        for year, edges_df in self.edges.items():
            for c in ["citing", "cited"]:
                edges_df[f"{c}_id"] = (
                    edges_df[f"{c}_dimensions_publication_id"] + "_" + str(year)
                )
                nodes_from_edges.append(
                    edges_df[[f"{c}_id", f"{c}_dimensions_publication_id"]].rename(
                        columns={
                            f"{c}_id": "node_id",
                            f"{c}_dimensions_publication_id": "dimensions_publication_id",
                        }
                    )
                )
        nodes_from_edges = pd.concat(nodes_from_edges).drop_duplicates()
        self.nodes = nodes_from_edges.merge(self.nodes, how="left")
        self.nodes.drop_duplicates(inplace=True)

    def _get_edge_index(self):
        """Generate an edge index tensor (from citing node index to cited node index).

        Returns:
            torch.Tensor: Edge index.

        """
        cited_idx = []
        citing_idx = []
        for df in self.edges.values():
            cited_idx.extend(df["cited_id"].map(self.node_mapping).tolist())
            citing_idx.extend(df["citing_id"].map(self.node_mapping).tolist())

        edges = torch.tensor([citing_idx, cited_idx])
        mask = ~torch.isnan(edges[0, :]) & ~torch.isnan(edges[1, :])

        return edges[:, mask]
    
    def _downsample(self, downsample):
        """Downsample size of the dataset."""

        if downsample > 1:
            downsample = downsample/self.nodes.shape[0]
            
        self.nodes = self.nodes.groupby(["year", "label"], group_keys=False).apply(
             lambda x: x.sample(frac=downsample, random_state=123, replace=True)
         )

        unique_nodes = list(self.nodes['dimensions_publication_id'])
        
        for year, edges in self.edges.items():
            first_order_citations = list(edges[edges['cited_dimensions_publication_id'].isin(unique_nodes)]['citing_dimensions_publication_id'])
            edges = edges[edges['cited_dimensions_publication_id'].isin(unique_nodes + first_order_citations)]
            self.edges[year] = edges

    def to_pyg(self, downsample=False):
        """Create a pytorch geometric dataset from source data extracted from WAG.
        This creates node features from embeddings, an edge index from citing node indices to
        cited node indices, and a label indicating whether the node was cited by a clinical trial or patent.
        Training, validation, and test masks are added to the labelled data, if provided.

        Returns:
            torch_geometric.data.Data: Pytorch geometric Data object describing the translation graph.

        """
        if downsample:
            self._downsample(downsample)

        self.collate_nodes_from_edges()

        self.node_mapping = {
            nid: idx for idx, nid in enumerate(self.nodes["node_id"].tolist())
        }
        x = torch.tensor(
            np.array(
                self.nodes["dimensions_publication_id"].map(self.embeddings).tolist()
            ).astype(np.float32)
        )

        if self.features:
            x_features = torch.tensor(
                np.array(
                    self.nodes["dimensions_publication_id"].map(self.metadata).tolist()
                ).astype(np.float32)
            )

            if x_features.shape[1] > 32:
                pca = PCA(n_components=32)
                x_features = pca.fit_transform(x_features)
                x_features = torch.tensor(x_features.astype(np.float32))

            x = torch.cat((x, x_features), dim=1)
        
        num_classes = len(self.nodes["label"].dropna().unique())

        y = torch.tensor(np.array(self.nodes["label"].tolist()).astype(np.float32))

        edge_index = self._get_edge_index()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_classes = num_classes

        data.weight = (y == 0).sum() / (y == 1).sum()

        if "mask" in self.nodes.columns:
            data.train_mask = torch.tensor(np.array(self.nodes["mask"] == "train"))
            data.val_mask = torch.tensor(np.array(self.nodes["mask"] == "val"))
            data.test_mask = torch.tensor(np.array(self.nodes["mask"] == "test"))

        return data, pd.DataFrame(list(self.node_mapping.items()), columns=['node_id', 'index'])
