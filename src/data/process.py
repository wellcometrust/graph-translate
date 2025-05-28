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


class TranslationDataset:
    """Methods to process citation data into Pytorch geometric format."""

    def __init__(self, edges_fpath, nodes_fpath, embeddings_fpath):
        self.edges = self.load_edges(edges_fpath)
        self.nodes = self.load_source_nodes(nodes_fpath)
        self.embeddings = self.load_embeddings(embeddings_fpath)
        self.filter_edges_by_embeddings()
        self.filter_nodes_by_edges()
        self.node_mapping = {}

    @staticmethod
    def load_edges(path):
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

    @staticmethod
    def load_source_nodes(path):
        """Load source publication nodes.

        Args:
            path(str): s3 path to source publication data.

        Returns:
            pd.DataFrame: Source nodes data.

        """
        print("Loading nodes...")
        nodes = wr.s3.read_parquet(path)
        nodes["label"] = nodes["label"].astype(int)
        return nodes

    @staticmethod
    def load_embeddings(path):
        """Load text embeddings from s3.

        Args:
            path(str): s3 path to text embeddings for all publications including citations.

        Returns:
            dict: Mapping of publication ID to embedding.

        """
        print("Loading embeddings...")
        embeddings = wr.s3.read_parquet(path)
        return embeddings.set_index("publication_id")["embeddings"].to_dict()

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
            """ Generate node masks for training, validation, and testing.

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
        return torch.tensor([citing_idx, cited_idx])

    def to_pyg(self):
        """Create a pytorch geometric dataset from source data extracted from WAG.
        This creates node features from embeddings, an edge index from citing node indices to
        cited node indices, and a label indicating whether the node was cited by a clinical trial or patent.
        Training, validation, and test masks are added to the labelled data, if provided.

        Returns:
            torch_geometric.data.Data: Pytorch geometric Data object describing the translation graph.

        """
        self.collate_nodes_from_edges()
        self.node_mapping = {
            nid: idx for idx, nid in enumerate(self.nodes["node_id"].tolist())
        }
        x = torch.tensor(
            np.array(
                self.nodes["dimensions_publication_id"].map(self.embeddings).tolist()
            ).astype(np.float32)
        )
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

        return data
