from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data as PygData

from graphpfn.util import TaskType


class MaskArrays(TypedDict):
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


FEATURE_KEYS: list[Literal["num", "frac", "cat", "other"]] = [
    "num",
    "frac",
    "cat",
    "other",
]


class FeatureArrays(TypedDict):
    num: NotRequired[np.ndarray]  # numerical features
    frac: NotRequired[np.ndarray]  # fractional features
    cat: NotRequired[np.ndarray]  # categorical features
    other: NotRequired[np.ndarray]  # other features (neural/sparse embeddings, ...)


@dataclass
class GraphDataset:
    name: str
    graph: dgl.DGLGraph
    features: FeatureArrays
    targets: np.ndarray
    masks: MaskArrays
    task_type: TaskType

    @property
    def n_nodes(self) -> int:
        return self.graph.num_nodes()

    @property
    def n_features(self) -> int:
        n_features = 0
        for _, array in self.features.items():
            assert isinstance(array, np.ndarray)
            assert array.ndim == 2
            n_features += array.shape[1]
        return n_features

    @property
    def n_classes(self) -> int | None:
        if self.task_type == "regression":
            return None
        assert self.task_type in ["binclass", "multiclass"]
        return len(np.unique(self.targets[~np.isnan(self.targets)]))

    @classmethod
    def from_pyg(
        cls,
        pyg_dataset: PygData,
        task_type: TaskType,
        *,
        name: str,
        split_index: int = 0,
    ) -> "GraphDataset":
        other_features: np.ndarray = pyg_dataset.x.numpy().astype(np.float32)  # type: ignore
        features = {"other": other_features}

        targets: np.ndarray = pyg_dataset.y.squeeze().numpy().astype(np.float32)  # type: ignore

        masks = {
            part: getattr(pyg_dataset, f"{part}_mask")[:, split_index].numpy()
            for part in ["train", "val", "test"]
        }

        edges: Tensor = pyg_dataset.edge_index  # type: ignore
        graph = dgl.graph(
            data=(edges[0], edges[1]),
            num_nodes=targets.shape[0],
            idtype=torch.int32,
        )

        return GraphDataset(
            name=name,
            graph=graph,
            features=cast(FeatureArrays, features),
            targets=targets,
            masks=cast(MaskArrays, masks),
            task_type=task_type,
        )

    @classmethod
    def from_graphland(
        cls,
        name: str,
        split: Literal["RL", "RH", "TH"] = "RL",
        root_dir: str | Path = "data",
    ) -> "GraphDataset":
        path = Path(root_dir).resolve() / name

        # >>> Load info
        with open(path / "info.yaml") as file:
            info = yaml.safe_load(file)

        # >>> Load features
        features_df = pd.read_csv(path / "features.csv", index_col=0).astype(np.float32)

        # >>> Drop constant features
        features_df = features_df.loc[:, features_df.apply(pd.Series.nunique) != 1]
        columns_remained = list(features_df.columns)

        # >>> Load labels
        targets_df = pd.read_csv(path / "targets.csv", index_col=0).astype(np.float32)
        labels = targets_df.values.squeeze().astype(np.float32)

        # >>> Load & prepare data split
        masks_df = pd.read_csv(
            path / f"split_masks_{split}.csv",
            index_col=0,
        ).astype(bool)
        masks: MaskArrays = {
            part: masks_df[part].values for part in ["train", "val", "test"]
        }  # type: ignore
        mask_labeled = ~np.isnan(labels)
        for part in ["train", "val", "test"]:
            masks[part] = masks[part] & mask_labeled

        # >> Separate features of different types
        # NOTE: fraction features in GraphLand are treated as numerical features
        frac_features_names = [
            feature_name
            for feature_name in info["fraction_features_names"]
            if feature_name in columns_remained
        ]
        frac_features = (
            features_df.loc[:, frac_features_names].values.astype(np.float32)
            if frac_features_names
            else None
        )
        if frac_features is not None:
            assert not np.isnan(frac_features).any().item(), (
                "Fraction features can not contain nans"
            )

        # NOTE: categorical features in GraphLand do not contain nans
        # so casting to integer dtype should not throw exceptions
        cat_features_names = [
            feature_name
            for feature_name in info["categorical_features_names"]
            if feature_name in columns_remained
        ]
        cat_features = (
            features_df.loc[:, cat_features_names].values.astype(np.int32)
            if cat_features_names
            else None
        )

        num_features_names = [
            feature_name
            for feature_name in info["numerical_features_names"]
            if feature_name not in frac_features_names
            and feature_name in columns_remained
        ]
        num_features = (
            features_df.loc[:, num_features_names].values.astype(np.float32)
            if num_features_names
            else None
        )
        features: FeatureArrays = {}
        if num_features is not None:
            features["num"] = num_features
        if frac_features is not None:
            features["frac"] = frac_features
        if cat_features is not None:
            features["cat"] = cat_features

        # >>> Construct graph
        edges_df = pd.read_csv(path / "edgelist.csv")
        edges = torch.from_numpy(edges_df.values[:, :2]).T
        graph = dgl.graph(
            data=(edges[0], edges[1]),
            num_nodes=len(labels),
            idtype=torch.int32,
        )

        # >>> Task type
        task_type: TaskType = {
            "binary_classification": "binclass",
            "multiclass_classification": "multiclass",
            "regression": "regression",
        }[info["task"]]  # type: ignore

        return GraphDataset(
            name=name,
            graph=graph,
            features=features,
            targets=labels,
            masks=masks,
            task_type=task_type,
        )
