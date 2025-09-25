import enum
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Generic, Literal, TypedDict, TypeVar, cast

import dgl
import numpy as np
import pandas as pd
import scipy.stats as sps
import sklearn.impute
import sklearn.preprocessing
import torch
import yaml
from ogb.nodeproppred import DglNodePropPredDataset as OGBDataset
from loguru import logger
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric import datasets as pyg_datasets
from torch_geometric.data import Data as PygData

from lib.data import _SCORE_SHOULD_BE_MAXIMIZED
from lib.graph.util import Setting
from lib.graph import structure_encodings
from lib.metrics import calculate_metrics as calculate_metrics_
from lib.util import DATA_PARTS, PartKey, PredictionType, Score, TaskType


class GraphLandInfo(TypedDict):
    dataset_name: str
    task: str
    metric: str
    num_features_names: list[str]
    cat_features_names: list[str]
    proportion_features_names: list[str]
    target_name: str
    graph_is_directed: bool
    graph_is_weighted: bool
    has_unlabeled_nodes: bool
    has_nans_in_num_features: bool


GRAPHLAND_DATASETS = [
    "tolokers-2",
    "city-reviews",
    "artnet-exp",
    "web-fraud",
    "hm-categories",
    "pokec-regions",
    "web-topics",
    "hm-prices",
    "avazu-ctr",
    "city-roads-M",
    "city-roads-L",
    "twitch-views",
    "artnet-views",
    "web-traffic",
]

PYG_DATASETS = [
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
    "cora",
    "citeseer",
    "pubmed",
    "coauthor-cs",
    "coauthor-physics",
    "amazon-computers",
    "amazon-photo",
    "lastfm-asia",
    "facebook",
    "wiki-cs",
]

OGB_DATASETS = ["ogbn-arxiv", "ogbn-products"]

MULTICLASS_DATASETS = [
    "browser-games",
    "hm-categories",
    "roman-empire",
    "amazon-ratings",
    "cora",
    "citeseer",
    "pubmed",
    "coauthor-cs",
    "coauthor-physics",
    "amazon-computers",
    "amazon-photo",
    "lastfm-asia",
    "facebook",
    "ogbn-arxiv",
    "ogbn-products",
    "hm-categories",
    "pokec-regions",
    "web-topics",
    "wiki-cs",
]

BINCLASS_DATASETS = [
    "city-reviews",
    "web-fraud",
    "minesweeper",
    "tolokers",
    "questions",
    "tolokers-2",
    "city-reviews",
    "artnet-exp",
    "web-fraud",
]

REGRESSION_DATASETS = [
    "city-roads-M",
    "city-roads-L",
    "avazu-devices",
    "hm-prices",
    "web-traffic",
    "hm-prices",
    "avazu-ctr",
    "city-roads-M",
    "city-roads-L",
    "twitch-views",
    "artnet-views",
    "web-traffic",
]

PREDEFINED_SPLIT_DATASETS = [
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
    *GRAPHLAND_DATASETS,
]

HOMOGENEOUS_DATASETS = [
    *PYG_DATASETS,
    *OGB_DATASETS,
]

HETEROGENEOUS_DATASETS = [
    *GRAPHLAND_DATASETS,
]


def load_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as file:
        d = yaml.safe_load(file)
    return d


def _mask_labeled_nodes(labels: np.ndarray, masks: dict[PartKey, np.ndarray]) -> None:
    mask_labeled = ~np.isnan(labels)
    for part in DATA_PARTS:
        masks[part] = masks[part] & mask_labeled


GraphDataKey = str


class GraphData(TypedDict):
    name: str
    graph: dgl.DGLGraph
    labels: np.ndarray
    masks: dict[PartKey, np.ndarray]
    num_features: np.ndarray  # require scaling
    cat_features: None | np.ndarray  # require encoding
    ratio_features: None | np.ndarray  # do not require any processing


def _load_graphland_data(
    path: str | Path,
    *,
    internal_split_name: Literal["RL", "RH", "TH"],
    **kwargs,
) -> GraphData:
    del kwargs
    path = Path(path).resolve()
    name = path.name

    # >>> Load info
    info_path = path / "info.yaml"
    info = cast(GraphLandInfo, load_dict(info_path))

    # >>> Load features
    features_path = path / "features.csv"
    features_df = pd.read_csv(features_path, index_col=0).astype(np.float32)

    # >>> Drop constant features
    features_df = features_df.loc[:, features_df.apply(pd.Series.nunique) != 1]
    columns_remained = list(features_df.columns)

    # >>> Load labels
    labels_path = path / "targets.csv"
    labels = pd.read_csv(labels_path, index_col=0).astype(np.float32)
    labels = labels.iloc[:, 0].values

    # >>> Load & prepare data split
    split_path = path / f"split_masks_{internal_split_name}.csv"
    split_masks = pd.read_csv(split_path, index_col=0).astype(bool)
    masks: dict[PartKey, np.ndarray] = {
        part: split_masks[f"{part}"].values for part in DATA_PARTS
    }
    _mask_labeled_nodes(labels, masks)

    # >> Separate features of different types
    # NOTE: ratio features in GraphLand are treated as numerical features
    ratio_features_names = [
        feature_name
        for feature_name in info["fraction_features_names"]
        if feature_name in columns_remained
    ]
    ratio_features = (
        features_df.loc[:, ratio_features_names].values.astype(np.float32)
        if ratio_features_names
        else None
    )
    if ratio_features is not None:
        assert not np.isnan(ratio_features).any().item(), (
            "Ratio features can not contain nans"
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
        if feature_name not in ratio_features_names and feature_name in columns_remained
    ]
    num_features = (
        features_df.loc[:, num_features_names].values.astype(np.float32)
        if num_features_names
        else None
    )

    # >>> Construct graph
    edges_path = path / "edgelist.csv"
    edges_df = pd.read_csv(edges_path)
    edges = torch.from_numpy(edges_df.values[:, :2]).T
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=len(labels),
        idtype=torch.int32,
    )

    graph_data = {
        "name": name,
        "graph": graph,
        "labels": labels,
        "masks": masks,
        "num_features": num_features,
        "cat_features": cat_features,
        "ratio_features": ratio_features,
    }
    return cast(GraphData, graph_data)


def _load_pyg_data(
    path: str | Path,
    *,
    external_split: bool = True,
    internal_split_index: None | int = None,
    **kwargs,
) -> GraphData:
    del kwargs
    assert external_split ^ (internal_split_index is not None), (
        "Specifying `internal_split_index` exludes `external_split` set to True"
    )

    path = Path(path).resolve()
    name = path.name
    root_ = str(path)

    if name in [
        "roman-empire",
        "amazon-ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ]:
        dataset_ = pyg_datasets.HeterophilousGraphDataset(name=name, root=root_)

    elif name in ["cora", "citeseer", "pubmed"]:
        dataset_ = pyg_datasets.Planetoid(name=name, root=root_)

    elif name in ["coauthor-cs", "coauthor-physics"]:
        dataset_ = pyg_datasets.Coauthor(name=name.split("-")[1], root=root_)

    elif name in ["amazon-computers", "amazon-photo"]:
        dataset_ = pyg_datasets.Amazon(name=name.split("-")[1], root=root_)

    elif name == "lastfm-asia":
        dataset_ = pyg_datasets.LastFMAsia(root=root_)

    elif name == "facebook":
        dataset_ = pyg_datasets.FacebookPagePage(root=root_)

    elif name == "wiki-cs":
        dataset_ = pyg_datasets.WikiCS(root=root_)

    else:
        raise ValueError(f"Unknown {name=}")

    data: PygData = dataset_[0]

    labels: np.ndarray = data.y.squeeze().numpy().astype(np.float32)
    num_features: np.ndarray = data.x.numpy().astype(np.float32)
    cat_features = None
    ratio_features = None

    edges: Tensor = data.edge_index
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=len(labels),
        idtype=torch.int32,
    )

    if external_split:
        split_path = path / "split.npz"
        split_masks = np.load(split_path, allow_pickle=True)
        masks: dict[PartKey, np.ndarray] = {
            part: split_masks[part] for part in DATA_PARTS
        }

    else:
        if name in PREDEFINED_SPLIT_DATASETS:

            def _retrieve_data_split(split_index):
                return {
                    part: getattr(data, f"{part}_mask")[:, split_index].numpy()
                    for part in DATA_PARTS
                }

            masks: dict[PartKey, np.ndarray] = _retrieve_data_split(
                internal_split_index
            )

        else:

            def _generate_data_split(seed: int = 17):
                masks = {part: np.zeros(len(labels), dtype=bool) for part in DATA_PARTS}
                indices_train, indices_holdout = train_test_split(
                    np.arange(len(labels)),
                    test_size=0.5,
                    random_state=seed,
                    stratify=labels,
                )
                masks["train"][indices_train] = True

                indices_val, indices_test = train_test_split(
                    indices_holdout,
                    test_size=0.5,
                    random_state=seed,
                    stratify=labels[indices_holdout],
                )
                masks["val"][indices_val] = True
                masks["test"][indices_test] = True

                return masks

            masks: dict[PartKey, np.ndarray] = _generate_data_split()

    _mask_labeled_nodes(labels, masks)

    graph_data = {
        "name": name,
        "graph": graph,
        "labels": labels,
        "masks": masks,
        "num_features": num_features,
        "cat_features": cat_features,
        "ratio_features": ratio_features,
    }
    return cast(GraphData, graph_data)


def _load_ogb_data(
    path: str | Path,
    *,
    external_split: str = False,
    **kwargs,
) -> GraphData:
    del kwargs
    assert not external_split, "No external splits are available for OGB datasets!"

    path = Path(path).resolve()
    name = path.name
    root_ = str(path)

    dataset = OGBDataset(name, root=root_)
    graph: dgl.DGLGraph = dataset[0][0]
    features_: Tensor = graph.ndata["feat"]
    labels_: Tensor = dataset[0][1]

    labels = labels_.squeeze().numpy().astype(np.float32)
    num_features = features_.numpy().astype(np.float32)
    cat_features = None
    ratio_features = None

    del graph.ndata["feat"]
    graph = graph.int()

    data_split_: dict = dataset.get_idx_split()
    masks: dict[PartKey, np.ndarray] = dict()
    for part in DATA_PARTS:
        mask = np.zeros(len(labels), dtype=bool)
        # NOTE: quick fix for validation part name
        mask[data_split_[part if part != "val" else "valid"]] = True
        masks[part] = mask

    _mask_labeled_nodes(labels, masks)

    graph_data = {
        "name": name,
        "graph": graph,
        "labels": labels,
        "masks": masks,
        "num_features": num_features,
        "cat_features": cat_features,
        "ratio_features": ratio_features,
    }
    return cast(GraphData, graph_data)


def load_data(path: str | Path, **data_params) -> GraphData:
    """
    Load data, drop constant categorical features,
    Transform binary ratio features to categorical, if applicable.
    """

    path = Path(path).resolve()
    name = path.name

    if name in GRAPHLAND_DATASETS:
        _load_data_fn = _load_graphland_data

    elif name in PYG_DATASETS:
        _load_data_fn = _load_pyg_data

    elif name in OGB_DATASETS:
        _load_data_fn = _load_ogb_data

    else:
        raise ValueError(f"Unknown {name=}")

    data = _load_data_fn(path, **data_params)

    cat_features = data["cat_features"]
    if cat_features is not None:
        # >>> Drop constant features
        mask = np.array([len(np.unique(x)) > 1 for x in cat_features.T])
        assert mask.any().item()
        cat_features = cat_features[:, mask]
        data["cat_features"] = cat_features

    ratio_features = data["ratio_features"]
    if ratio_features is not None:
        # >>> Transform binary ratio features to categorical
        mask = np.array([np.all((x == 0.0) | (x == 1.0)) for x in ratio_features.T])
        if mask.any().item():
            binary_ratio_features = ratio_features[:, mask].astype(np.float32)
            cat_features = data["cat_features"]
            cat_features = (
                np.concatenate([cat_features, binary_ratio_features], axis=1)
                if ratio_features is not None
                else binary_ratio_features
            )
            data["cat_features"] = cat_features
            data["ratio_features"] = (
                None if mask.all().item() else ratio_features[:, ~mask]
            )

    # >>> Post-process graph
    graph = data["graph"]
    graph = dgl.remove_self_loop(graph)
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)
    data["graph"] = graph

    return data


@dataclass(frozen=True)
class GraphTask:
    labels: np.ndarray
    masks: dict[PartKey, np.ndarray]
    type_: TaskType
    setting: Setting
    score: Score

    @classmethod
    def from_dir(
        cls, path: str | Path, setting: str | Setting, **data_params
    ) -> "GraphTask":
        data = load_data(path, **data_params)
        task_type = _get_task_type(data["name"])
        score = _get_score(task_type)
        return GraphTask(
            labels=data["labels"],
            masks=data["masks"],
            type_=task_type,
            setting=Setting(setting),
            score=score,
        )

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    @property
    def is_transductive(self) -> bool:
        return self.setting == Setting.TRANSDUCTIVE

    def compute_n_classes(self) -> int:
        assert self.is_binclass or self.is_classification
        labels = self.labels
        labels = labels[~np.isnan(labels)]
        n_classes = len(np.unique(labels))
        return n_classes

    def try_compute_n_classes(self) -> None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(
        self,
        predictions: dict[PartKey, np.ndarray],
        prediction_type: str | PredictionType,
    ) -> dict[PartKey, Any]:
        # NOTE: such inconsistency between labels and predictions
        # makes `GraphTask.calculate_metrics` compatible with other scripts
        labels = {part: self.labels[self.masks[part]] for part in DATA_PARTS}
        metrics = {
            part: calculate_metrics_(
                labels[part], predictions[part], self.type_, prediction_type
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics["score"] = (
                1.0 if _SCORE_SHOULD_BE_MAXIMIZED[self.score] else -1.0
            ) * part_metrics[self.score.value]
        return metrics


T = TypeVar("T", np.ndarray, Tensor)


def _get_task_type(name: str) -> TaskType:
    if name in BINCLASS_DATASETS:
        return TaskType.BINCLASS

    elif name in MULTICLASS_DATASETS:
        return TaskType.MULTICLASS

    elif name in REGRESSION_DATASETS:
        return TaskType.REGRESSION

    else:
        raise ValueError(f"Unknown {name=}")


def _get_score(task_type: TaskType) -> Score:
    return {
        TaskType.BINCLASS: Score.AP,
        TaskType.MULTICLASS: Score.ACCURACY,
        TaskType.REGRESSION: Score.R2,
    }[task_type]


@dataclass
class GraphDataset(Generic[T]):  # noqa: UP046
    data: GraphData
    task: GraphTask

    @classmethod
    def from_dir(
        cls, path: str | Path, setting: str | Setting, **data_params
    ) -> "GraphDataset[np.ndarray]":
        data = load_data(path, **data_params)
        task_type = _get_task_type(data["name"])
        score = _get_score(task_type)
        task = GraphTask(
            labels=data["labels"],
            masks=data["masks"],
            type_=task_type,
            setting=Setting(setting),
            score=score,
        )
        return GraphDataset(data, task)

    @classmethod
    def from_data(
        cls,
        name: str,
        graph: dgl.DGLGraph,
        features: dict[str, np.ndarray],  # TODO: str -> FeatureType
        labels: np.ndarray,
        masks: np.ndarray,
        task_type: str | TaskType,
        setting: str | Setting = "transductive",
    ) -> "GraphDataset[np.ndarray]":
        task_type = TaskType(task_type)
        setting = Setting(setting)

        data = GraphData(
            name=name,
            graph=graph,
            labels=labels,
            masks=masks,
            num_features=features.get("num_features", None),
            ratio_features=features.get("ratio_features", None),
            cat_features=features.get("cat_features", None),
        )

        task = GraphTask(
            labels=labels,
            masks=masks,
            type_=task_type,
            setting=setting,
            score=_get_score(task_type),
        )

        return GraphDataset(data, task)

    def _is_numpy(self) -> bool:
        for features_type in ["num_features", "ratio_features", "cat_features"]:
            if self.data.get(features_type, None) is not None:
                return isinstance(self.data[features_type], np.ndarray)
        raise ValueError("Cannot infer features type: no features available")

    @property
    def is_heterogeneous(self) -> bool:
        return self.data["name"] in HETEROGENEOUS_DATASETS

    @property
    def n_num_features(self) -> int:
        # NOTE: homogeneous features are treated as numerical
        num_features = self.data.get("num_features", None)
        return num_features.shape[1] if num_features is not None else 0

    @property
    def n_cat_features(self) -> int:
        if not self.is_heterogeneous:
            return 0
        cat_features = self.data.get("cat_features", None)
        return cat_features.shape[1] if cat_features is not None else 0

    @property
    def n_ratio_features(self) -> int:
        if not self.is_heterogeneous:
            return 0
        ratio_features = self.data.get("ratio_features", None)
        return ratio_features.shape[1] if ratio_features is not None else 0

    @property
    def n_features(self) -> int:
        return (
            (self.n_num_features + self.n_cat_features + self.n_ratio_features)
            if self.is_heterogeneous
            else self.n_num_features
        )

    def compute_cat_cardinalities(self) -> list[int]:
        cat_features = self.data.get("cat_features")
        if cat_features is None:
            return []
        unique = np.unique if self._is_numpy() else torch.unique
        return [
            len(unique(column))
            for column in cat_features[self.data["masks"]["train"]].T
        ]

    def size(self, part: None | PartKey = None) -> int:
        return (
            self.data["masks"][part].sum().item()
            if part is not None
            else len(self.data["labels"])
        )

    def to_torch(self, device: None | str | torch.device) -> "GraphDataset[Tensor]":
        data_casted = cast(GraphData, dict())
        for key, value in self.data.items():
            data_casted[key] = (
                {
                    subkey: torch.as_tensor(subvalue).to(device)
                    for subkey, subvalue in value.items()
                }
                if isinstance(value, dict)
                else torch.as_tensor(value).to(device)
                if isinstance(value, np.ndarray)
                else value.to(device)
                if isinstance(value, dgl.DGLGraph)
                else value
            )
        return GraphDataset(data_casted, self.task)


class NumPolicy(enum.Enum):
    STANDARD = "standard"
    QUANTILE_NORMAL = "quantile-normal"
    QUANTILE_UNIFORM_ALL_DATA = "quantile-uniform-all-data"
    NOISY_QUANTILE_NORMAL = "noisy-quantile-normal"
    NOISY_QUANTILE_UNIFORM = "noisy-quantile-uniform"
    MIN_MAX = "min-max"
    NONE = "none"


def transform_num_features(
    features: np.ndarray | None,
    masks: dict[PartKey, np.ndarray],
    setting: str | Setting,
    policy: None | str | NumPolicy,
    seed: None | int,
) -> np.ndarray:
    if features is None:
        return None

    setting = Setting(setting)
    mask_seen = (
        np.ones_like(masks["train"], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks["train"]
    )

    if (policy is not None) and (NumPolicy(policy) != NumPolicy.NONE):
        policy = NumPolicy(policy)  # throws exception in case of unknown value
        # NOTE: reserve a separate variable to avoid modifications in the original features
        features_seen = features[mask_seen]

        if policy == NumPolicy.STANDARD:
            normalizer = sklearn.preprocessing.StandardScaler()

        elif policy in {
            NumPolicy.NOISY_QUANTILE_NORMAL,
            NumPolicy.NOISY_QUANTILE_UNIFORM,
        }:
            assert seed is not None
            if policy == NumPolicy.NOISY_QUANTILE_NORMAL:
                output_distribution = "normal"
            elif policy == NumPolicy.NOISY_QUANTILE_UNIFORM:
                output_distribution = "uniform"
            else:
                raise ValueError()

            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(features_seen.shape[0] // 30, 1000), 10),
                output_distribution=output_distribution,
                subsample=1_000_000_000,
                random_state=seed,
            )
            features_seen = features_seen + np.random.RandomState(seed).normal(
                0.0, 1e-5, features_seen.shape
            ).astype(features_seen.dtype)

        elif policy == NumPolicy.MIN_MAX:
            normalizer = sklearn.preprocessing.MinMaxScaler()

        elif policy == NumPolicy.QUANTILE_NORMAL:
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution="normal", subsample=None
            )

        elif policy == NumPolicy.QUANTILE_UNIFORM_ALL_DATA:
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=max(features.shape[0] // 5, 2),
                subsample=None,
            )

        else:
            raise ValueError()

        normalizer.fit(features_seen)
        features = normalizer.transform(features)

    # >>> Impute nans
    imputer = sklearn.impute.SimpleImputer(strategy="most_frequent")
    imputer.fit(features[mask_seen])
    features = imputer.transform(features)

    return features


class CatPolicy(enum.Enum):
    ORDINAL = "ordinal"
    ONE_HOT = "one-hot"


def transform_cat_features(
    features: np.ndarray | None,
    masks: dict[PartKey, np.ndarray],
    setting: str | Setting,
    policy: None | str | CatPolicy,
) -> np.ndarray:
    if features is None:
        return None

    if policy is None:
        return features

    setting = Setting(setting)
    mask_seen = (
        np.ones_like(masks["train"], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks["train"]
    )

    policy = CatPolicy(policy)  # throws exception in case of unknown value
    unknown_value = -1

    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        dtype=np.float32,
    ).fit(features[mask_seen])
    features_encoded = encoder.transform(features)

    if policy == CatPolicy.ORDINAL:
        return features_encoded

    if policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )

    encoder.fit(features_encoded[mask_seen])
    features_transformed = encoder.transform(features_encoded)
    return features_transformed


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


def standardize_labels(
    labels: np.ndarray,
    masks: dict[PartKey, np.ndarray],
) -> tuple[np.ndarray, RegressionLabelStats]:
    assert labels.dtype == np.float32

    labels_seen = labels[masks["train"]]
    mean = float(labels_seen.mean())
    std = float(labels_seen.std())

    labels_standardized = (labels - mean) / std
    regression_stats = RegressionLabelStats(mean=mean, std=std)
    return labels_standardized, regression_stats


def shuffle_categories(cat_features: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled_cat_features = []

    for i in range(cat_features.shape[1]):
        features = cat_features[:, i]
        categories = np.unique(features)
        permuted_categories = rng.permutation(categories)
        shuffle = {
            categories[j]: permuted_categories[j] for j in range(categories.shape[0])
        }
        shuffled_cat_features.append(
            np.array([shuffle[features[j]] for j in range(features.shape[0])])
        )

    return np.stack(shuffled_cat_features, axis=-1)


def _nfa_reduce_single_hop(
    g: dgl.DGLGraph,
    x: np.ndarray,
    *,
    mode: Literal["mean", "max", "min"],
    hop: int,
) -> np.ndarray:
    x = torch.tensor(x)

    not_nan_mask = ~torch.isnan(x)
    not_nan_size = not_nan_mask.float()
    for _ in range(hop):
        not_nan_size = dgl.ops.copy_u_sum(g, not_nan_size)

    neutral_value = {
        "mean": 0.0,
        "max": -torch.inf,
        "min": +torch.inf,
    }[mode]

    operation = {
        "mean": dgl.ops.copy_u_sum,
        "max": dgl.ops.copy_u_max,
        "min": dgl.ops.copy_u_min,
    }[mode]

    denominator = {
        "mean": not_nan_size,
        "max": (not_nan_size > 0.0).float(),
        "min": (not_nan_size > 0.0).float(),
    }[mode]

    numerator = torch.where(not_nan_mask, x, neutral_value)
    for _ in range(hop):
        numerator = operation(g, numerator)

    x_reduced = torch.where(denominator != 0.0, numerator / denominator, torch.nan)
    return x_reduced.cpu().numpy()


def _nfa_reduce(
    g: dgl.DGLGraph,
    x: np.ndarray,
    *,
    mode: Literal["mean", "max", "min"],
    weights: list[float],
) -> np.ndarray:
    x_reduced = [x]
    for hop in range(1, len(weights)):
        x_reduced.append(_nfa_reduce_single_hop(g, x, mode=mode, hop=hop))
    return (np.array(weights) * np.stack(x_reduced, axis=-1)).sum(-1).astype(np.float32)


def apply_nfa(
    dataset: GraphDataset,
    weights: list[float] = [0.0, 1.0],
    num_modes: list[str] = ["mean", "max", "min"],
) -> dict[str, np.ndarray | None]:
    nfa: dict[str, list[Tensor]] = {"num_features": [], "ratio_features": []}
    graph = dataset.data["graph"]

    if dataset.data[f"num_features"] is not None:
        for mode in num_modes:
            nfa["num_features"].append(
                _nfa_reduce(
                    graph, dataset.data["num_features"], mode=mode, weights=weights
                )
            )

    if dataset.data["ratio_features"] is not None:
        nfa["ratio_features"].append(
            _nfa_reduce(
                graph, dataset.data["ratio_features"], mode="mean", weights=weights
            )
        )

    if dataset.data["cat_features"] is not None:
        cat_features = dataset.data["cat_features"]
        cat_features = transform_cat_features(
            cat_features, dataset.data["masks"], dataset.task.setting, "one-hot"
        )
        nfa_cat_features = _nfa_reduce(
            graph, cat_features, mode="mean", weights=weights
        )
        nfa["ratio_features"].append(nfa_cat_features)

    for k in list(nfa.keys()):
        if len(nfa[k]) == 0:
            nfa[k] = None
        else:
            nfa[k] = np.concatenate(nfa[k], axis=-1)

    return nfa


def apply_pca(
    dataset: GraphDataset,
    features_dict: dict[str, np.ndarray | None],
    dim: int | str,
    mode: str = "PCA",
    apply_random_orth: bool = False,
    postprocess: NumPolicy | str = NumPolicy.NONE,
) -> dict[str, np.ndarray | None]:
    assert dataset.task.setting == Setting.TRANSDUCTIVE

    # >>> Check if we need to apply pca
    total_n_featuers = 0
    for feature_type in features_dict.keys():
        features = features_dict.get(feature_type, None)
        if features is not None:
            total_n_featuers += features.shape[1]
    if (dim != "same") and (total_n_featuers <= dim):
        return features_dict

    # >>> Concat all features
    features_list = []
    for feature_type in list(features_dict.keys()):
        features = features_dict.pop(feature_type, None)
        features_dict[feature_type] = None
        if features is not None:
            # TODO: handle nans and one-hot
            if feature_type in ["num_features", "ratio_features"]:
                features = transform_num_features(
                    features,
                    dataset.data["masks"],
                    dataset.task.setting,
                    policy="none",
                    seed=0,
                )
            elif feature_type == "cat_features":
                transform_cat_features(
                    features,
                    dataset.data["masks"],
                    dataset.task.setting,
                    "one-hot",
                )
            features_list.append(features)

    features = np.concatenate(features_list, axis=1)

    # >>> Apply PCA itself

    # On questions dataset, PCA shows bad results if we apply it directly.
    # This comes from the fact that the first feature in this dataset is
    # essentially binary indicator, so while applying PCA to a text embedding is ok,
    # applying it to the pair (binary indicator, embedding) seems to work bad
    if dataset.data["name"] == "questions":
        assert len(np.unique(features[:, 0])) == 2, (
            "This feature for questions dataset should be binary."
        )
        features = features[:, 1:]

    if dim == "same":
        dim = features.shape[-1]

    pca = {
        "PCA": partial(PCA, n_components=dim),
        "KernelPCA": partial(KernelPCA, kernel="rbf", n_components=dim),
    }[mode]()
    features = pca.fit_transform(features)

    if apply_random_orth:
        random_orth = sps.special_ortho_group(dim=features.shape[-1]).rvs()
        assert random_orth.ndim == 2
        features = (features @ random_orth).astype(np.float32)

    features = transform_num_features(
        features,
        dataset.data["masks"],
        dataset.task.setting,
        postprocess,
        seed=42,
    )

    return {"num_features": features}


def merge_features(
    features_list: list[dict[str, np.ndarray | None] | None],
) -> dict[str, np.ndarray | None]:
    merged_features = dict()

    for features in features_list:
        if features is None:
            continue
        for k, v in features.items():
            if v is None:
                continue
            if k not in merged_features:
                merged_features[k] = v
            else:
                merged_features[k] = np.concatenate(
                    [merged_features[k], v],
                    axis=-1,
                )

    return merged_features


def compute_graph_encodings(
    dataset: GraphDataset,
    config: dict,
) -> dict[str, np.ndarray]:
    encodings_list = []

    preporcessing_policy = config.get("preprocess", None)

    for method, method_kwargs in config.items():
        if method == "preprocess":
            continue
        func = {
            "random_walk_pe": dgl.random_walk_pe,
            "lap_pe": dgl.lap_pe,
            "svd_pe": dgl.svd_pe,
            "instant_embedding": structure_encodings.compute_instant_embedding,
            "degree": structure_encodings.compute_degree_encoding,
            "pagerank": structure_encodings.compute_pagerank,
            "clustering_coefficient": structure_encodings.clustering_coefficient,
        }[method]

        encodings = func(dataset.data["graph"], **method_kwargs)
        encodings_list.append(encodings.cpu().numpy())

    graph_encodings = np.concatenate(encodings_list, axis=-1)
    if preporcessing_policy is not None:
        graph_encodings = transform_num_features(
            graph_encodings,
            dataset.data["masks"],
            dataset.task.setting,
            policy=preporcessing_policy,
            seed=42,
        )

    return {"num_features": graph_encodings}


def build_dataset(
    path: str | Path | None,
    *,
    dataset: GraphDataset | None = None,
    seed: int = 0,
    cache: bool = False,
    num_policy: None | str | NumPolicy = None,
    ratio_policy: None | str | NumPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    nfa: dict | None = None,
    pca: dict | None = None,
    feature_types=["num_features", "ratio_features", "cat_features"],
    graph_encodings: dict | None = None,
    add_self_loops: bool = False,
    **dataset_params,
) -> GraphDataset[np.ndarray]:
    assert not cache, "Cache is not implemented so far"

    if dataset is None:
        path = Path(path).resolve()
        dataset = GraphDataset.from_dir(path, **dataset_params)
    setting = Setting(dataset_params["setting"])

    nfa_features = apply_nfa(dataset, **nfa) if (nfa is not None) else dict()
    original_features = {key: dataset.data.pop(key, None) for key in feature_types}

    for features in [original_features, nfa_features]:
        features["num_features"] = transform_num_features(
            features.get("num_features", None),
            dataset.data["masks"],
            setting,
            num_policy,
            seed,
        )
        features["ratio_features"] = transform_num_features(
            features.get("ratio_features", None),
            dataset.data["masks"],
            setting,
            ratio_policy,
            seed,
        )
        features["cat_features"] = transform_cat_features(
            features.get("cat_features", None),
            dataset.data["masks"],
            setting,
            cat_policy,
        )
        if (pca is not None) and pca.get("apply_before_merge", True):
            new_features = apply_pca(dataset, features, **pca["params"])
            for key in feature_types:
                new_value = new_features.get(key, None)
                features.pop(key)
                features[key] = new_value

    if graph_encodings is not None:
        graph_encodings = compute_graph_encodings(dataset, config=graph_encodings)

    merged_features = merge_features([original_features, nfa_features, graph_encodings])

    if (pca is not None) and not pca.get("apply_before_merge", True):
        merged_features = apply_pca(dataset, merged_features, **pca["params"])

    for key in feature_types:
        dataset.data[key] = merged_features.get(key, None)

    if add_self_loops:
        dataset.data["graph"] = dgl.add_self_loop(dataset.data["graph"])

    return dataset
