import copy
import enum
from dataclasses import dataclass

import dgl
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)
from torch import Tensor

from graphpfn.data import FEATURE_KEYS, FeatureArrays


class NumTransform(enum.Enum):
    STANDARD = "standard"
    QUANTILE_NORMAL = "quantile-normal"
    QUANTILE_UNIFORM = "quantile-uniform"


class CatTransform(enum.Enum):
    ORDINAL = "ordinal"
    ONE_HOT = "one-hot"


@dataclass(frozen=True, kw_only=True)
class RegressionTargetStats:
    mean: float
    std: float


def prepare_features_tensor(
    features: FeatureArrays,
    train_mask: np.ndarray,
    *,
    num_transform: str | NumTransform | None = NumTransform.QUANTILE_NORMAL,
    frac_transform: str | NumTransform | None = NumTransform.QUANTILE_UNIFORM,
    cat_transform: str | CatTransform | None = CatTransform.ORDINAL,
    other_transform: str | NumTransform | None = NumTransform.STANDARD,
    d_pca: int | None = None,
    seed: int = 0,
) -> Tensor:
    # NOTE: in transductive setting, we have access to **all** features,
    #       including val and test features.

    features = copy.deepcopy(features)

    # Transforms
    transforms = {
        "num": num_transform,
        "frac": frac_transform,
        "cat": cat_transform,
        "other": other_transform,
    }
    for key in FEATURE_KEYS:
        transform: str | NumTransform | CatTransform | None = transforms[key]
        if key not in features:
            continue
        if transform is None:
            continue
        if key == "cat":
            features[key] = apply_cat_transform(features[key], CatTransform(transform))
        else:
            features[key] = apply_num_transform(
                features[key],
                NumTransform(transform),
                seed,
            )

    # Impute nans
    for key in list(features.keys()):
        features[key] = impute_nans(features[key])  # type: ignore

    # Apply PCA, if required
    if d_pca is not None:
        features = apply_d_reduction(features, d_pca)

    # Flatten, drop constant features and return
    features_tensor = np.concatenate(list(features.values()), axis=1)
    features_tensor = torch.tensor(features_tensor)
    features_tensor = drop_constant_features(features_tensor, torch.tensor(train_mask))

    return features_tensor.float()


def prepare_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    graph = dgl.remove_self_loop(graph)
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)
    return graph


def apply_num_transform(
    features: np.ndarray,
    transform: NumTransform,
    seed: int,
) -> np.ndarray:
    # NOTE: reserve a separate variable to avoid modifications in the original features
    features_seen = features

    if transform == NumTransform.STANDARD:
        normalizer = StandardScaler()

    elif transform in [NumTransform.QUANTILE_NORMAL, NumTransform.QUANTILE_UNIFORM]:
        distribution = {
            NumTransform.QUANTILE_NORMAL: "normal",
            NumTransform.QUANTILE_UNIFORM: "uniform",
        }[transform]

        assert seed is not None
        normalizer = QuantileTransformer(
            n_quantiles=max(min(features_seen.shape[0] // 30, 1000), 10),
            output_distribution=distribution,
            subsample=1_000_000_000,
            random_state=seed,
        )
        features_seen = features_seen + np.random.RandomState(seed).normal(
            0.0, 1e-5, features_seen.shape
        ).astype(features_seen.dtype)
    else:
        raise ValueError(f"Unknown {transform=}")

    normalizer.fit(features_seen)
    features_trasformed = normalizer.transform(features)

    return features_trasformed


def apply_cat_transform(
    features: np.ndarray,
    transform: CatTransform,
) -> np.ndarray:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int32,
    ).fit(features)
    features_encoded = encoder.transform(features)

    if transform == CatTransform.ORDINAL:
        return features_encoded

    if transform == CatTransform.ONE_HOT:
        encoder = OneHotEncoder(
            drop="if_binary",
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )

    encoder.fit(features_encoded)
    features_transformed = encoder.transform(features_encoded)
    return features_transformed


def apply_d_reduction(
    feature_arrays: FeatureArrays,
    d: int,
) -> FeatureArrays:
    # >>> Check if we need to apply
    total_n_featuers = 0
    for features in feature_arrays.values():
        assert isinstance(features, np.ndarray)
        total_n_featuers += features.shape[1]
    if total_n_featuers <= d:
        return feature_arrays

    # >>> Concat all features
    features_list = []
    for feature_type in list(feature_arrays.keys()):
        features = feature_arrays[feature_type]  # type: ignore
        features = impute_nans(features)
        if feature_type == "cat_features":
            features = apply_cat_transform(features, CatTransform.ONE_HOT)
        features_list.append(features)
    features_array = np.concatenate(features_list, axis=1)

    # >>> Apply dim reduction
    features_array = PCA(n_components=d).fit_transform(features_array)
    return {"other": features_array}


def impute_nans(
    features: np.ndarray,
    strategy: str = "most_frequent",
) -> np.ndarray:
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(features)


def drop_constant_features(features: Tensor, train_mask: Tensor) -> Tensor:
    n_features = features.shape[1]
    n_unique = [len(torch.unique(features[train_mask, i])) for i in range(n_features)]
    mask = torch.tensor(n_unique) > 1
    return features[:, mask]


def standardize_targets(
    *,
    y_train: Tensor,
) -> tuple[Tensor, RegressionTargetStats]:
    assert y_train.dtype == torch.float32

    mean = float(y_train.mean())
    std = float(y_train.std())

    labels_standardized = (y_train - mean) / std
    regression_stats = RegressionTargetStats(mean=mean, std=std)
    return labels_standardized, regression_stats
