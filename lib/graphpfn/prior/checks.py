import torch

from lib.util import TaskType


class SanityCheckError(Exception):
    pass


def check_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_train_nodes: int,
    task_type: TaskType,
    min_features: int,
    n_classes: int | None,
) -> None:
    n_nodes = features.shape[0]

    if n_train_nodes >= n_nodes:
        raise SanityCheckError(
            f"n_train_nodes ({n_train_nodes}) must be < n_nodes ({n_nodes})"
        )

    check_no_nan(features, "features")
    check_no_nan(labels, "labels")
    check_min_features(features.shape[1], min_features)

    if task_type == TaskType.BINCLASS:
        check_n_classes(labels, 2)
        check_class_coverage(labels, n_train_nodes)
    elif task_type == TaskType.MULTICLASS:
        assert n_classes is not None
        check_n_classes(labels, n_classes)
        check_class_coverage(labels, n_train_nodes)


def check_no_nan(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        raise SanityCheckError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise SanityCheckError(f"{name} contains Inf values")


def check_min_features(n_features: int, min_features: int) -> None:
    if n_features < min_features:
        raise SanityCheckError(
            f"Only {n_features} features remain after dropping constants, "
            f"but min_features={min_features}"
        )


def check_n_classes(labels: torch.Tensor, n_classes: int) -> None:
    actual_n_classes = len(labels.unique())
    if actual_n_classes != n_classes:
        raise SanityCheckError(
            f"Expected {n_classes} classes, but got {actual_n_classes}"
        )


def check_class_coverage(labels: torch.Tensor, n_train_nodes: int) -> None:
    train_labels = labels[:n_train_nodes]
    test_labels = labels[n_train_nodes:]

    train_classes = set(train_labels.unique().tolist())
    test_classes = set(test_labels.unique().tolist())

    if not train_classes:
        raise SanityCheckError("No training examples")
    if not test_classes:
        raise SanityCheckError("No test examples")

    all_classes = train_classes | test_classes
    missing_in_train = all_classes - train_classes
    missing_in_test = all_classes - test_classes

    if missing_in_train:
        raise SanityCheckError(
            f"Classes {missing_in_train} present in test but not in train"
        )
    if missing_in_test:
        raise SanityCheckError(
            f"Classes {missing_in_test} present in train but not in test"
        )
