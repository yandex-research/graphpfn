import random

import torch

from ..prior_typings import TaskConfig


def permute_classes(labels: torch.Tensor) -> torch.Tensor:
    """Randomly permute class labels while preserving encoding.

    Args:
        labels: Class labels of shape (n_nodes,).

    Returns:
        Labels with permuted class mapping.
    """
    unique_vals, _ = torch.unique(labels, return_inverse=True)
    n_classes = len(unique_vals)

    if n_classes <= 1:
        return labels

    indices = unique_vals.argsort()
    mapped = indices[torch.searchsorted(unique_vals, labels)]

    perm = torch.randperm(n_classes, device=labels.device)
    return perm[mapped].to(labels.dtype)


def apply_task(
    labels: torch.Tensor,
    task_config: TaskConfig,
    permute_labels: bool,
) -> torch.Tensor:
    """Convert regression labels to classification if needed.

    Dispatches based on task_config["_type_"]:
    - "regression": return labels unchanged
    - "binclass": random boundary -> {0, 1}
    - "multiclass": rank/value-based -> {0, ..., n_classes-1}

    Args:
        labels: Regression targets of shape (n_nodes,).
        task_config: Task configuration with _type_ field.
        permute_labels: Whether to apply final label permutation.

    Returns:
        Converted labels of shape (n_nodes,).
    """
    labels = standard_scale_labels(labels)

    match task_config["_type_"]:
        case "regression":
            result = labels
        case "binclass":
            result = regression_to_binclass(
                labels,
                task_config["quantile"],
                task_config["p_reverse"],
            )
        case "multiclass":
            result = regression_to_multiclass(
                labels,
                task_config["n_classes"],
                task_config["multiclass_type"],
                task_config["p_ordered"],
                task_config["p_reverse"],
            )
        case _:
            raise ValueError(f"Unknown task type: {task_config['_type_']}")

    if permute_labels and task_config["_type_"] != "regression":
        result = permute_classes(result)

    return result


def standard_scale_labels(labels: torch.Tensor) -> torch.Tensor:
    """Standardize labels before class assignment.

    Args:
        labels: Regression targets of shape (n_nodes,).

    Returns:
        Standardized labels.
    """
    from .features import standard_scaling

    return standard_scaling(labels.unsqueeze(-1)).squeeze(-1)


def regression_to_binclass(
    labels: torch.Tensor,
    quantile: float,
    p_reverse: float,
) -> torch.Tensor:
    """Binary classification by splitting at a given quantile.

    Args:
        labels: Regression targets of shape (n_nodes,).
        quantile: Quantile in [0, 1] used as the decision boundary.
        p_reverse: Probability of flipping the class labels.

    Returns:
        Binary labels {0, 1} of shape (n_nodes,).
    """
    boundary = torch.quantile(labels.float(), quantile)
    classes = (labels < boundary).float()

    if random.random() < p_reverse:
        classes = 1.0 - classes

    return classes


def regression_to_multiclass(
    labels: torch.Tensor,
    n_classes: int,
    multiclass_type: str,
    p_ordered: float,
    p_reverse: float,
) -> torch.Tensor:
    """Multiclass assignment via random boundaries.

    Args:
        labels: Regression targets of shape (n_nodes,).
        n_classes: Number of classes to create (>= 2).
        multiclass_type: "rank" (boundaries from input) or "value" (from normal dist).
        p_ordered: Probability of keeping natural order (vs permuting).
        p_reverse: Probability of reversing class order.

    Returns:
        Class labels {0, ..., n_classes-1} of shape (n_nodes,).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    n_nodes = labels.shape[0]
    device = labels.device

    if multiclass_type == "rank":
        boundary_indices = torch.randint(0, n_nodes, (n_classes - 1,), device=device)
        boundaries = labels[boundary_indices]
    elif multiclass_type == "value":
        boundaries = torch.randn(n_classes - 1, device=device)
    else:
        raise ValueError(f"Unknown multiclass_type: {multiclass_type}")

    classes = (labels.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

    if random.random() > p_ordered:
        classes = permute_classes(classes)

    if random.random() < p_reverse:
        classes = n_classes - 1 - classes

    return classes.float()


def compute_n_train_nodes(n_nodes_requested: int, train_ratio: float) -> int:
    """Compute number of train nodes from requested graph size.

    Args:
        n_nodes_requested: Requested node count from config.
        train_ratio: Fraction of nodes for training.

    Returns:
        Number of training nodes.
    """
    return int(n_nodes_requested * train_ratio)
