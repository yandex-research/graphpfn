import random

import numpy as np
import torch


def torch_nanstd(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    ddof: int = 0,
) -> torch.Tensor:
    """Standard deviation ignoring NaNs, using NumPy internally."""
    device = input.device
    np_input = input.cpu().numpy()
    with np.errstate(all="ignore"):
        std = np.nanstd(np_input, axis=dim, keepdims=keepdim, ddof=ddof)
    return torch.from_numpy(np.asarray(std)).float().to(device)


def standard_scaling(
    input: torch.Tensor,
    clip_value: float = 100.0,
) -> torch.Tensor:
    """Standardize features by removing mean and scaling to unit variance.

    NaNs are ignored in mean/std calculation.

    Args:
        input: Input tensor of shape (n_nodes, n_features) or (n_nodes,).
        clip_value: Clip standardized values to [-clip_value, clip_value].

    Returns:
        Standardized tensor, clipped to prevent extreme outliers.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clamp(
        min=1e-6
    )
    scaled = (input - mean) / std
    return scaled.clamp(min=-clip_value, max=clip_value)


def outlier_removing(
    input: torch.Tensor,
    threshold: float = 4.0,
) -> torch.Tensor:
    """Clamp outliers based on number of standard deviations.

    Two-pass approach: first identify outliers, then recalculate stats
    from non-outlier values for more robust clamping.

    Args:
        input: Input tensor of shape (n_nodes, n_features).
        threshold: Number of standard deviations for cutoff.

    Returns:
        Tensor with outliers clamped.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clamp(
        min=1e-6
    )
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(
        masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0
    ).clamp(min=1e-6)

    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


def num_to_categorical(
    features: torch.Tensor,
    p_cat: float,
    max_categories: int,
) -> torch.Tensor:
    """Randomly convert some numeric features to categorical.

    With probability p_cat, enters categorical conversion mode.
    Then each column has independent probability of being converted.

    Args:
        features: Input features of shape (n_nodes, n_features).
        p_cat: Probability of entering categorical conversion mode.
        max_categories: Maximum number of categories per converted feature.

    Returns:
        Features with some columns potentially converted to categorical.
    """
    if random.random() >= p_cat:
        return features

    features = features.clone()
    col_prob = random.random()
    n_nodes = features.shape[0]

    for col in range(features.shape[1]):
        if random.random() < col_prob:
            num_cats = min(max(round(random.gammavariate(1, 10)), 2), max_categories)

            boundary_indices = torch.randint(
                0, n_nodes, (num_cats - 1,), device=features.device
            )
            boundaries = features[boundary_indices, col]
            classes = (features[:, col].unsqueeze(-1) > boundaries.unsqueeze(0)).sum(
                dim=1
            )

            if random.random() > 0.3:
                unique_vals, _ = torch.unique(classes, return_inverse=True)
                n_classes = len(unique_vals)
                if n_classes > 1:
                    indices = unique_vals.argsort()
                    mapped = indices[torch.searchsorted(unique_vals, classes)]
                    perm = torch.randperm(n_classes, device=features.device)
                    classes = perm[mapped]

            features[:, col] = classes.float()

    return features


def permute_features(features: torch.Tensor) -> torch.Tensor:
    """Randomly permute feature order.

    Args:
        features: Input features of shape (n_nodes, n_features).

    Returns:
        Features with columns randomly permuted.
    """
    n_features = features.shape[1]
    perm = torch.randperm(n_features, device=features.device)
    return features[:, perm]


def drop_constant_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove features with zero variance.

    Args:
        features: Input features of shape (n_nodes, n_features).

    Returns:
        filtered_features: Features with constants removed.
        mask: Boolean mask indicating which features were kept.
    """
    n_features = features.shape[1]
    mask = torch.zeros(n_features, dtype=torch.bool, device=features.device)

    for j in range(n_features):
        n_unique = len(torch.unique(features[:, j]))
        mask[j] = n_unique > 1

    return features[:, mask], mask


def process_features(
    features: torch.Tensor,
    p_cat: float,
    max_categories: int,
    do_permute_features: bool,
    outlier_threshold: float = 4.0,
) -> torch.Tensor:
    """Full feature preprocessing pipeline.

    Applies in order: categorical conversion, outlier removal, standard scaling,
    feature permutation.

    Args:
        features: Input features of shape (n_nodes, n_features).
        p_cat: Probability of categorical conversion mode.
        max_categories: Max categories for categorical conversion.
        do_permute_features: Whether to randomly permute feature order.
        outlier_threshold: Std threshold for outlier clamping.

    Returns:
        Preprocessed features.
    """
    features = num_to_categorical(features, p_cat, max_categories)
    features = outlier_removing(features, threshold=outlier_threshold)
    features = standard_scaling(features)
    if do_permute_features:
        features = permute_features(features)
    return features
