"""Input sampling utilities for SCM data generation."""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch


InputStrategy = Literal["normal", "uniform", "mixed"]


def sample_inputs(
    n_nodes: int,
    n_features: int,
    strategy: InputStrategy = "mixed",
    pre_sample_stats: bool = False,
) -> torch.Tensor:
    """Sample initial input features for SCM.

    Args:
        n_nodes: Number of nodes (rows)
        n_features: Number of features (columns)
        strategy: Sampling strategy
            - "normal": Standard normal distribution
            - "uniform": Uniform [0, 1]
            - "mixed": Random combination of normal, multinomial, zipf, uniform
        pre_sample_stats: If True and strategy="normal", pre-sample mean/std
            for each feature column

    Returns:
        Tensor of shape (n_nodes, n_features), float32
    """
    if strategy == "normal":
        return _sample_normal(n_nodes, n_features, pre_sample_stats)
    elif strategy == "uniform":
        return _sample_uniform(n_nodes, n_features)
    elif strategy == "mixed":
        return _sample_mixed(n_nodes, n_features)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def _sample_normal(
    n_nodes: int, n_features: int, pre_sample_stats: bool
) -> torch.Tensor:
    if pre_sample_stats:
        # Pre-sample mean and std for each feature column
        # std is correlated with mean (matches old implementation)
        means = torch.randn(n_features)
        stds = torch.abs(torch.randn(n_features) * means)
        return torch.randn(n_nodes, n_features, dtype=torch.float32) * stds + means
    return torch.randn(n_nodes, n_features, dtype=torch.float32)


def _sample_uniform(n_nodes: int, n_features: int) -> torch.Tensor:
    return torch.rand(n_nodes, n_features, dtype=torch.float32)


def _sample_multinomial_feature(n_nodes: int) -> torch.Tensor:
    """Sample a single categorical feature with random number of categories."""
    n_categories = random.randint(2, 20)
    probs = torch.rand(n_categories)
    x = torch.multinomial(probs, n_nodes, replacement=True).float()
    return (x - x.mean()) / (x.std() + 1e-6)


def _sample_zipf_feature(n_nodes: int) -> torch.Tensor:
    """Sample a single Zipf-distributed feature."""
    arr = np.random.zipf(2.0 + random.random() * 2, n_nodes)
    x = torch.tensor(arr, dtype=torch.float32).clamp(max=10)
    return x - x.mean()


def _sample_mixed(n_nodes: int, n_features: int) -> torch.Tensor:
    """Sample features with mixed distributions.

    Each feature is sampled from one of: normal, multinomial, zipf, or uniform.
    The selection cascades: try normal first, then multinomial, then zipf,
    finally uniform. Random thresholds control how often each is skipped.
    """
    features = []
    p_skip_normal = random.random() * 0.66
    p_skip_multi = random.random() * 0.66
    p_skip_zipf = random.random() * 0.66

    for _ in range(n_features):
        if random.random() > p_skip_normal:
            x = torch.randn(n_nodes, dtype=torch.float32)
        elif random.random() > p_skip_multi:
            x = _sample_multinomial_feature(n_nodes)
        elif random.random() > p_skip_zipf:
            x = _sample_zipf_feature(n_nodes)
        else:
            x = torch.rand(n_nodes, dtype=torch.float32)
        features.append(x)

    return torch.stack(features, dim=-1)


def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_std: float | torch.Tensor,
) -> torch.Tensor:
    """Add Gaussian noise to tensor.

    Args:
        tensor: Input tensor
        noise_std: Standard deviation of noise (scalar or per-feature tensor)

    Returns:
        Tensor with added noise, same shape as input
    """
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise
