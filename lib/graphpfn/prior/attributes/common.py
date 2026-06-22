"""Shared utilities for SCM-based attribute generation."""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn


def initialize_weights(
    module: nn.Module,
    std: float,
    block_wise_dropout: bool,
    p_dropout: float,
    scale_std_by_dropout: bool,
) -> None:
    """Initialize SCM weights with optional dropout patterns."""
    weight_index = 0
    for _, param in module.named_parameters():
        if param.dim() != 2:
            continue
        if block_wise_dropout:
            _init_block_dropout(param, std, scale_std_by_dropout)
        else:
            _init_normal_with_dropout(
                param, std, p_dropout, scale_std_by_dropout, weight_index
            )
        weight_index += 1


def _init_block_dropout(
    param: torch.Tensor, init_std: float, scale_std_by_dropout: bool
) -> None:
    """Initialize weight matrix with block-wise dropout pattern."""
    nn.init.zeros_(param)
    n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
    block_size = [dim // n_blocks for dim in param.shape]
    p_keep = (n_blocks * block_size[0] * block_size[1]) / param.numel()

    std = init_std / (p_keep**0.5) if scale_std_by_dropout else init_std

    for block in range(n_blocks):
        block_slice = tuple(slice(dim * block, dim * (block + 1)) for dim in block_size)
        nn.init.normal_(param[block_slice], std=std)


def _init_normal_with_dropout(
    param: torch.Tensor,
    init_std: float,
    p_dropout: float,
    scale_std_by_dropout: bool,
    layer_index: int,
) -> None:
    """Initialize weight matrix with normal distribution and dropout mask."""
    p_dropout = p_dropout if layer_index > 0 else 0.0
    p_dropout = min(p_dropout, 0.99)

    std = init_std / ((1 - p_dropout) ** 0.5) if scale_std_by_dropout else init_std
    nn.init.normal_(param, std=std)
    param.data *= torch.bernoulli(torch.full_like(param, 1 - p_dropout))


def extract_features_and_labels(
    causes: torch.Tensor,
    outputs: list[torch.Tensor],
    n_features: int,
    n_outputs: int,
    enabled: bool,
    y_is_effect: bool,
    in_clique: bool,
    sort_features: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract X and y from SCM outputs.

    Args:
        causes: Input causes tensor
        outputs: List of intermediate SCM outputs
        n_features: Number of features to extract
        n_outputs: Number of output dimensions (typically 1)
        enabled: Whether causal mode is enabled
        y_is_effect: If True, y comes from last layer; else from early outputs
        in_clique: Sample X,y from contiguous block
        sort_features: Sort feature indices after selection

    Returns:
        features: (n_nodes, n_features)
        labels: (n_nodes, n_outputs)
    """
    if not enabled:
        return causes, outputs[-1]

    # Causal mode: sample X and y from concatenated hidden states
    outputs_flat = torch.cat(outputs, dim=-1)
    total_dim = outputs_flat.shape[-1]

    if in_clique:
        # Sample X and y from a contiguous block
        start = random.randint(0, total_dim - n_outputs - n_features)
        random_perm = start + torch.randperm(n_outputs + n_features)
    else:
        # Random permutation for feature selection
        random_perm = torch.randperm(total_dim - 1)

    # Select feature indices (after first n_outputs positions)
    indices_X = random_perm[n_outputs : n_outputs + n_features]

    if sort_features:
        indices_X, _ = torch.sort(indices_X)

    X = outputs_flat[:, indices_X]

    if y_is_effect:
        # y comes from the last layer outputs (effects)
        y = outputs_flat[:, -n_outputs:]
    else:
        # y comes from the beginning of the permuted list
        y = outputs_flat[:, random_perm[:n_outputs]]

    return X, y
