"""Core SCM building blocks shared between MLP and GNN SCMs."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..config import resolve_or_sample
from ..prior_typings import DistributionSpec, NoiseConfig
from .activations import GaussianNoiseLayer, make_activation


def build_scm(
    n_inputs: int,
    n_outputs: int,
    n_layers: int,
    hidden_dim: int,
    activation_type: str | DistributionSpec,
    is_causal: bool,
    noise: NoiseConfig,
    make_layer: Callable[[int, int], nn.Module],
) -> nn.Sequential:
    """Build an SCM network with pluggable layers.

    Args:
        n_inputs: Input dimension (causes + structural features for GNN)
        n_outputs: Output dimension (typically 1 for regression)
        n_layers: Number of layers (must be >= 2)
        hidden_dim: Hidden layer dimension
        activation_type: Activation function type or distribution
        is_causal: Whether to use causal mode (no final output layer)
        noise: Noise configuration
        make_layer: Callable (d_in, d_out) -> nn.Module

    Returns:
        SCM as nn.Sequential
    """
    assert n_layers >= 2, "Number of layers must be at least 2"

    layers: list[nn.Module] = [make_layer(n_inputs, hidden_dim)]

    for _ in range(n_layers - 1):
        layers.append(
            _make_scm_block(
                hidden_dim, hidden_dim, activation_type, make_layer, **noise
            )
        )

    if not is_causal:
        layers.append(
            _make_scm_block(hidden_dim, n_outputs, activation_type, make_layer, **noise)
        )

    return nn.Sequential(*layers)


def _make_scm_block(
    d_in: int,
    d_out: int,
    activation_type: str | DistributionSpec,
    make_layer: Callable[[int, int], nn.Module],
    std: float,
    pre_sample_std: bool,
) -> nn.Sequential:
    """Create a single SCM block: activation -> layer -> noise."""
    resolved_activation = str(resolve_or_sample(activation_type))
    activation = make_activation(resolved_activation, with_scale=True)
    layer = make_layer(d_in, d_out)

    if pre_sample_std:
        sampled_std = torch.abs(torch.randn(1, d_out) * std)
        noise = GaussianNoiseLayer(sampled_std)
    else:
        noise = GaussianNoiseLayer(std)

    return nn.Sequential(activation, layer, noise)


@torch.no_grad()
def forward_collecting_outputs(
    scm: nn.Sequential, inputs: torch.Tensor
) -> list[torch.Tensor]:
    """Forward pass collecting outputs after each layer (excluding first)."""
    outputs = []
    x = inputs

    for i, layer in enumerate(scm):
        x = layer(x)
        if i > 0:
            outputs.append(x)

    return outputs


def compute_causal_hidden_dim(
    hidden_dim: int,
    n_features: int,
    n_outputs: int,
    is_causal: bool,
) -> int:
    """Compute hidden dim ensuring enough capacity for causal sampling."""
    if is_causal:
        return max(hidden_dim, n_outputs + 2 * n_features)
    return hidden_dim
