"""Activation functions for SCM diversity.

Provides functional activation implementations and minimal nn.Module wrappers
for use in SCM networks.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# >>> Functional activations (pure functions)


def sign_activation(x: torch.Tensor) -> torch.Tensor:
    return 2 * (x >= 0.0).float() - 1.0


def rbf_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(x**2))


def square_activation(x: torch.Tensor) -> torch.Tensor:
    return x**2


def sqrt_abs_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x))


def unit_interval_indicator(x: torch.Tensor) -> torch.Tensor:
    return (torch.abs(x) <= 1.0).float()


# >>> Stateful layers (require nn.Module for state tracking)


class StdScaleLayer(nn.Module):
    """Standardizes input based on stats computed on first forward pass."""

    def __init__(self):
        super().__init__()
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-6
        return (x - self.mean) / self.std


class RandomScaleLayer(nn.Module):
    """Applies random scale and bias initialized on first forward pass."""

    def __init__(self):
        super().__init__()
        self.scale: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            self.scale = torch.exp(2 * torch.randn(1, device=x.device))
            self.bias = torch.randn(1, device=x.device)
        return self.scale * (x + self.bias)


class GaussianNoiseLayer(nn.Module):
    """Adds Gaussian noise during forward pass."""

    def __init__(self, std: float | torch.Tensor):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class RandomFourierActivation(nn.Module):
    """Random Fourier feature activation with learned-once parameters."""

    def __init__(self, n_frequencies: int = 256):
        super().__init__()
        freqs = n_frequencies * torch.rand(n_frequencies)
        bias = 2 * math.pi * torch.rand(n_frequencies)
        decay_exp = -np.exp(np.random.uniform(np.log(0.7), np.log(3.0)))
        freq_factors = freqs**decay_exp
        freq_factors = freq_factors / (freq_factors**2).sum().sqrt()
        weights = freq_factors * torch.randn(n_frequencies)

        self.register_buffer("freqs", freqs)
        self.register_buffer("bias", bias)
        self.register_buffer("weights", weights)
        self.stdscale = StdScaleLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stdscale(x)
        x = torch.sin(self.freqs * x[..., None] + self.bias)
        return (self.weights * x).sum(dim=-1)


# >>> Activation construction


FUNCTIONAL_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # Standard PyTorch activations (using F. for NN activations)
    "relu": F.relu,
    "tanh": F.tanh,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "selu": F.selu,
    "silu": F.silu,
    "softplus": F.softplus,
    "sigmoid": F.sigmoid,
    "relu6": F.relu6,
    "hardtanh": F.hardtanh,
    "identity": lambda x: x,
    # Custom activations (using torch. for math ops)
    "sign": sign_activation,
    "rbf": rbf_activation,
    "sine": torch.sin,
    "square": square_activation,
    "abs": torch.abs,
    "exp": torch.exp,
    "sqrt_abs": sqrt_abs_activation,
    "unit_interval_indicator": unit_interval_indicator,
}


class FunctionalActivation(nn.Module):
    """Thin wrapper to use a functional activation as nn.Module."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def make_activation(activation_type: str, with_scale: bool = True) -> nn.Module:
    """Create an activation module.

    Args:
        activation_type: One of the keys in FUNCTIONAL_ACTIVATIONS,
            or "random_fourier" for RandomFourierActivation
        with_scale: Prepend StdScaleLayer + RandomScaleLayer

    Returns:
        nn.Module implementing the activation.
    """
    base: nn.Module
    if activation_type == "random_fourier":
        base = RandomFourierActivation()
    elif activation_type in FUNCTIONAL_ACTIVATIONS:
        base = FunctionalActivation(FUNCTIONAL_ACTIVATIONS[activation_type])
    else:
        raise ValueError(f"Unknown activation: {activation_type}")

    if with_scale:
        return nn.Sequential(StdScaleLayer(), RandomScaleLayer(), base)
    return base


def sample_random_activation(with_scale: bool = True) -> nn.Module:
    """Sample a random activation from the available set.

    RandomFourierActivation is sampled with higher probability.
    """
    choices = list(FUNCTIONAL_ACTIVATIONS.keys()) + ["random_fourier"] * 10
    return make_activation(random.choice(choices), with_scale=with_scale)
