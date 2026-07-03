"""MLP-based Structural Causal Model for tabular attribute generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..checks import SanityCheckError
from ..prior_typings import MLPSCMConfig
from .base_scm import build_scm, compute_causal_hidden_dim, forward_collecting_outputs
from .common import extract_features_and_labels, initialize_weights
from .input_sampler import sample_inputs


def sample_attributes_mlp(
    n_nodes: int,
    config: MLPSCMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate node features and labels using MLP-based SCM."""
    n_features = config["n_features"]
    n_outputs = 1
    is_causal = config["causal"]["enabled"]

    hidden_dim = compute_causal_hidden_dim(
        config["hidden_dim"], n_features, n_outputs, is_causal
    )
    n_causes = config["n_causes"] if is_causal else n_features

    scm = build_scm(
        n_inputs=n_causes,
        n_outputs=n_outputs,
        n_layers=config["n_layers"],
        hidden_dim=hidden_dim,
        activation_type=config["activation_type"],
        is_causal=is_causal,
        noise=config["noise"],
        make_layer=nn.Linear,
    )
    initialize_weights(scm, **config["init"])

    causes = sample_inputs(n_nodes, n_causes, **config["causes"])
    outputs = forward_collecting_outputs(scm, causes)

    X, y = extract_features_and_labels(
        causes=causes,
        outputs=outputs,
        n_features=n_features,
        n_outputs=n_outputs,
        **config["causal"],
    )

    if torch.isnan(X).any() or torch.isnan(y).any():
        raise SanityCheckError("SCM produced NaN values")

    if not torch.isfinite(X).all() or not torch.isfinite(y).all():
        raise SanityCheckError("SCM produced infinite values")

    return X, y.squeeze(-1) if n_outputs == 1 else y
