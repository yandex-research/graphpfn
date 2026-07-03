"""GNN-based Structural Causal Model for graph-aware attribute generation."""

from __future__ import annotations

from functools import partial

import dgl
import torch

from ..checks import SanityCheckError
from ..prior_typings import GNNSCMConfig
from .base_scm import build_scm, compute_causal_hidden_dim, forward_collecting_outputs
from .common import extract_features_and_labels, initialize_weights
from .input_sampler import sample_inputs
from .layers import MixedGraphLinear
from .structural import compute_structural_features, get_structural_feature_count


def sample_attributes_gnn(
    graph: dgl.DGLGraph,
    config: GNNSCMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate node features and labels using GNN-based SCM."""
    base = config["base"]
    n_nodes = graph.num_nodes()
    n_features = base["n_features"]
    n_outputs = 1
    is_causal = base["causal"]["enabled"]

    hidden_dim = compute_causal_hidden_dim(
        base["hidden_dim"], n_features, n_outputs, is_causal
    )
    n_causes = base["n_causes"] if is_causal else n_features

    n_structural = get_structural_feature_count(**config["structural"])
    n_inputs = n_causes + n_structural

    make_layer = partial(
        MixedGraphLinear,
        conv_type=config["conv_type"],
        graph_conv_ratio=config["graph_conv_ratio"],
        graph=graph,
    )

    scm = build_scm(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_layers=base["n_layers"],
        hidden_dim=hidden_dim,
        activation_type=base["activation_type"],
        is_causal=is_causal,
        noise=base["noise"],
        make_layer=make_layer,
    )
    initialize_weights(scm, **base["init"])

    causes = sample_inputs(n_nodes, n_causes, **base["causes"])
    struct_features = compute_structural_features(graph, **config["structural"])
    if struct_features is not None:
        inputs = torch.cat([causes, struct_features], dim=1)
    else:
        inputs = causes

    outputs = forward_collecting_outputs(scm, inputs)

    X, y = extract_features_and_labels(
        causes=causes,
        outputs=outputs,
        n_features=n_features,
        n_outputs=n_outputs,
        **base["causal"],
    )

    if torch.isnan(X).any() or torch.isnan(y).any():
        raise SanityCheckError("SCM produced NaN values")

    if not torch.isfinite(X).all() or not torch.isfinite(y).all():
        raise SanityCheckError("SCM produced infinite values")

    return X, y.squeeze(-1) if n_outputs == 1 else y
