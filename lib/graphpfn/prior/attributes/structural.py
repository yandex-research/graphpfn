"""Structural feature computation from graph topology."""

from __future__ import annotations

import random

import dgl
import torch
from loguru import logger
from scipy.sparse.linalg import ArpackNoConvergence


def get_structural_feature_count(
    use_degree: bool,
    use_pagerank: bool,
    lappe_k: int,
) -> int:
    """Return the number of structural features that will be computed."""
    return int(use_degree) + int(use_pagerank) + lappe_k


def compute_structural_features(
    graph: dgl.DGLGraph,
    use_degree: bool,
    use_pagerank: bool,
    lappe_k: int,
) -> torch.Tensor | None:
    """Compute structural features from graph topology.

    Features are concatenated in order: degree, pagerank, lappe (if enabled).

    Args:
        graph: Input graph
        use_degree: Include normalized degree features
        use_pagerank: Include PageRank scores
        lappe_k: Number of Laplacian eigenvectors (0 = disabled)

    Returns:
        Tensor of shape (n_nodes, n_structural_features) or None if all disabled
    """
    features = []

    if use_degree:
        features.append(compute_degree_features(graph))

    if use_pagerank:
        features.append(compute_pagerank_features(graph))

    if lappe_k > 0:
        features.append(compute_lappe_features(graph, lappe_k))

    if not features:
        return None

    return torch.cat(features, dim=1)


def compute_degree_features(graph: dgl.DGLGraph) -> torch.Tensor:
    """Compute log-normalized degree features.

    Args:
        graph: Input graph

    Returns:
        Tensor of shape (n_nodes, 1)
    """
    degrees = graph.in_degrees().float()
    degrees = torch.log(1 + degrees)
    return degrees.unsqueeze(-1)


@torch.no_grad()
def compute_pagerank_features(
    graph: dgl.DGLGraph,
    alpha: float | None = None,
    max_iterations: int = 30,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute log PageRank scores.

    Args:
        graph: Input graph
        alpha: Damping factor (default: random between 0.6 and 0.9)
        max_iterations: Maximum number of power iterations
        tol: Convergence tolerance

    Returns:
        Tensor of shape (n_nodes, 1)
    """
    if alpha is None:
        alpha = 0.6 + 0.3 * random.random()

    n_nodes = graph.num_nodes()
    pv = torch.ones(n_nodes, device=graph.device) / n_nodes
    degrees = graph.out_degrees().float().clamp(min=1.0)
    p_reset = (1 - alpha) / n_nodes

    for _ in range(max_iterations):
        prev_pv = pv.clone()
        pv = dgl.ops.copy_u_sum(graph, pv / degrees)
        pv = alpha * pv + p_reset

        if torch.abs(pv - prev_pv).sum() < tol:
            break

    pv = torch.log(tol + pv)
    return pv.unsqueeze(-1)


def compute_lappe_features(graph: dgl.DGLGraph, k: int) -> torch.Tensor:
    """Compute Laplacian positional encodings.

    Args:
        graph: Input graph
        k: Number of eigenvectors to use

    Returns:
        Tensor of shape (n_nodes, k)
    """
    try:
        lappe = dgl.lap_pe(graph, k=k).to(graph.device)
    except ArpackNoConvergence:
        logger.warning("LapPE did not converge. Using random values.")
        lappe = torch.randn(graph.num_nodes(), k, device=graph.device)

    return lappe
