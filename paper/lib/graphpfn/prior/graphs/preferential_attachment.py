"""Preferential attachment (Barabási-Albert) graph sampler."""

import random

import dgl
import numpy as np
import torch

from .util import shuffle_nodes, to_simple


def sample_preferential_attachment(
    n_nodes: int,
    avg_degree: float | None = None,
    *,
    max_degree: int | None = None,
    base_graph: dgl.DGLGraph | None = None,
    shuffle: bool = True,
) -> dgl.DGLGraph:
    """Sample Barabási-Albert preferential attachment graph.

    Generates scale-free graphs with power-law degree distribution.
    New nodes connect to existing nodes with probability proportional to
    their degree.

    Args:
        n_nodes: Total number of nodes in the final graph.
        avg_degree: Target average degree. Mutually exclusive with max_degree.
        max_degree: Maximum edges per new node. Mutually exclusive with avg_degree.
        base_graph: Optional base graph to extend. If provided, PA nodes are
            added on top of this graph. The base graph should already be
            processed (e.g., largest component extracted).
        shuffle: Whether to shuffle node ordering.

    Returns:
        Graph with n_nodes nodes. Always connected by construction when
        base_graph is None or connected.
    """
    if (avg_degree is None) == (max_degree is None):
        raise ValueError("Exactly one of avg_degree or max_degree must be provided")

    if max_degree is None:
        assert avg_degree is not None
        max_degree = max(1, int(avg_degree))

    if base_graph is not None:
        src, dst = base_graph.edges()
        src, dst = src.tolist(), dst.tolist()
        degrees = base_graph.in_degrees().tolist()
        n_base = len(degrees)
    else:
        src, dst = [0, 1], [1, 0]
        degrees = [1, 1]
        n_base = 2

    for new_node in range(n_base, n_nodes):
        n_new_edges = min(random.randint(1, max_degree), len(degrees))

        probs = np.array(degrees, dtype=np.float64)
        probs /= probs.sum()
        neighbors = np.random.choice(len(degrees), n_new_edges, replace=False, p=probs)

        for neighbor in neighbors:
            src.extend([new_node, neighbor])
            dst.extend([neighbor, new_node])
            degrees[neighbor] += 1

        degrees.append(n_new_edges)

    graph = dgl.graph(
        (torch.tensor(src), torch.tensor(dst)),
        num_nodes=n_nodes,
    )
    graph = to_simple(graph)
    if shuffle:
        graph = shuffle_nodes(graph)
    return graph
