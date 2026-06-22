"""Erdős-Rényi random graph sampler."""

import dgl
import torch

from .util import sample_adjacency, to_simple


def sample_erdos_renyi(n_nodes: int, avg_degree: float) -> dgl.DGLGraph:
    """Sample Erdős-Rényi random graph.

    Uniform edge probability model with approximately Poisson degree distribution.
    May be disconnected. Apply extract_largest_component() if connectivity is required.
    """
    # Edge probability for target average degree
    edge_prob = avg_degree / (n_nodes - 1)
    probs = torch.full((n_nodes, n_nodes), edge_prob).triu(diagonal=1)

    adj = sample_adjacency(probs)
    src, dst = torch.nonzero(adj, as_tuple=True)

    graph = dgl.graph(
        (torch.cat([src, dst]), torch.cat([dst, src])),
        num_nodes=n_nodes,
    )
    return to_simple(graph)
