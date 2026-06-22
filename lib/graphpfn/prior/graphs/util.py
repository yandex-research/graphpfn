"""Graph utility functions for node shuffling, component extraction, and
simplification."""

import dgl
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components


def random_partition(n: int, n_parts: int, min_size: int = 1) -> list[int]:
    """Partition n into n_parts random positive integers, each >= min_size.

    Uses "random cuts" algorithm: place n_parts-1 random cut points,
    then compute differences.
    """
    assert 1 <= n_parts <= n
    assert min_size * n_parts <= n

    if n_parts == 1:
        return [n]

    remaining = n - (min_size - 1) * n_parts
    cuts = np.sort(
        np.random.choice(np.arange(1, remaining), size=n_parts - 1, replace=False)
    )
    parts = np.diff(np.concatenate(([0], cuts, [remaining])))

    return [int(x) + (min_size - 1) for x in parts]


def sample_adjacency(probs: torch.Tensor) -> torch.Tensor:
    """Sample binary adjacency matrix from edge probability matrix.

    Handles probabilities > 1 by scaling to preserve expected edge count.
    Input should be upper triangular (for undirected graphs).
    """
    probs = probs.double()
    target_sum = probs.sum().item()

    # Scale probabilities if any exceed 1.0 (preserve expected edge count)
    if probs.max() > 1.0:
        c_lo, c_hi = 1.0, 1.0 / probs[probs > 0].min().item()
        for _ in range(1000):
            c_mid = (c_lo + c_hi) / 2
            current_sum = (c_mid * probs).clip(max=1.0).sum().item()
            if abs(current_sum - target_sum) < 1e-4 * target_sum:
                break
            if current_sum < target_sum:
                c_lo = c_mid
            else:
                c_hi = c_mid
        probs = ((c_lo + c_hi) / 2 * probs).clip(max=1.0)

    return (torch.rand_like(probs) < probs).int()


def shuffle_nodes(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Randomly permute node ordering.

    Ensures train/test split is not correlated with generation order.
    Graph must have no node data (ndata).

    Args:
        graph: Input graph (no ndata)

    Returns:
        Graph with shuffled node IDs
    """
    assert len(graph.ndata) == 0, "shuffle_nodes does not support graphs with ndata"
    n_nodes = graph.num_nodes()
    perm = torch.randperm(n_nodes)

    src, dst = graph.edges()
    new_src, new_dst = perm[src], perm[dst]

    return dgl.graph((new_src, new_dst), num_nodes=n_nodes)


def extract_largest_component(
    graph: dgl.DGLGraph,
) -> tuple[dgl.DGLGraph, torch.Tensor]:
    """Extract largest connected component.

    May reduce node count. This ensures generated graphs are connected,
    which is required for many graph algorithms.

    Args:
        graph: Input graph (possibly disconnected, no ndata expected).

    Returns:
        A tuple of (subgraph, mask) where mask is a boolean tensor of shape
        (n_nodes,) indicating which original nodes belong to the largest
        component.
    """
    n_nodes = graph.num_nodes()
    adj_scipy = graph.adj_external(scipy_fmt="csr")
    n_components, labels = connected_components(adj_scipy, directed=False)

    if n_components == 1:
        return graph, torch.ones(n_nodes, dtype=torch.bool)

    labels_tensor = torch.tensor(labels)
    largest_idx = torch.bincount(labels_tensor).argmax().item()
    mask = labels_tensor == largest_idx

    new_ids = torch.cumsum(mask.int(), dim=0) - 1

    src, dst = graph.edges()
    edge_mask = mask[src] & mask[dst]
    new_src = new_ids[src[edge_mask]]
    new_dst = new_ids[dst[edge_mask]]

    return dgl.graph((new_src, new_dst), num_nodes=mask.sum().item()), mask


def merge_graphs(graphs: list[dgl.DGLGraph]) -> dgl.DGLGraph:
    """Merge multiple graphs with the same node set into one.

    All graphs must have the same number of nodes. The result contains
    the union of all edges (may have duplicates; apply to_simple after).
    """
    n_nodes = graphs[0].num_nodes()
    assert all(g.num_nodes() == n_nodes for g in graphs)

    all_src = []
    all_dst = []
    for g in graphs:
        src, dst = g.edges()
        all_src.append(src)
        all_dst.append(dst)

    return dgl.graph(
        (torch.cat(all_src), torch.cat(all_dst)),
        num_nodes=n_nodes,
    )


def to_simple(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Convert to simple undirected graph.

    Removes self-loops and multi-edges. Ensures bidirectional edges.

    Args:
        graph: Input graph (possibly with self-loops/multi-edges)

    Returns:
        Simple undirected graph
    """
    graph = dgl.to_bidirected(graph)  # type: ignore
    graph = dgl.to_simple(graph)  # type: ignore
    graph = dgl.remove_self_loop(graph)
    return graph
