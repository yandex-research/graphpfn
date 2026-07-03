"""Degree-corrected Stochastic Block Model sampler."""

import dgl
import numpy as np
import torch

from ..checks import SanityCheckError
from .util import random_partition, shuffle_nodes, to_simple


def _sample_block_n_edges(
    group_sizes: list[int],
    offdiagonal_coef: float,
    avg_degree: float,
) -> torch.Tensor:
    n_groups = len(group_sizes)
    n_nodes = sum(group_sizes)

    uniform_density = np.random.random() < 0.5
    block_density = torch.zeros([n_groups, n_groups], dtype=torch.float32)

    for i in range(n_groups):
        block_density[i, i] = 1.0 if uniform_density else np.random.random()

    offdiag_base = np.random.random()
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if uniform_density:
                block_density[i, j] = offdiagonal_coef * offdiag_base
            else:
                max_val = (block_density[i, i] * block_density[j, j]) ** 0.5
                block_density[i, j] = offdiagonal_coef * np.random.random() * max_val

    sizes = torch.tensor(group_sizes, dtype=torch.float32)
    block_n_edges = block_density * sizes[:, None] * sizes[None, :]
    block_n_edges = block_n_edges / block_n_edges.sum()
    block_n_edges = (0.5 * n_nodes * avg_degree) * block_n_edges

    return block_n_edges


def _sample_block_normalized_degrees(groups: torch.Tensor) -> torch.Tensor:
    n_nodes = groups.numel()

    if np.random.random() < 0.5:
        gamma = np.random.uniform(2, 3)
        degrees = torch.tensor(np.random.zipf(gamma, [n_nodes]), dtype=torch.float32)
    else:
        degrees = torch.rand([n_nodes], dtype=torch.float32)

    for group_id in groups.unique():
        mask = groups == group_id
        degrees[mask] /= degrees[mask].sum()

    return degrees


def _sample_dc_sbm(
    groups: torch.Tensor,
    block_n_edges: torch.Tensor,
    degrees: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_groups = groups.max() + 1
    src_list, dst_list = [], []

    for first_group in range(n_groups):
        for second_group in range(first_group, n_groups):
            first_group_nodes = torch.nonzero(groups == first_group).squeeze(-1)
            second_group_nodes = torch.nonzero(groups == second_group).squeeze(-1)

            if (first_group_nodes.numel() == 0) or (second_group_nodes.numel() == 0):
                raise SanityCheckError("Empty group found")

            n_edges = int(torch.poisson(block_n_edges[first_group, second_group]))

            if n_edges == 0:
                src_list.append(first_group_nodes.new_empty(0))
                dst_list.append(second_group_nodes.new_empty(0))
                continue

            if first_group == second_group:
                if first_group_nodes.numel() < 2:
                    src_list.append(first_group_nodes.new_empty(0))
                    dst_list.append(second_group_nodes.new_empty(0))
                    continue
                src_dst = torch.multinomial(
                    degrees[first_group_nodes].expand(n_edges, -1),
                    2,
                    replacement=False,
                )
                src, dst = src_dst[:, 0], src_dst[:, 1]
                src_list.append(first_group_nodes[src])
                dst_list.append(first_group_nodes[dst])
            else:
                src = torch.multinomial(
                    degrees[first_group_nodes],
                    n_edges,
                    replacement=True,
                )
                dst = torch.multinomial(
                    degrees[second_group_nodes],
                    n_edges,
                    replacement=True,
                )
                src_list.append(first_group_nodes[src])
                dst_list.append(second_group_nodes[dst])

    return torch.cat(src_list), torch.cat(dst_list)


def sample_sbm_groups(
    n_nodes: int,
    n_groups: int,
    min_group_size: int = 4,
    min_group_fraction: float = 0.1,
) -> torch.Tensor:
    """Randomly partition nodes into groups for SBM generation."""
    min_size = max(min_group_size, int((n_nodes / n_groups) * min_group_fraction))
    group_sizes = random_partition(n_nodes, n_groups, min_size)
    return torch.cat(
        [torch.full([size], i, dtype=torch.int32) for i, size in enumerate(group_sizes)]
    )


def sample_sbm_from_groups(
    groups: torch.Tensor,
    avg_degree: float,
    offdiagonal_coef: float,
) -> dgl.DGLGraph:
    """Sample degree-corrected SBM graph given pre-assigned groups.

    Args:
        groups: Integer tensor of shape (n_nodes,) with group assignments.
        avg_degree: Target average degree.
        offdiagonal_coef: Controls inter-group edge density.

    Returns:
        Undirected simple graph with shuffled node ordering.
    """
    n_nodes = groups.numel()
    _, group_sizes_tensor = groups.unique(return_counts=True)
    group_sizes = group_sizes_tensor.tolist()

    block_n_edges = _sample_block_n_edges(group_sizes, offdiagonal_coef, avg_degree)
    degrees = _sample_block_normalized_degrees(groups)
    src, dst = _sample_dc_sbm(groups, block_n_edges, degrees)

    graph = dgl.graph((src, dst), num_nodes=n_nodes)
    return to_simple(graph)


def sample_sbm(
    n_nodes: int,
    avg_degree: float,
    n_groups: int,
    offdiagonal_coef: float,
) -> dgl.DGLGraph:
    """Sample degree-corrected Stochastic Block Model graph.

    Generates graphs with community structure. Each node belongs to one group,
    and edge probabilities depend on group memberships and node-specific
    degree parameters.

    May be disconnected (if offdiagonal_coef is low). Apply
    extract_largest_component() if connectivity is required.
    """
    groups = sample_sbm_groups(n_nodes, n_groups)
    graph = sample_sbm_from_groups(groups, avg_degree, offdiagonal_coef)
    graph = shuffle_nodes(graph)
    return graph
