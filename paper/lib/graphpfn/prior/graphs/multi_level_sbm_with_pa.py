"""Multi-level Stochastic Block Model with Preferential Attachment sampler.

Generates hierarchical community structure graphs with scale-free properties.
Architecture:
1. First level: Multiple disjoint SBM subgraphs (local communities)
2. Second level: Global SBM over all nodes (inter-community connections)
3. PA: Add nodes via preferential attachment (scale-free properties)
"""

from typing import cast

import dgl

from ..config import ConfigValue, resolve_or_sample
from .preferential_attachment import sample_preferential_attachment
from .sbm import sample_sbm
from .util import (
    extract_largest_component,
    merge_graphs,
    random_partition,
    shuffle_nodes,
    to_simple,
)


def sample_multi_level_sbm_with_pa(
    n_nodes: int,
    avg_degree: float,
    *,
    # PA parameters
    pa_nodes_ratio: float,
    pa_max_degree: int,
    # First level (local communities)
    n_first_level_subgraphs: int,
    first_level_degree_ratio: float,
    min_first_level_n_nodes: int,
    # SBM parameters (used for both levels)
    n_groups: int | ConfigValue,
    offdiagonal_coef: float,
) -> dgl.DGLGraph:
    """Sample multi-level SBM with preferential attachment graph.

    Generates hierarchical community structure with scale-free properties:
    1. Split nodes into PA nodes and pre-PA nodes
    2. First level: Create disjoint SBM subgraphs (local communities)
    3. Second level: Add global SBM edges (inter-community connections)
    4. Extract largest component and add PA nodes

    Args:
        n_nodes: Total number of nodes in the final graph.
        avg_degree: Target average degree.
        pa_nodes_ratio: Fraction of nodes added via preferential attachment.
        pa_max_degree: Maximum edges per PA node.
        n_first_level_subgraphs: Number of first-level SBM subgraphs.
        first_level_degree_ratio: Fraction of avg_degree for first level.
        min_first_level_n_nodes: Minimum nodes per first-level subgraph.
        n_groups: Number of groups for SBM sampling.
        offdiagonal_coef: Off-diagonal coefficient for SBM.

    Returns:
        Graph with hierarchical community structure. May have fewer nodes
        than requested after extracting largest component.
    """
    # Compute node counts
    n_nodes_before_pa = int(n_nodes * (1.0 - pa_nodes_ratio))
    n_nodes_before_pa = max(n_nodes_before_pa, min_first_level_n_nodes + 1)
    n_pa_nodes = n_nodes - n_nodes_before_pa

    # Adjust average degree to account for PA edges
    if pa_nodes_ratio > 0:
        pa_avg_degree = (1.0 + pa_max_degree) / 2
        adjusted_avg_degree = (avg_degree - pa_nodes_ratio * pa_avg_degree) / (
            1.0 - pa_nodes_ratio
        )
        adjusted_avg_degree = max(adjusted_avg_degree, 1.0)
    else:
        adjusted_avg_degree = avg_degree

    # Clamp first_level_degree_ratio to avoid invalid first-level avg degree
    # (first_level_avg_degree must be < min_first_level_n_nodes - 1)
    max_first_level_degree_ratio = min(
        first_level_degree_ratio,
        (min_first_level_n_nodes - 1) / adjusted_avg_degree,
    )
    first_level_degree_ratio = max_first_level_degree_ratio

    # Compute degree split between levels
    first_level_avg_degree = adjusted_avg_degree * first_level_degree_ratio
    second_level_avg_degree = adjusted_avg_degree * (1.0 - first_level_degree_ratio)

    # Adjust number of subgraphs based on available nodes
    actual_n_subgraphs = min(
        n_first_level_subgraphs, n_nodes_before_pa // min_first_level_n_nodes
    )
    actual_n_subgraphs = max(actual_n_subgraphs, 1)

    # Partition nodes into subgraphs
    subgraph_sizes = random_partition(
        n_nodes_before_pa,
        actual_n_subgraphs,
        min_size=min_first_level_n_nodes,
    )

    # Sample first level (disjoint communities)
    first_level_subgraphs = []
    for size in subgraph_sizes:
        actual_n_groups = min(cast(int, resolve_or_sample(n_groups)), size)
        subgraph = sample_sbm(
            size, first_level_avg_degree, actual_n_groups, offdiagonal_coef
        )
        first_level_subgraphs.append(subgraph)

    first_level = dgl.batch(first_level_subgraphs)
    first_level = shuffle_nodes(first_level)

    # Sample second level (global structure)
    actual_n_groups = min(cast(int, resolve_or_sample(n_groups)), n_nodes_before_pa)
    second_level = sample_sbm(
        n_nodes_before_pa, second_level_avg_degree, actual_n_groups, offdiagonal_coef
    )

    graph = merge_graphs([first_level, second_level])
    graph = to_simple(graph)

    graph, _ = extract_largest_component(graph)

    # Add PA nodes if any
    if n_pa_nodes > 0:
        graph = sample_preferential_attachment(
            graph.num_nodes() + n_pa_nodes,
            max_degree=pa_max_degree,
            base_graph=graph,
        )

    graph = to_simple(shuffle_nodes(graph))
    return graph
