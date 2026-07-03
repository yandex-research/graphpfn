"""Graph sampling building blocks."""

from ..prior_typings import GraphConfig, unpack
from .erdos_renyi import sample_erdos_renyi
from .geometric import sample_geometric
from .multi_level_sbm_with_pa import sample_multi_level_sbm_with_pa
from .preferential_attachment import sample_preferential_attachment
from .sbm import sample_sbm
from .util import (
    extract_largest_component,
    merge_graphs,
    random_partition,
    sample_adjacency,
    shuffle_nodes,
    to_simple,
)


def sample_graph(config: GraphConfig):
    n_nodes = config["n_nodes"]
    avg_degree = config["avg_degree"]
    sampler = config["sampler"]

    match sampler["_type_"]:
        case "sbm":
            graph = sample_sbm(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case "geometric":
            graph = sample_geometric(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case "preferential-attachment":
            graph = sample_preferential_attachment(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
            )
        case "erdos-renyi":
            graph = sample_erdos_renyi(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
            )
        case "multi-level-sbm-with-pa":
            graph = sample_multi_level_sbm_with_pa(
                n_nodes=n_nodes,
                avg_degree=avg_degree,
                **unpack(sampler),
            )
        case _:
            raise ValueError(f"Unknown graph sampler: {sampler['_type_']}")

    graph = to_simple(graph)
    graph, _ = extract_largest_component(graph)
    graph = shuffle_nodes(graph)

    return graph


__all__ = [
    "extract_largest_component",
    "merge_graphs",
    "random_partition",
    "sample_adjacency",
    "sample_erdos_renyi",
    "sample_geometric",
    "sample_graph",
    "sample_multi_level_sbm_with_pa",
    "sample_preferential_attachment",
    "sample_sbm",
    "shuffle_nodes",
    "to_simple",
]
