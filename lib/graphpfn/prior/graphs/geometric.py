"""Random geometric graph sampler."""

import dgl
import numpy as np
import torch

from .util import to_simple


def sample_geometric(
    n_nodes: int,
    avg_degree: float,
    n_latent_features: int,
) -> dgl.DGLGraph:
    """Sample random geometric graph.

    Nodes are points in latent space, edges connect nearby points.
    Distance threshold is chosen to achieve target average degree.

    May be disconnected. Apply extract_largest_component() if connectivity
    is required.
    """
    # Sample points from uniform or normal distribution
    if np.random.random() < 0.5:
        points = torch.rand((n_nodes, n_latent_features))
    else:
        points = torch.randn((n_nodes, n_latent_features))

    # Compute pairwise distances
    distances = torch.cdist(points, points)

    # Find distance threshold via quantile (subsample for efficiency)
    n_pairs = n_nodes * (n_nodes - 1) // 2
    target_edges = int(0.5 * n_nodes * avg_degree)
    q = target_edges / n_pairs

    upper_distances = distances.triu(diagonal=1).flatten()
    upper_distances = upper_distances[upper_distances > 0]

    if len(upper_distances) > 100_000:
        upper_distances = upper_distances[
            torch.randperm(len(upper_distances))[:100_000]
        ]

    threshold = torch.quantile(upper_distances, q).item()

    # Create edges for all pairs within threshold
    adj = (distances < threshold).triu(diagonal=1)
    src, dst = torch.nonzero(adj, as_tuple=True)

    graph = dgl.graph(
        (torch.cat([src, dst]), torch.cat([dst, src])),
        num_nodes=n_nodes,
    )
    return to_simple(graph)
