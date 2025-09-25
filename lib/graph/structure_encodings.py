import random
import networkx as nx
from collections import deque, defaultdict
from math import log
from sklearn.decomposition import PCA

import dgl
import numpy as np
import torch

# >>> utils


def convert_to_adj_matrix(graph: dgl.DGLGraph) -> list[list[str]]:
    adj = [[] for _ in range(graph.num_nodes())]
    src, dst = graph.edges()
    for u, v in zip(src, dst):
        if u >= v:
            continue
        adj[u].append(v.item())
        adj[v].append(u.item())
    return adj


class QueueWithoutDuplicates:
    def __init__(self):
        self._queue = deque()
        self._set = set()

    def append(self, item):
        if item not in self._set:
            self._queue.append(item)
            self._set.add(item)

    def popleft(self):
        item = self._queue.popleft()
        self._set.remove(item)
        return item

    def __len__(self):
        return len(self._queue)


def adjust_alpha(alpha_orig: float = 0.15) -> float:
    return alpha_orig / (2 - alpha_orig)


# <<< utils


# >>> SAPPR


def sparse_approx_ppr(graph, s, alpha_orig=0.15, eps=1e-3):
    """
    Compute sparse approximate Personalized PageRank for starting distribution s.

    Returns:
        p - a defaultdict (with zero default value) representation of the PPR vector p. The keys are nodes
            and the values are PPR probabilities.
        r - a defaultdict (with zero default value) representation of the r vector at the end of the algorithm.
            The keys are nodes and the values are the probabilities that remain in the r vector for these nodes.
    """
    alpha = adjust_alpha(alpha_orig)

    ### YOUR CODE HERE ###
    p = defaultdict(lambda: 0.0)
    r = defaultdict(lambda: 0.0)

    q = QueueWithoutDuplicates()

    for u, s_value in s.items():
        r[u] = s_value
        if r[u] >= eps * len(graph[u]):
            q.append(u)

    while len(q) > 0:
        u = q.popleft()
        for v in graph[u]:
            r[v] = r[v] + (1 - alpha) * r[u] / (2 * len(graph[u]))
            if r[v] >= eps * len(graph[v]):
                q.append(v)
        p[u] = p[u] + alpha * r[u]
        r[u] = (1 - alpha) * r[u] / 2

    ######################

    return p, r


# <<< SAPPR


# >>> InstantEmbedding


def create_hash_function_d(d, seed, prime=10**9 + 7):
    rng = random.Random(seed)
    a = rng.randint(1, prime)
    b = rng.randint(0, prime)
    return lambda x: ((a * x + b) % prime) % d


def create_hash_function_sgn(seed, prime=10**9 + 7):
    hash_function_2 = create_hash_function_d(d=2, seed=seed, prime=prime)
    return lambda x: hash_function_2(x) * 2 - 1


def node_instant_embedding(
    graph,
    v,
    h_d,
    h_sgn,
    d,
    eps,
):
    """
    Create d-dimensional embedding for node v.

    Returns:
        w - the embedding - a list with d elements.
    """
    ### YOUR CODE HERE ###
    n_vertices = len(graph)

    s = {v: 1.0}
    pi_v, _ = sparse_approx_ppr(graph, s, eps=eps)

    w = [0.0 for _ in range(d)]
    for j, r in pi_v.items():
        w[h_d(j)] += h_sgn(j) * max(log(r * n_vertices), 0)
    ######################

    return w


def compute_instant_embedding(
    graph: dgl.DGLGraph,
    dim: int,
    pca_dim: int | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    hash_function_d = create_hash_function_d(d=dim, seed=42)
    hash_function_sgn = create_hash_function_sgn(seed=42)

    graph_adj = convert_to_adj_matrix(graph)

    embedding = [
        node_instant_embedding(
            graph=graph_adj,
            v=v,
            h_d=hash_function_d,
            h_sgn=hash_function_sgn,
            d=dim,
            eps=eps,
        )
        for v in range(graph.num_nodes())
    ]

    embedding = np.array(embedding)

    if pca_dim is not None:
        embedding = embedding - embedding.mean(0)
        embedding = PCA(pca_dim).fit_transform(embedding)

    return torch.tensor(embedding, dtype=torch.float32)


# <<< InstantEmbedding


# >>> Misc Encodings


def compute_degree_encoding(
    graph: dgl.DGLGraph,
    log: bool = True,
) -> np.ndarray:
    degrees = 0.5 * (graph.in_degrees() + graph.out_degrees())
    if log:
        degrees = torch.log(1 + degrees)
    return degrees[..., None]


@torch.no_grad()
def compute_pagerank(
    graph: dgl.DGLGraph,
    alpha=0.85,
    max_iterations=100,
    tol=1e-6,
    train_index=None,
    log: bool = True,
):
    assert alpha > 0.5, "Please make sure that you provde alpha, not 1-alpha"
    g = graph
    # Initialize node features
    n_nodes = g.num_nodes()
    pv = torch.ones(n_nodes) / n_nodes
    degrees = g.out_degrees().float()

    # Personalization vector (uniform)
    if train_index is None:
        reset_prob = (1 - alpha) / n_nodes
    else:
        reset_prob = torch.zeros(n_nodes)
        reset_prob[train_index] = 1
        reset_prob /= reset_prob.sum()
        reset_prob *= 1 - alpha

    for _ in range(max_iterations):
        # Save old PageRank values
        prev_pv = pv.clone()

        # Message passing
        pv = dgl.ops.copy_u_sum(g, pv / degrees)

        # Update PageRank scores
        pv = alpha * pv + reset_prob

        # Check convergence
        err = torch.abs(pv - prev_pv).sum()
        if err < tol:
            break

    if log:
        pv = torch.log(tol + pv)
    return pv[..., None]


def clustering_coefficient(graph):
    raise NotImplementedError()


def betweenness_centrality(graph):
    graph = nx.Graph(graph.to_networkx())
    bc = nx.betweenness_centrality(graph)
    return torch.tensor([bc[i] for i in range(graph.number_of_nodes())])[..., None]


def eigenvector_centrality(graph):
    graph = nx.Graph(graph.to_networkx())
    centrality = nx.eigenvector_centrality(graph)
    return torch.tensor([centrality[i] for i in range(graph.number_of_nodes())])[
        ..., None
    ]


# <<< Misc Encodings
