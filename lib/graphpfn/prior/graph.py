import abc
import random
from collections.abc import Sequence
from typing import Literal

import dgl
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.sparse.csgraph import connected_components
from torch import Tensor

import lib.util


class GraphSamplerBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        n_nodes: int,
        device: torch.device,
        strict_n_nodes: bool = True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.strict_n_nodes = strict_n_nodes

    @abc.abstractmethod
    def forward(self) -> dgl.DGLGraph:
        raise NotImplementedError()

    def sample(self) -> dgl.DGLGraph:
        graph = self.forward()
        if self.strict_n_nodes:
            assert graph.num_nodes() == self.n_nodes
        assert graph.device == self.device
        return graph


# >>> Utils


def shuffle_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    n_nodes = graph.num_nodes()
    ndata = graph.ndata

    perm = torch.randperm(n_nodes, device=graph.device)
    perm_inv = torch.argsort(perm)

    src, dst = graph.edges()
    src, dst = perm[src], perm[dst]

    for key in list(ndata.keys()):
        ndata[key] = ndata[key][perm_inv, ...]

    graph = dgl.graph((src, dst), num_nodes=n_nodes, device=graph.device)
    return graph


def create_graph_from_edgelist(
    src: Tensor,
    dst: Tensor,
    *,
    device: torch.device | str,
    add_reverse_edges: bool = True,
    n_nodes: int | None = None,
) -> dgl.DGLGraph:
    if n_nodes is None:
        n_nodes = int(max(src.max().item(), dst.max().item())) + 1

    if add_reverse_edges:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])

    graph = dgl.graph(
        (src, dst),
        device=device,
        num_nodes=n_nodes,
    )

    return graph


def dgl_graph_from_adj(
    adj: Tensor,
    *,
    device: torch.device,
    undirected: bool = True,
) -> dgl.DGLGraph:
    n_nodes = adj.shape[0]
    if undirected:
        adj = adj.triu()
        adj = torch.where(
            torch.eye(n_nodes, device=adj.device, dtype=torch.bool),
            0,
            adj,
        )
    src, dst = torch.nonzero(adj, as_tuple=True)
    return create_graph_from_edgelist(
        src,
        dst,
        device=device,
        add_reverse_edges=True,
        n_nodes=n_nodes,
    )


def adjust_intensity(
    intensity: Tensor,
    max_binsearch_iters: int = 1_000,
    rtol: float = 1e-4,
) -> Tensor:
    intensity = intensity.double()
    initial_sum = intensity.sum()
    f = lambda c: (c * intensity).clip(max=1.0).sum().item()  # noqa: E731
    c_lb = 1.0
    c_rb = 1 / intensity[intensity > 0].min()

    if (intensity > 0).float().sum() < (1 + rtol) * initial_sum:
        intensity = (c_rb * intensity).clip(max=1)
        return intensity

    assert f(c_lb) <= (1 + rtol) * initial_sum
    assert f(c_rb) >= (1 - rtol) * initial_sum, (
        f(c_rb),
        c_rb,
        intensity[intensity > 0].min(),
        initial_sum,
        (intensity > 0).float().sum(),
    )

    for _ in range(max_binsearch_iters):
        c_mid = (c_lb + c_rb) / 2
        f_value = f(c_mid)
        if abs(f_value - initial_sum) < rtol * initial_sum:
            break
        elif f_value < initial_sum:
            c_lb = c_mid
        else:
            c_rb = c_mid

    c = (c_lb + c_rb) / 2
    intensity = (c * intensity).clip(max=1)
    return intensity


# TODO: rewrite to binary search
def sample_from_intensity(
    intensity: Tensor,
) -> Tensor:
    intensity = adjust_intensity(intensity).clip(max=1)
    adj = (torch.rand_like(intensity) < intensity).to(torch.int32)
    return adj


def random_partition(
    n: int,
    n_parts: int,
    *,
    min_size: int = 1,
    min_part_ratio: float | None = 0.1,  # TODO: maybe rename
) -> list[int]:
    """
    Generates a random partition: list of positive integers of len n_parts summing to n
    """
    if n_parts < 1 or n < n_parts:
        raise ValueError("Require 1 <= n_parts <= n.")
    if n_parts == 1:
        return [n]

    if min_part_ratio is not None:
        min_size = max(
            min_size,
            int((n / n_parts) * min_part_ratio),
        )

    assert min_size * n_parts <= n
    n -= (min_size - 1) * n_parts

    cuts = np.sort(np.random.choice(np.arange(1, n), size=n_parts - 1, replace=False))
    parts = np.diff(np.concatenate(([0], cuts, [n]))).astype(np.int32)
    parts = (min_size - 1) + parts
    return parts.tolist()  # type: ignore


def random_float(low: float, high: float) -> float:
    return np.random.random() * (high - low) + low


def random_bool(p: float = 0.5) -> bool:
    return np.random.random() < p


# <<<


# >>> Base Graph Samplers


class LineSampler(GraphSamplerBase):
    def __init__(self, n_nodes: int, device: torch.device):
        super().__init__(n_nodes, device)

    def forward(self) -> dgl.DGLGraph:
        nodes = torch.randperm(self.n_nodes, device=self.device)
        src, dst = nodes[:-1], nodes[1:]
        graph = dgl.graph(
            (torch.cat([src, dst]), torch.cat([dst, src])),
            device=self.device,
            num_nodes=self.n_nodes,
        )
        return graph


# TODO: maybe optimize
# TODO: rewrite to Prüfer sequence
class TreeSampler(GraphSamplerBase):
    def __init__(self, n_nodes: int, device: torch.device):
        super().__init__(n_nodes, device)

    def forward(self) -> dgl.DGLGraph:
        src = torch.arange(1, self.n_nodes, device=self.device)
        dst = []

        for i in range(1, self.n_nodes):
            dst.append(np.random.randint(0, i))

        dst = torch.tensor(dst, device=self.device)

        graph = create_graph_from_edgelist(
            src,
            dst,
            device=self.device,
            n_nodes=self.n_nodes,
        )
        graph = shuffle_graph(graph)
        return graph


class ERSampler(GraphSamplerBase):
    def __init__(self, n_nodes: int, avg_degree: float, device: torch.device):
        super().__init__(n_nodes, device)
        self.n_edges = int(n_nodes * avg_degree)

    def forward(self) -> dgl.DGLGraph:
        graph = dgl.rand_graph(
            num_nodes=self.n_nodes,
            num_edges=self.n_edges,
            device=self.device,
        )
        return graph


# TODO: double-check and refactor
class DegreeCorrectedSBMSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        min_n_groups: int = 2,
        max_n_groups: int = 8,
        p_uniform_density: float = 0.5,
        p_power_law_degrees: float = 0.5,
        avg_degree: float,
        device: torch.device,
        offdiagonal_coef: float = 0.1,
    ):
        super().__init__(n_nodes, device)
        self.avg_degree = avg_degree
        self.offdiagonal_coef = offdiagonal_coef
        self.min_n_groups = min_n_groups
        self.max_n_groups = max_n_groups
        self.p_uniform_density = p_uniform_density
        self.p_power_law_degrees = p_power_law_degrees

    def get_groups(self, group_sizes: list[int]) -> Tensor:
        n_groups = len(group_sizes)
        groups = torch.cat(
            [
                torch.full(
                    [group_size], group_idx, dtype=torch.int32, device=self.device
                )
                for group_idx, group_size in enumerate(group_sizes)
            ],
            dim=0,
        )

        for i, group_size in enumerate(group_sizes):
            group_start = sum(group_sizes[:i])
            group_end = group_start + group_size
            groups[group_start:group_end] = i

        assert groups.ndim == 1
        assert groups.shape[0] == self.n_nodes
        assert groups.min().item() == 0
        assert groups.max().item() == n_groups - 1
        assert len(torch.unique(groups)) == n_groups
        return groups

    # TODO: rename
    def get_n_edges_expected(self, group_sizes: list[int]) -> Tensor:
        n_groups = len(group_sizes)
        uniform_density = random_bool(p=self.p_uniform_density)

        # >>> Generate n_edges_expected
        n_edges_expected = torch.zeros(
            [n_groups, n_groups],
            dtype=torch.float32,
            device=self.device,
        )

        # Generate diagonal values
        for i in range(n_groups):
            n_edges_expected[i, i] = 1.0 if uniform_density else np.random.random()

        # Generate off-diagonal values
        uniform_offdiagonal_density = np.random.random()
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                max_prob = (n_edges_expected[i, i] * n_edges_expected[j, j]) ** 0.5
                n_edges_expected[i, j] = self.offdiagonal_coef * (
                    uniform_offdiagonal_density
                    if uniform_density
                    else np.random.random() * max_prob
                )

        # >>> Normalize & return
        group_sizes_tensor = torch.tensor(group_sizes, device=self.device)
        n_edges_expected = (
            n_edges_expected * group_sizes_tensor[:, None] * group_sizes_tensor[None, :]
        )
        n_edges_expected = (
            n_edges_expected + n_edges_expected.T - torch.diag(n_edges_expected.diag())
        )

        return n_edges_expected

    def get_degs(self, group_sizes: list[int]) -> Tensor:
        # >>> Generate degs
        if random_bool(self.p_power_law_degrees):
            gamma = random_float(2, 3)
            degs = torch.tensor(
                np.random.zipf(gamma, [self.n_nodes]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            degs = torch.rand([self.n_nodes], dtype=torch.float32, device=self.device)

        # >>> Normalize
        for i, group_size in enumerate(group_sizes):
            group_start = sum(group_sizes[:i])
            group_end = group_start + group_size
            degs[group_start:group_end] /= degs[group_start:group_end].sum()

        # >>> Sanity checks & return
        assert torch.all(degs >= 0)
        return degs

    def sample_adjacency_matrix(
        self,
        groups: Tensor,
        n_edges_expected: Tensor,
        degs: Tensor,
    ) -> Tensor:
        intensity = (
            n_edges_expected[groups[:, None], groups[None, :]]
            * degs[:, None]
            * degs[None, :]
        ).triu()
        intensity = intensity / intensity.sum()
        intensity = (0.5 * self.n_nodes * self.avg_degree) * intensity

        # Sanity checks
        avg_edges = intensity.sum().item()
        avg_degree = 2 * avg_edges / self.n_nodes
        assert torch.allclose(
            torch.tensor(avg_degree),
            torch.tensor(self.avg_degree, dtype=torch.float32),
        )
        assert torch.allclose(intensity.triu(), intensity)
        assert torch.all(intensity >= 0)

        # Sample & return
        adj = sample_from_intensity(intensity)
        return adj

    def forward(self) -> dgl.DGLGraph:
        n_groups = np.random.randint(self.min_n_groups, self.max_n_groups)
        group_sizes = random_partition(self.n_nodes, n_groups)
        groups = self.get_groups(group_sizes)
        n_edges_expected = self.get_n_edges_expected(group_sizes)
        degs = self.get_degs(group_sizes)
        adj = self.sample_adjacency_matrix(groups, n_edges_expected, degs)
        graph = dgl_graph_from_adj(adj, device=self.device)
        graph = shuffle_graph(graph)
        return graph


# TODO: maybe optimize
class PreferentialAttachmentSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        device: torch.device,
        *,
        max_degree: int | None = None,
        base: GraphSamplerBase | None = None,
        avg_degree: float | None = None,
    ):
        assert (max_degree is None) ^ (avg_degree is None)
        if max_degree is None:
            assert avg_degree is not None
            max_degree = int(2 * avg_degree - 1)
        if base is None:
            self.n_nodes_pa = n_nodes
            strict_n_nodes = True
        else:
            assert base.n_nodes <= n_nodes
            self.n_nodes_pa = n_nodes - base.n_nodes
            strict_n_nodes = base.strict_n_nodes

        super().__init__(n_nodes, device, strict_n_nodes)
        self.base = base
        self.max_degree = max_degree

    def forward(self) -> dgl.DGLGraph:
        n_nodes = self.n_nodes_pa

        if self.base is not None:
            base_graph = self.base.sample()
            src, dst = base_graph.edges()
            src, dst = src.tolist(), dst.tolist()
            degrees = base_graph.in_degrees().tolist()
            n_nodes += base_graph.num_nodes()
        else:
            src, dst = [0], [1]
            degrees = [1, 1]

        for _ in range(len(degrees), n_nodes):
            u = len(degrees)
            u_degree = random.randint(1, self.max_degree)
            u_degree = min(u_degree, len(degrees))
            p = np.array(degrees)
            p = p / p.sum()
            v_list = np.random.choice(len(degrees), [u_degree], replace=False, p=p)
            for v in v_list:
                src.append(u)
                src.append(v)
                dst.append(v)
                dst.append(u)
                degrees[v] += 1
            degrees.append(u_degree)

        graph = create_graph_from_edgelist(
            torch.tensor(src, device=self.device),
            torch.tensor(dst, device=self.device),
            device=self.device,
            n_nodes=n_nodes,
            add_reverse_edges=False,
        )
        graph = shuffle_graph(graph)
        return graph


class GeometricGraphSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        min_n_latent_features: int = 2,
        max_n_latent_features: int = 16,
        device: torch.device,
    ):
        super().__init__(n_nodes, device)

        self.avg_degree = avg_degree
        self.min_n_latent_features = min_n_latent_features
        self.max_n_latent_features = max_n_latent_features

    def generate_points(self) -> Tensor:
        n_latent_features = np.random.randint(
            self.min_n_latent_features,
            self.max_n_latent_features + 1,
        )
        if random_bool(p=0.5):
            random_generator = torch.rand
        else:
            random_generator = torch.randn
        points = random_generator((self.n_nodes, n_latent_features), device=self.device)
        return points

    def compute_distances(self, points: Tensor, how: Literal["l2"] = "l2") -> Tensor:
        if how == "l2":
            # [None, ...] and [0, ...] are here since cdist assumes batched input
            distances = torch.cdist(points[None, ...], points[None, ...])[0, ...]
        else:
            raise ValueError(f"Unknown type: {how}")
        return distances

    def compute_threshold(
        self,
        distances: Tensor,
        subsample_size: int = 100_000,
    ) -> float:
        n_edges = 0.5 * (self.n_nodes * self.avg_degree)
        distances = distances.triu()
        distances = distances[distances > 0.0]
        assert distances.ndim == 1
        q = n_edges / distances.shape[0]
        if distances.shape[0] > subsample_size:
            perm = torch.randperm(distances.shape[0], device=distances.device)
            distances = distances[perm]
            distances = distances[:subsample_size]
        return torch.quantile(distances, q).item()

    def forward(self) -> dgl.DGLGraph:
        points = self.generate_points()
        distances = self.compute_distances(points)
        threshold = self.compute_threshold(distances)
        adj = (distances < threshold).to(torch.int32)
        graph = dgl_graph_from_adj(adj, device=self.device)
        graph.ndata["features"] = points
        return graph


# <<<


# >>> Util Samplers & Postprocessing


# TODO: refactor everything to functional style
def extract_largest_component(graph: dgl.DGLGraph) -> tuple[dgl.DGLGraph, Tensor]:
    nodes = torch.arange(graph.num_nodes(), device=graph.device)

    adj_scipy = graph.adj_external(scipy_fmt="csr")
    n_components, labels = connected_components(adj_scipy, directed=False)

    if n_components == 1:
        return graph, nodes

    labels = torch.tensor(labels, device=graph.device)
    largest_component_idx = torch.bincount(labels).argmax()
    component_nodes = nodes[labels == largest_component_idx]
    graph = dgl.node_subgraph(graph, component_nodes)

    return graph, component_nodes


class ExtractLargestComponent(GraphSamplerBase):
    def __init__(self, base: GraphSamplerBase):
        super().__init__(base.n_nodes, base.device, strict_n_nodes=False)
        self.base = base

    def forward(self) -> dgl.DGLGraph:
        graph = self.base.sample()
        graph, _ = extract_largest_component(graph)
        return graph


class ToSimple(GraphSamplerBase):
    def __init__(self, base: GraphSamplerBase):
        super().__init__(base.n_nodes, base.device, base.strict_n_nodes)
        self.base = base

    def forward(self) -> dgl.DGLGraph:
        graph = self.base.sample().to("cpu")
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.to_simple(graph, copy_ndata=True)
        graph = dgl.remove_self_loop(graph)
        graph = graph.to(self.device)
        return graph


class DisjointUnion(GraphSamplerBase):
    def __init__(self, samplers: Sequence[GraphSamplerBase]):
        n_nodes = sum([g.n_nodes for g in samplers])
        device = samplers[0].device
        assert all([g.device == device for g in samplers])

        super().__init__(n_nodes, device)
        self.samplers = samplers

    def forward(self) -> dgl.DGLGraph:
        graph = dgl.batch([sampler.sample() for sampler in self.samplers])
        graph = shuffle_graph(graph)
        return graph


class Merge(GraphSamplerBase):
    def __init__(self, samplers: Sequence[GraphSamplerBase]):
        n_nodes = samplers[0].n_nodes
        device = samplers[0].device
        assert all([g.device == device for g in samplers])
        assert all([g.n_nodes == n_nodes for g in samplers])

        super().__init__(n_nodes, device)
        self.samplers = samplers

    def forward(self) -> dgl.DGLGraph:
        return dgl.merge([sampler.sample() for sampler in self.samplers])


class DistributedSampler(GraphSamplerBase):
    def __init__(self, base: GraphSamplerBase):
        super().__init__(base.n_nodes, base.device, base.strict_n_nodes)
        self.base = base

    def forward(self) -> dgl.DGLGraph:
        if not lib.util.is_ddp():
            return self.base.sample()

        if lib.util.is_master_process():
            graph = self.base.sample()
            edges = torch.stack(graph.edges(), dim=0)
            n_nodes = graph.num_nodes()
            if "features" in graph.ndata:
                features = graph.ndata["features"]
                assert features.ndim == 2  # type: ignore
                n_features = features.shape[1]  # type: ignore
            else:
                features = torch.empty([0, 0], device=self.device, dtype=torch.float32)
                n_features = 0
        else:
            edges = torch.empty([0, 0], device=self.device, dtype=torch.int64)
            features = torch.empty([0, 0], device=self.device, dtype=torch.float32)
            n_nodes = None
            n_features = None

        n_nodes = lib.util.broadcast_int(n_nodes)
        n_edges = lib.util.broadcast_int(edges.shape[1])
        n_features = lib.util.broadcast_int(n_features)

        if not lib.util.is_master_process():
            edges = torch.empty([2, n_edges], device=self.device, dtype=torch.int64)
            features = torch.empty(
                [n_nodes, n_features], device=self.device, dtype=torch.float32
            )

        torch.distributed.broadcast(edges, src=0)  # type: ignore
        if n_features > 0:
            torch.distributed.broadcast(features, src=0)  # type: ignore

        graph = dgl.graph((edges[0, :], edges[1, :]), num_nodes=n_nodes)
        if n_features > 0:
            graph.ndata["features"] = features
        graph = graph.to(self.device)
        return graph


# <<<


# >>> End-to-end samplers


class MultiLevelSbmWithPaSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        min_pa_nodes_ratio: float = 0.0,
        max_pa_nodes_ratio: float = 0.3,
        max_pa_degree: int = 2,
        min_n_first_level_subgraphs: int = 2,
        max_n_first_level_subgraphs: int = 8,
        min_first_level_degree_ratio: float = 0.05,
        max_first_level_degree_ratio: float = 0.95,
        min_first_level_n_nodes: int = 64,
        device: torch.device | str,
    ):
        device = torch.device(device)

        super().__init__(n_nodes, device, strict_n_nodes=False)

        pa_nodes_ratio = random_float(min_pa_nodes_ratio, max_pa_nodes_ratio)
        n_nodes_before_pa = int(n_nodes * (1.0 - pa_nodes_ratio))

        pa_avg_degree = (1.0 + max_pa_degree) / 2
        avg_degree = (avg_degree - pa_nodes_ratio * pa_avg_degree) / (
            1.0 - pa_nodes_ratio
        )

        max_first_level_degree_ratio = min(
            max_first_level_degree_ratio, (min_first_level_n_nodes - 1) / avg_degree
        )
        first_level_degree_ratio = random_float(
            min_first_level_degree_ratio, max_first_level_degree_ratio
        )
        first_level_avg_degree = avg_degree * first_level_degree_ratio
        second_level_avg_degree = avg_degree * (1.0 - first_level_degree_ratio)

        # First-level sampler
        n_first_level_subgraphs = np.random.randint(
            min_n_first_level_subgraphs,
            max_n_first_level_subgraphs + 1,
        )

        first_level_subgraph_samplers = [
            DegreeCorrectedSBMSampler(
                n_nodes_subgraph,
                avg_degree=first_level_avg_degree,
                device=device,
            )
            for n_nodes_subgraph in random_partition(
                n=n_nodes_before_pa,
                n_parts=n_first_level_subgraphs,
                min_size=min_first_level_n_nodes,
            )
        ]

        first_level_sampler = DisjointUnion(first_level_subgraph_samplers)

        # Second-level sampler
        second_level_sampler = DegreeCorrectedSBMSampler(
            n_nodes_before_pa,
            avg_degree=second_level_avg_degree,
            device=device,
        )

        # Merge & apply PA
        sampler = Merge([first_level_sampler, second_level_sampler])
        sampler = ExtractLargestComponent(sampler)
        sampler = PreferentialAttachmentSampler(
            n_nodes,
            device,
            max_degree=max_pa_degree,
            base=sampler,
        )

        self.sampler = sampler

    def forward(self) -> dgl.DGLGraph:
        return self.sampler.sample()


class GraphSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        sampler_type_list: list[str],
        device: torch.device | str,
    ):
        device = torch.device(device)
        super().__init__(n_nodes, device, strict_n_nodes=False)

        sampler_type = random.choice(sampler_type_list)
        sampler = {
            "multi-level-sbm-with-pa": MultiLevelSbmWithPaSampler,
            "sbm": DegreeCorrectedSBMSampler,
            "geometric": GeometricGraphSampler,
            "preferential-attachment": PreferentialAttachmentSampler,
            "erdos-renyi": ERSampler,
        }[sampler_type](n_nodes, avg_degree=avg_degree, device=device)  # type: ignore

        # Postprocessing & sanity checks
        sampler = ExtractLargestComponent(sampler)
        sampler = ToSimple(sampler)

        assert sampler.n_nodes == n_nodes
        self.sampler = sampler

    def forward(self) -> dgl.DGLGraph:
        return self.sampler.sample()


# <<<
