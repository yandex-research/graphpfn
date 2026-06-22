"""Graph convolution layers for GNN-based SCM."""

from __future__ import annotations

from typing import Literal

import dgl
import torch
import torch.nn as nn


class GCNConv(nn.Module):
    """GCN-style graph convolution with symmetric normalization."""

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        degrees = torch.clamp(graph.out_degrees().float(), 1.0)
        degree_edge_products = dgl.ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1.0 / degree_edge_products**0.5
        message = dgl.ops.u_mul_e_sum(graph, x, norm_coefs)
        return message


class SAGEConv(nn.Module):
    """GraphSAGE-style aggregation (mean/min/max)."""

    def __init__(self, mode: Literal["mean", "min", "max"] = "mean"):
        super().__init__()
        self.op = {
            "mean": dgl.ops.copy_u_mean,
            "min": dgl.ops.copy_u_min,
            "max": dgl.ops.copy_u_max,
        }[mode]

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        return self.op(graph, x)


class GTConv(nn.Module):
    """Graph Transformer-style attention convolution."""

    def __init__(self, d: int, n_heads: int = 1):
        super().__init__()
        assert d % n_heads == 0

        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.attn_scores_coef = 1.0 / self.d_head**0.5

        self.attn_qkv_linear = nn.Linear(d, d * 3)
        self.output_linear = nn.Linear(d, d)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "Batching is not supported"
        graph = dgl.add_self_loop(graph)

        qkv = self.attn_qkv_linear(x)
        qkv = qkv.reshape(-1, self.n_heads, self.d_head * 3)
        q, k, v = qkv.split((self.d_head, self.d_head, self.d_head), dim=-1)

        attn_scores = dgl.ops.u_dot_v(graph, k, q) * self.attn_scores_coef
        attn_probs = dgl.ops.edge_softmax(graph, attn_scores)

        x = dgl.ops.u_mul_e_sum(graph, v, attn_probs)
        x = x.reshape(-1, self.d)
        x = self.output_linear(x)
        return x


ConvType = Literal["gcn", "sage-mean", "sage-min", "sage-max", "gt"]


def make_conv(conv_type: ConvType, d_output: int) -> nn.Module:
    """Create a graph convolution module by type."""
    if conv_type == "gcn":
        return GCNConv()
    elif conv_type == "sage-mean":
        return SAGEConv("mean")
    elif conv_type == "sage-min":
        return SAGEConv("min")
    elif conv_type == "sage-max":
        return SAGEConv("max")
    elif conv_type == "gt":
        return GTConv(d=d_output)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


class MixedGraphLinear(nn.Module):
    """Hybrid layer mixing linear transform and graph convolution.

    For each output feature, randomly chooses (based on graph_conv_ratio)
    whether to use graph convolution or linear transformation.

    The graph is stored at construction time, allowing uniform `forward(x)`
    signature across all layer types (enabling nn.Sequential for both MLP and GNN).
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        conv_type: ConvType,
        graph_conv_ratio: float,
        graph: dgl.DGLGraph,
    ):
        super().__init__()
        self._graph = graph
        self.linear = nn.Linear(d_input, d_output)
        self.conv = make_conv(conv_type, d_output)
        mask = torch.bernoulli(
            graph_conv_ratio * torch.ones([d_output], dtype=torch.float32)
        ).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        message = self.conv(self._graph, x)
        y = torch.where(self.mask, message, x)
        return y
