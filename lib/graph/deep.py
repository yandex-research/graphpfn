from functools import partial

import dgl
import rtdl_num_embeddings as E
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from dgl import ops
from torch import Tensor


def get_activation_module(activation: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }[activation]()


class ResidualModule(nn.Module):
    def __init__(
        self,
        base_class: nn.Module,
        norm_class: nn.Module,
        d_hidden: int,
        *,
        residual: bool = True,
        **module_kwargs,
    ):
        super().__init__()
        self.norm = norm_class(d_hidden)
        self.base = base_class(d=d_hidden, **module_kwargs)
        self.residual = residual

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        x_res = self.norm(x)
        x_res = self.base(graph, x_res, edge_weights)
        x = x + x_res if self.residual else x_res
        return x


class MLPModule(nn.Module):
    def __init__(
        self, d: int, dropout: float = 0.0, activation: str = "relu", **kwargs
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d),
            nn.Dropout(dropout),
            get_activation_module(activation),
        )

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        x = self.layers(x)
        return x


class FFNModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_inputs: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d * n_inputs, d),
            nn.Dropout(dropout),
            get_activation_module(activation),
            nn.Linear(d, d),
            nn.Dropout(dropout),
        )

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        x = self.layers(x)
        return x


class GCNModule(nn.Module):
    def __init__(
        self, d: int, dropout: float = 0.0, activation: str = "relu", **kwargs
    ):
        super().__init__()
        self.ffn = FFNModule(d, dropout=dropout, activation=activation)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        degrees = torch.clamp(graph.out_degrees().float(), 1.0)
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1.0 / degree_edge_products**0.5

        if edge_weights is not None:
            norm_coefs *= edge_weights

        x = ops.u_mul_e_sum(graph, x, norm_coefs)
        x = self.ffn(graph, x)
        return x


class GCNSepModule(nn.Module):
    def __init__(
        self, d: int, dropout: float = 0.0, activation: str = "relu", **kwargs
    ):
        super().__init__()
        self.ffn = FFNModule(d, n_inputs=2, dropout=dropout, activation=activation)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        degrees = torch.clamp(graph.out_degrees().float(), 1.0).to(x.dtype)
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1.0 / degree_edge_products**0.5

        if edge_weights is not None:
            norm_coefs *= edge_weights

        message = ops.u_mul_e_sum(graph, x, norm_coefs)
        x = torch.cat([x, message], axis=1)
        x = self.ffn(graph, x)
        return x


class GraphSAGEModule(nn.Module):
    def __init__(
        self, d: int, dropout: float = 0.0, activation: str = "relu", **kwargs
    ):
        super().__init__()
        self.ffn = FFNModule(d, n_inputs=2, dropout=dropout, activation=activation)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        message = (
            ops.copy_u_mean(graph, x)
            if edge_weights is None
            else ops.u_mul_e_mean(graph, x, edge_weights)
        )
        x = torch.cat([x, message], axis=1)
        x = self.ffn(graph, x)
        return x


def _check_d_and_n_heads_consistency(d: int, n_heads: int) -> None:
    if d % n_heads != 0:
        raise ValueError("Dimension mismatch: d should be a multiple of n_head.")


class GATModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        _check_d_and_n_heads_consistency(d, n_heads)
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.input_linear = nn.Linear(d, d)
        self.attn_linear_u = nn.Linear(d, n_heads)
        self.attn_linear_v = nn.Linear(d, n_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.ffn = FFNModule(d, dropout=dropout, activation=activation)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        assert edge_weights is None, "So far, we can not apply edge weights."

        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.d_head, self.n_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.d)
        x = self.ffn(graph, x)
        return x


class GATSepModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        _check_d_and_n_heads_consistency(d, n_heads)
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.input_linear = nn.Linear(d, d)
        self.attn_linear_u = nn.Linear(d, n_heads)
        self.attn_linear_v = nn.Linear(d, n_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)
        self.ffn = FFNModule(d, n_inputs=2, dropout=dropout, activation=activation)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        assert edge_weights is None, "So far, we can not apply edge weights."

        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.d_head, self.n_heads)
        message: Tensor = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.d)
        message = message.reshape(-1, self.d)
        x = torch.cat([x, message], axis=1)
        x = self.ffn(graph, x)
        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.0, **kwargs):
        super().__init__()

        _check_d_and_n_heads_consistency(d, n_heads)
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.attn_scores_coef = 1.0 / self.d_head**0.5

        self.attn_qkv_linear = nn.Linear(d, d * 3)
        self.output_linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        assert edge_weights is None, "So far, we can not apply edge weights."

        qkv: Tensor = self.attn_qkv_linear(x)
        qkv = qkv.reshape(-1, self.n_heads, self.d_head * 3)
        q, k, v = qkv.split(split_size=(self.d_head, self.d_head, self.d_head), dim=-1)

        attn_scores = ops.u_dot_v(graph, k, q) * self.attn_scores_coef
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, v, attn_probs)
        x = x.reshape(-1, self.d)

        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class TransformerAttentionSepModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        _check_d_and_n_heads_consistency(d, n_heads)
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.attn_scores_coef = 1.0 / self.d_head**0.5

        self.attn_qkv_linear = nn.Linear(d, d * 3)
        self.output_linear = nn.Linear(d * 2, d)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        assert edge_weights is None, "So far, we can not apply edge weights."

        qkv: Tensor = self.attn_qkv_linear(x)
        qkv = qkv.reshape(-1, self.n_heads, self.d_head * 3)
        q, k, v = qkv.split(split_size=(self.d_head, self.d_head, self.d_head), dim=-1)

        attn_scores = ops.u_dot_v(graph, k, q) * self.attn_scores_coef
        attn_probs = ops.edge_softmax(graph, attn_scores)

        message: Tensor = ops.u_mul_e_sum(graph, v, attn_probs)
        message = message.reshape(-1, self.d)
        x = torch.cat([x, message], axis=1)

        x = self.output_linear(x)
        x = self.dropout(x)
        return x


CONVOLUTIONAL_MODULES = {
    "mlp": [MLPModule],
    "resnet": [FFNModule],
    "sage": [GraphSAGEModule],
    "gcn": [GCNModule],
    "gcn-sep": [GCNSepModule],
    "gat": [GATModule],
    "gat-sep": [GATSepModule],
    "gt": [TransformerAttentionModule, FFNModule],
    "gt-sep": [TransformerAttentionSepModule, FFNModule],
}

NORM_MODULES = {
    "none": nn.Identity,
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
}


class BaseGraphBackbone(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_hidden: int,
        conv_name: str,
        norm_name: str,
        activation: str,
        *,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        norm_class = NORM_MODULES[norm_name]

        for _ in range(n_layers):
            for base_class in CONVOLUTIONAL_MODULES[conv_name]:
                self.layers.append(
                    ResidualModule(
                        base_class,
                        norm_class,
                        d_hidden=d_hidden,
                        residual=residual,
                        dropout=dropout,
                        activation=activation,
                    )
                )

    def forward(
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(graph, x, edge_weights)
        return x


class BaseNumFeaturesModule(nn.Module):
    def __init__(
        self,
        name: str,
        *,
        n_features: Tensor,
        bin_edges: list[Tensor] | None = None,
        **num_features_kwargs,
    ):
        super().__init__()

        if name == "piecewise":
            assert bin_edges is not None
            self.module = E.PiecewiseLinearEmbeddings(
                bins=bin_edges, **num_features_kwargs
            )
        elif name == "periodic":
            self.module = E.PeriodicEmbeddings(
                n_features=n_features, **num_features_kwargs
            )
        else:
            raise ValueError(f"Unknown {name=}")

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x).flatten(start_dim=1)


GRAPH_MODULES = {
    module_class.__name__: module_class
    for module_class in [
        BaseGraphBackbone,
        BaseNumFeaturesModule,
    ]
}


def make_module(type: str, *args, **kwargs):
    graph_module = GRAPH_MODULES[type]
    return graph_module(*args, **kwargs)
