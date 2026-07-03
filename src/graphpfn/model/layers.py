import warnings

import dgl
import torch
from torch import Tensor, nn

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*is deprecated.*",
    module=r"dgl\.backend\.pytorch\.sparse",
)


class GraphHolder:
    __slots__ = ("graph",)
    graph: dgl.DGLGraph | None

    def __init__(self) -> None:
        self.graph = None


class GraphPFNResidualModule(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_hidden: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.base = base

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        x_res = self.norm(x)
        x_res = self.base(graph, x_res)
        return x + x_res


class GraphPFNLayerWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        graph_holder: GraphHolder,
        zero_init: bool = True,
    ):
        super().__init__()
        self._graph_holder = graph_holder

        self.base = base
        self.conv = GraphPFNResidualModule(
            base=GraphPFNGraphAttentionModule(d=192, zero_init=zero_init),
            d_hidden=192,
        )
        self.mlp = GraphPFNResidualModule(
            base=GraphPFNMLPModule(d=192, zero_init=zero_init),
            d_hidden=192,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_atten_mask: torch.Tensor,
        eval_pos: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        graph = self._graph_holder.graph
        assert graph is not None

        # >>> Apply base layer
        x, feature_attenion, sample_attention = self.base(
            x=x,
            feature_atten_mask=feature_atten_mask,
            eval_pos=eval_pos,
            layer_idx=layer_idx,
        )

        # >>> Apply GNN & MLP layers
        x = self.conv(graph, x)
        x = self.mlp(graph, x)

        # >>> Return
        return x, feature_attenion, sample_attention


class GraphPFNGraphAttentionModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()

        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.attn_scores_coef = 1.0 / self.d_head**0.5

        self.attn_qkv_linear = nn.Linear(d, d * 3)
        self.output_linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(p=dropout)

        if zero_init:
            torch.nn.init.zeros_(self.output_linear.weight)
            torch.nn.init.zeros_(self.output_linear.bias)

    def forward(
        self,
        graph: dgl.DGLGraph,
        x: Tensor,
    ) -> Tensor:
        assert x.ndim == 4
        assert x.shape[0] == 1, "Batches are not supported yet"

        x = x.squeeze(0)
        x_shape = x.shape
        qkv: Tensor = self.attn_qkv_linear(x)
        qkv = qkv.reshape(*x_shape[:-1], self.n_heads, self.d_head * 3)
        q, k, v = qkv.split(split_size=(self.d_head, self.d_head, self.d_head), dim=-1)

        attn_scores = dgl.ops.u_dot_v(graph, k, q) * self.attn_scores_coef  # type: ignore
        attn_probs = dgl.ops.edge_softmax(graph, attn_scores)
        x = dgl.ops.u_mul_e_sum(graph, v, attn_probs)  # type: ignore

        x = x.reshape(*x_shape[:-1], self.d)

        x = self.output_linear(x)
        x = self.dropout(x)
        x = x.unsqueeze(0)
        return x


class GraphPFNMLPModule(nn.Module):
    def __init__(self, d: int, zero_init: bool = True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

        if zero_init:
            torch.nn.init.zeros_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(
        self,
        graph: dgl.DGLGraph,
        x: Tensor,
    ) -> Tensor:
        x = self.layers(x)
        return x
