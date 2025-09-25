from functools import partial

import dgl
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from typing import NotRequired
from typing_extensions import TypedDict

import lib.tfm
import lib.graph.deep
from lib.limix.model.layer import MultiheadAttention as MHA
from lib.util import TaskType


class GraphPFNOutput(TypedDict):
    predictions: Tensor
    features_pred: Tensor
    edge_predictions: NotRequired[Tensor]


class GraphPFN(nn.Module):
    def __init__(
        self,
        edge_head: bool,
        layer_ids=list(range(12)),
    ) -> None:
        super().__init__()
        self.tfm = lib.tfm.load_tfm(tfm_name="LimiX", tfm_config={})
        self.tfm.mask_prediction = True

        for idx in layer_ids:
            layer = self.tfm.transformer_encoder.layers[idx]
            wrapped_layer = GraphPFNLayerWrapper(base=layer)
            self.tfm.transformer_encoder.layers[idx] = wrapped_layer

        # >>> By default, we freeze all params of TFM
        for param in self.tfm.parameters():
            param.requires_grad = False

        for idx in layer_ids:
            wrapped_layer = self.tfm.transformer_encoder.layers[idx]
            layer_params = [
                *wrapped_layer.mlp.parameters(),
                *wrapped_layer.conv.parameters(),
            ]
            for param in layer_params:
                param.requires_grad = True

        # >>> We also have a separate head for edge reconstruction
        if edge_head:
            self.edge_head = nn.Sequential(
                nn.Linear(self.tfm.embed_dim, self.tfm.hid_dim),
                nn.LayerNorm(self.tfm.hid_dim),
                nn.GELU(),
                nn.Linear(self.tfm.hid_dim, 1),
            )

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: Tensor,
        y_train: Tensor,
        train_mask: Tensor,
        task_type: TaskType,
        *,
        edges: tuple[Tensor, Tensor] | None = None,
        checkpointing: bool = True,
        # Seems like efficient attn kernels do not support indices larger than 2**16,
        # so we need batched version for such cases.
        batched_attn: bool = False,
    ) -> GraphPFNOutput:
        assert features.ndim == 2
        assert y_train.ndim == 1
        assert train_mask.ndim == 1
        assert y_train.shape[0] == train_mask.int().sum().item()

        # TFM input and targets
        tfm_features = torch.cat(
            [
                features[train_mask, ...],
                features[~train_mask, ...],
            ],
            dim=0,
        )

        for module in self.modules():
            if isinstance(module, GraphPFNLayerWrapper):
                module.train_mask = train_mask
                module.graph = graph
            if isinstance(module, MHA):
                module.batched = batched_attn

        if batched_attn:
            sdpa_backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            sdpa_backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]

        with torch.nn.attention.sdpa_kernel(sdpa_backends):
            out = self.tfm.forward(
                x=tfm_features.unsqueeze(0),
                y=y_train.unsqueeze(0),
                eval_pos=y_train.shape[0],
                task_type="reg" if (task_type == TaskType.REGRESSION) else "cls",
                checkpointing=checkpointing,
            )

        # >>> Extract everything
        num_nodes = graph.num_nodes()
        inv_order = torch.argsort((~train_mask).float(), stable=True)
        order = torch.argsort(inv_order, stable=True)

        # Extract preds
        pred = (
            (
                out["reg_output"]
                if task_type == TaskType.REGRESSION
                else out["cls_output"]
            )
            .float()
            .squeeze(0)
        )
        dummy_pred = pred.new_zeros([num_nodes - pred.shape[0], *pred.shape[1:]])
        pred = torch.cat([dummy_pred, pred], dim=0)
        pred = pred[order, ...]

        if task_type == TaskType.REGRESSION:
            pred = pred.squeeze(-1)

        # Extract feature pred
        extract_feat = (  # noqa E731
            lambda x: x.reshape(*x.shape[:-2], -1)[..., : features.shape[-1]]
        )
        features_pred = extract_feat(out["feature_pred"].squeeze(0)[order, ...])
        feature_mean = extract_feat(out["process_config"]["mean_for_normalization"])
        feature_std = extract_feat(out["process_config"]["std_for_normalization"])
        features_pred = features_pred * feature_std + feature_mean

        # Extract encoder embeddings
        encoder_embed = out["encoder_embed"].squeeze(0)[order, ...]
        if edges is not None:
            src, dst = edges
            edge_embeds = encoder_embed[src, :] * encoder_embed[dst, :]
            edge_predictions = self.edge_head(edge_embeds)
        else:
            edge_predictions = None

        # Check that no features were filtered
        num_used_features = out["process_config"]["num_used_features"].sum().item()
        if num_used_features != features.shape[-1]:
            logger.error(f"{num_used_features=}, while {features.shape[-1]=}")

        return {
            "predictions": pred,
            "features_pred": features_pred,
            "edge_predictions": edge_predictions,
        }


class GraphPFNLayerWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        zero_init: bool = True,
    ):
        super().__init__()
        self.graph: dgl.DGLGraph | None = None  # placeholder
        self.train_mask: Tensor | None = None  # placeholder

        self.base = base
        self.conv = lib.graph.deep.ResidualModule(
            base_class=partial(GraphPFNGraphAttentionModule, zero_init=zero_init),
            norm_class=nn.LayerNorm,
            d_hidden=192,
            activation="gelu",
        )
        self.mlp = lib.graph.deep.ResidualModule(
            base_class=partial(GraphPFNMLPModule, zero_init=zero_init),
            norm_class=nn.LayerNorm,
            d_hidden=192,
            activation="gelu",
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_atten_mask: torch.Tensor,
        eval_pos: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # >>> Apply base layer
        x, feature_attenion, sample_attention = self.base(
            x=x,
            feature_atten_mask=feature_atten_mask,
            eval_pos=eval_pos,
            layer_idx=layer_idx,
        )

        # >>> Apply GNN layer
        # This is tricky: x has train samples first, while GNN needs initial
        # ordering. So we reorder to initial ordering and backs.
        assert self.train_mask is not None
        inv_order = torch.argsort((~self.train_mask).float(), stable=True)
        order = torch.argsort(inv_order, stable=True)

        assert x.shape[0] == 1
        x = x.squeeze(0)
        x = x[order, ...]
        x = self.conv(self.graph, x)
        x = x[inv_order, ...]
        x = x.unsqueeze(0)
        x = self.mlp(self.graph, x)

        # >>> Return
        return x, feature_attenion, sample_attention


class GraphPFNGraphAttentionModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        zero_init: bool = True,
        **kwargs,
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
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        assert edge_weights is None, "So far, we can not apply edge weights."
        x_shape = x.shape

        qkv: Tensor = self.attn_qkv_linear(x)
        qkv = qkv.reshape(*x_shape[:-1], self.n_heads, self.d_head * 3)
        q, k, v = qkv.split(split_size=(self.d_head, self.d_head, self.d_head), dim=-1)

        attn_scores = dgl.ops.u_dot_v(graph, k, q) * self.attn_scores_coef
        attn_probs = dgl.ops.edge_softmax(graph, attn_scores)

        x = dgl.ops.u_mul_e_sum(graph, v, attn_probs)
        x = x.reshape(*x_shape[:-1], self.d)

        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class GraphPFNMLPModule(nn.Module):
    def __init__(self, d: int, zero_init: bool = True, **kwargs):
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
        self, graph: dgl.DGLGraph, x: Tensor, edge_weights: None | Tensor = None
    ) -> Tensor:
        x = self.layers(x)
        return x
