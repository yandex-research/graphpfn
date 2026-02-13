"""
TODO: code here is pretty dirty and needs to be refactored
"""

from functools import partial
from typing import Literal, NotRequired

import dgl
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.profiler import record_function
from typing_extensions import TypedDict

import lib.graph.deep
import lib.tfm
from lib.limix.model.layer import MultiheadAttention as MHA
from lib.util import TaskType


# TODO: refactor naming
class GraphPFNOutput(TypedDict):
    predictions: Tensor
    features_pred: Tensor
    edge_predictions: NotRequired[Tensor]


class GraphPFN(nn.Module):
    def __init__(
        self,
        edge_head: dict | None,
        feat_head: bool = True,
        layer_ids: list[int] = list(range(12)),
        freeze_tfm: bool = True,
        random_init_tfm: bool = False,
        n_random_features: int | None = None,
    ) -> None:
        super().__init__()
        self.n_random_features = n_random_features
        self.tfm = lib.tfm.load_tfm(
            tfm_name="LimiX", tfm_config={"load_weights": not random_init_tfm}
        )
        self.tfm.mask_prediction = True  # type: ignore

        for idx in layer_ids:
            layer = self.tfm.transformer_encoder.layers[idx]
            wrapped_layer = GraphPFNLayerWrapper(base=layer)
            self.tfm.transformer_encoder.layers[idx] = wrapped_layer

        # >>> By default, we freeze all params of TFM backbone
        for param in self.tfm.parameters():
            param.requires_grad = not freeze_tfm

        for idx in layer_ids:
            wrapped_layer = self.tfm.transformer_encoder.layers[idx]
            layer_params = [
                *wrapped_layer.mlp.parameters(),
                *wrapped_layer.conv.parameters(),
            ]
            for param in layer_params:
                param.requires_grad = True

        # Unfreeze feature decoder
        if feat_head:
            for param in self.tfm.feature_decoder.parameters():
                param.requires_grad = True

        # >>> We also have a separate head for edge reconstruction
        if edge_head is not None:
            self.edge_head = EdgeHead(
                d_embedding=self.tfm.embed_dim,
                d_hidden=self.tfm.hid_dim,
                **edge_head,
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
        batched_attn: bool = False,
    ) -> GraphPFNOutput:
        assert features.ndim == 2
        assert y_train.ndim == 1
        assert train_mask.ndim == 1
        assert y_train.shape[0] == train_mask.int().sum().item()

        if self.n_random_features is not None:
            random_features = torch.randn(
                [features.shape[0], self.n_random_features],
                device=features.device,
            )
            features = torch.cat([features, random_features], dim=-1)

        # >>> Permutation and its inverse
        # This is tricky.
        # TFMs assume that first K samples are train, and remaining are not.
        # While in general, we have train samples in arbitrary positions.
        # So we need to permute input features and graph before passing to TFM.
        # And permute back TFM outputs.
        perm = torch.argsort((~train_mask).float(), stable=True)
        inv_perm = torch.argsort(perm, stable=True)

        # new_features[j] = old_features[perm[j]]
        features = features[perm, ...]

        # old_edges = [(u, v), ...] => new_edges = [(inv_perm[u], inv_perm[v]), ...]
        n_nodes = graph.num_nodes()
        src, dst = graph.edges()
        src = inv_perm[src]
        dst = inv_perm[dst]
        graph = dgl.graph((src, dst), num_nodes=n_nodes)

        # >>> Apply Backbone
        for module in self.modules():  # TODO: refactor this
            if isinstance(module, GraphPFNLayerWrapper):
                module.graph = graph
            if isinstance(module, MHA):
                module.batched = batched_attn

        sdpa_backends = [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        ]

        with torch.nn.attention.sdpa_kernel(sdpa_backends):
            out = self.tfm.forward(
                x=features.unsqueeze(0),
                y=y_train.unsqueeze(0),
                eval_pos=y_train.shape[0],
                task_type="reg" if (task_type == TaskType.REGRESSION) else "cls",
                checkpointing=checkpointing,
            )

            pred = (
                (
                    out["reg_output"]
                    if task_type == TaskType.REGRESSION
                    else out["cls_output"]
                )
                .float()
                .squeeze(0)
            )

        dummy_pred = pred.new_zeros([n_nodes - pred.shape[0], *pred.shape[1:]])
        pred = torch.cat([dummy_pred, pred], dim=0)
        pred = pred[inv_perm, ...]

        if task_type == TaskType.REGRESSION:
            pred = pred.squeeze(-1)

        # Extract feature pred
        extract_feat = (  # noqa E731
            lambda x: x.reshape(*x.shape[:-2], -1)[..., : features.shape[-1]]
        )
        features_pred = extract_feat(out["feature_pred"].squeeze(0)[inv_perm, ...])
        feature_mean = extract_feat(out["process_config"]["mean_for_normalization"])
        feature_std = extract_feat(out["process_config"]["std_for_normalization"])
        features_pred = features_pred * feature_std + feature_mean
        # TODO: double-check the features_pred, I guess it works incorrect

        # Extract encoder embeddings
        encoder_embed = out["encoder_embed"].squeeze(0)[inv_perm, ...]
        if edges is not None:
            src, dst = edges
            edge_predictions = self.edge_head(encoder_embed, src, dst)
        else:
            edge_predictions = None

        # Check that no features were filtered
        # TODO: maybe throw error here?
        num_used_features = out["process_config"]["num_used_features"].sum().item()
        if num_used_features != features.shape[-1]:
            logger.error(f"{num_used_features=}, while {features.shape[-1]=}")

        return {
            "predictions": pred,
            "features_pred": features_pred,
            "edge_predictions": edge_predictions,  # type: ignore
        }


class EdgeHead(nn.Module):
    def __init__(
        self,
        d_embedding: int,
        d_hidden: int,
        mode: str = "mul",
    ):
        super().__init__()

        self.mode = mode
        if mode == "mul":
            d_input = d_embedding
        elif mode == "cat":
            d_input = 2 * d_embedding
        else:
            raise ValueError(f"Unknown {mode=}")

        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        embedding: Tensor,
        src: Tensor,
        dst: Tensor,
    ) -> Tensor:
        assert embedding.ndim == 2
        src_embedding = embedding[src, :]
        dst_embedding = embedding[dst, :]
        if self.mode == "mul":
            edge_embedding = src_embedding * dst_embedding
        elif self.mode == "cat":
            edge_embedding = torch.cat([src_embedding, dst_embedding], dim=-1)
        else:
            raise NotImplementedError(f"{self.mode=}")
        return self.mlp(edge_embedding)


class GraphPFNLayerWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        zero_init: bool = True,
    ):
        super().__init__()
        self.graph: dgl.DGLGraph | None = None  # placeholder

        self.base = base
        self.conv = lib.graph.deep.ResidualModule(
            base_class=partial(
                GraphPFNGraphAttentionModule,
                zero_init=zero_init,
            ),  # type: ignore
            norm_class=nn.LayerNorm,
            d_hidden=192,
        )
        self.mlp = lib.graph.deep.ResidualModule(
            base_class=partial(GraphPFNMLPModule, zero_init=zero_init),  # type: ignore
            norm_class=nn.LayerNorm,
            d_hidden=192,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_atten_mask: torch.Tensor,  # TODO: maybe refactor this?
        eval_pos: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # >>> Apply base layer
        with record_function("TFM"):
            x, feature_attenion, sample_attention = self.base(
                x=x,
                feature_atten_mask=feature_atten_mask,
                eval_pos=eval_pos,
                layer_idx=layer_idx,
            )

        # >>> Apply GNN layer
        with record_function("GraphConv"):
            x = self.conv(self.graph, x)
        with record_function("MLP"):
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
        edge_weights: None | Tensor = None,
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
        edge_weights: None | Tensor = None,
    ) -> Tensor:
        x = self.layers(x)
        return x
