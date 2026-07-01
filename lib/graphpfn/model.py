from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired

import dgl
import torch
import torch.nn as nn
import torch.utils.checkpoint
from loguru import logger
from torch import Tensor
from torch.profiler import record_function
from typing_extensions import TypedDict

import lib.deep
from lib.graph.pearl import PEARL
from lib.tfm.limix import LimiXWrapper
from lib.util import KWArgs, TaskType
from vendor.limix.model.layer import MLP, EncoderBaseLayer, MultiheadAttention

MAX_SDPA_GRAPH_SIZE = 10_000


@dataclass
class SDPAInput:
    attn_mask: Tensor
    zero_degree_mask: Tensor

    def to(self, device: torch.device) -> "SDPAInput":
        return SDPAInput(
            attn_mask=self.attn_mask.to(device),
            zero_degree_mask=self.zero_degree_mask.to(device),
        )


GraphAdapterInput = dgl.DGLGraph | SDPAInput


class GraphHolder:
    __slots__ = ("graph",)
    graph: GraphAdapterInput | None

    def __init__(self) -> None:
        self.graph = None


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
        n_random_features: int = 0,
        pearl: KWArgs | None = None,
        autograd_cpu_offloading: bool = False,
    ) -> None:
        super().__init__()
        self.n_random_features = n_random_features
        self.autograd_cpu_offloading = autograd_cpu_offloading

        self.tfm = LimiXWrapper(load_weights=not random_init_tfm)

        if pearl is None:
            self.pearl = None
        else:
            self.pearl = PEARL(**pearl, checkpointing=True)

        self._graph_holder = GraphHolder()

        for idx in layer_ids:
            layer = self.tfm.module.transformer_encoder.layers[idx]
            wrapped_layer = GraphPFNLayerWrapper(
                base=layer, graph_holder=self._graph_holder
            )
            self.tfm.module.transformer_encoder.layers[idx] = wrapped_layer

        # >>> By default, we freeze all params of TFM backbone
        for param in self.tfm.parameters():
            param.requires_grad = not freeze_tfm

        for idx in layer_ids:
            wrapped_layer = self.tfm.module.transformer_encoder.layers[idx]
            layer_params = [
                *wrapped_layer.mlp.parameters(),
                *wrapped_layer.conv.parameters(),
            ]
            for param in layer_params:
                param.requires_grad = True

        # Unfreeze feature decoder
        if feat_head:
            for param in self.tfm.module.feature_decoder.parameters():
                param.requires_grad = True

        # >>> We also have a separate head for edge reconstruction
        if edge_head is not None:
            self.edge_head = EdgeHead(
                d_embedding=self.tfm.module.embed_dim,
                d_hidden=self.tfm.module.hid_dim,
                **edge_head,
            )

        # >>> Apply dynamic checkpointing
        lib.deep.apply_dynamic_checkpointing(
            self.tfm,
            should_checkpoint_fn=lambda x_train, y_train, x_eval, *_, **__: (
                x_train.numel() + x_eval.numel() > 4_000 * 100
            ),
            submodule_filter_fn=lambda name, submodule: (
                isinstance(
                    submodule,
                    GraphPFNResidualModule
                    | EncoderBaseLayer
                    | MultiheadAttention
                    | MLP,
                )
                or name.endswith("encoder_x")
                or name.endswith("x_preprocess")
                or name.endswith("cls_y_decoder")
                or name.endswith("reg_y_decoder")
                or name.endswith("feature_decoder")
            ),
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: Tensor,
        y_train: Tensor,
        train_mask: Tensor,
        task_type: TaskType,
        *,
        n_random_features: int | None = None,
        edges: tuple[Tensor, Tensor] | None = None,
    ) -> GraphPFNOutput:
        assert features.ndim == 2
        assert y_train.ndim == 1
        assert train_mask.ndim == 1
        assert y_train.shape[0] == train_mask.int().sum().item()

        if self.pearl is not None:
            features = torch.cat([features, self.pearl(graph)], dim=-1)

        if n_random_features is None:
            n_random_features = self.n_random_features

        if n_random_features > 0:
            random_features = torch.randn(
                [features.shape[0], n_random_features],
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

        # >>> Pre-compute graph adapter input
        if n_nodes < MAX_SDPA_GRAPH_SIZE:
            attn_mask = torch.zeros(
                n_nodes, n_nodes, dtype=torch.bool, device=features.device
            )
            attn_mask[dst, src] = True
            zero_degree_mask = graph.in_degrees() == 0
            attn_mask[zero_degree_mask, zero_degree_mask] = True
            graph = SDPAInput(attn_mask=attn_mask, zero_degree_mask=zero_degree_mask)  # type: ignore

        # >>> Apply Backbone
        self._graph_holder.graph = graph
        out: dict

        with (
            torch.autograd.graph.save_on_cpu()
            if self.autograd_cpu_offloading
            else nullcontext()
        ):
            preds, out = self.tfm(
                x_train=features[: y_train.shape[0]],
                x_eval=features[y_train.shape[0] :],
                y_train=y_train,
                task_type=task_type,
                return_preds_only=False,
            )

        self._graph_holder.graph = None

        dummy_pred = preds.new_zeros([n_nodes - preds.shape[0], *preds.shape[1:]])
        pred = torch.cat([dummy_pred, preds], dim=0)
        pred = pred[inv_perm, ...]

        # Extract feature pred
        extract_feat = (  # noqa E731
            lambda x: x.reshape(*x.shape[:-2], -1)[..., : features.shape[-1]]
        )
        features_pred = extract_feat(out["feature_pred"].squeeze(0)[inv_perm, ...])
        if out["process_config"]["mean_for_normalization"] is not None:
            feature_mean = extract_feat(out["process_config"]["mean_for_normalization"])
            feature_std = extract_feat(out["process_config"]["std_for_normalization"])
            features_pred = features_pred * feature_std + feature_mean

        # Extract encoder embeddings
        encoder_out = out["encoder_out"]
        assert encoder_out is not None
        encoder_embed = encoder_out[:, :, -1, :].squeeze(0)[inv_perm, ...]
        if edges is not None:
            src, dst = edges
            edge_predictions = self.edge_head(encoder_embed, src, dst)
        else:
            edge_predictions = None

        # Check that no features were filtered
        if out["process_config"]["num_used_features"] is not None:
            num_used_features = out["process_config"]["num_used_features"].sum().item()
            if num_used_features != features.shape[-1]:
                logger.error(f"{num_used_features=}, while {features.shape[-1]=}")

        return {
            "predictions": pred,
            "features_pred": features_pred,
            "edge_predictions": edge_predictions,  # type: ignore
        }  # type: ignore


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


class GraphPFNResidualModule(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_hidden: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.base = base

    def forward(self, graph: GraphAdapterInput, x: Tensor) -> Tensor:
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
        with record_function("TFM"):
            x, feature_attenion, sample_attention = self.base(
                x=x,
                feature_atten_mask=feature_atten_mask,
                eval_pos=eval_pos,
                layer_idx=layer_idx,
            )

        # >>> Apply GNN layer
        with record_function("GraphConv"):
            x = self.conv(graph, x)
        with record_function("MLP"):
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
        graph: GraphAdapterInput,
        x: Tensor,
    ) -> Tensor:
        assert x.ndim == 4
        assert x.shape[0] == 1, "Batches are not supported yet"

        x = x.squeeze(0)
        x_shape = x.shape
        qkv: Tensor = self.attn_qkv_linear(x)
        qkv = qkv.reshape(*x_shape[:-1], self.n_heads, self.d_head * 3)
        q, k, v = qkv.split(split_size=(self.d_head, self.d_head, self.d_head), dim=-1)

        if isinstance(graph, dgl.DGLGraph):
            attn_scores = dgl.ops.u_dot_v(graph, k, q) * self.attn_scores_coef  # type: ignore
            attn_probs = dgl.ops.edge_softmax(graph, attn_scores)
            x = dgl.ops.u_mul_e_sum(graph, v, attn_probs)  # type: ignore

        else:
            graph = graph.to(x.device)

            # (n_nodes, n_features, n_heads, d_head)
            # -> (n_features, n_heads, n_nodes, d_head)
            q = q.permute(1, 2, 0, 3)
            k = k.permute(1, 2, 0, 3)
            v = v.permute(1, 2, 0, 3)

            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=graph.attn_mask
            )

            x = torch.where(graph.zero_degree_mask[:, None], 0.0, x)

            x = x.permute(2, 0, 1, 3)

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
        graph: GraphAdapterInput,
        x: Tensor,
    ) -> Tensor:
        x = self.layers(x)
        return x


def resolve_checkpoint(
    name: str,
    *,
    local_dir: str | Path | None = "checkpoints/graphpfn",
) -> Path:
    if not name.startswith("hf://"):
        return Path(name)
    parts = name.removeprefix("hf://").split("/")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, "config.json", local_dir=local_dir)
    local_path = hf_hub_download(repo_id, filename=filename, local_dir=local_dir)
    return Path(local_path)


def load_graphpfn(checkpoint: str | Path, **model_kwargs: KWArgs) -> GraphPFN:
    model = GraphPFN(**model_kwargs)  # type: ignore
    path = resolve_checkpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    assert all([k in model.state_dict() for k in state.keys()])
    model.load_state_dict(state, strict=False)
    return model
