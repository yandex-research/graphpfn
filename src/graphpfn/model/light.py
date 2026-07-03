from contextlib import nullcontext

import dgl
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from graphpfn._vendor.limix.model.layer import MLP, EncoderBaseLayer, MultiheadAttention
from graphpfn.model.layers import (
    GraphHolder,
    GraphPFNLayerWrapper,
    GraphPFNResidualModule,
)
from graphpfn.model.limix import LimiXWrapper
from graphpfn.model.util import apply_dynamic_checkpointing
from graphpfn.util import TaskType


class GraphPFNLight(nn.Module):
    def __init__(
        self,
        n_random_features: int = 8,
        autograd_cpu_offloading: bool = False,
    ) -> None:
        super().__init__()
        self.n_random_features = n_random_features
        self.autograd_cpu_offloading = autograd_cpu_offloading
        self._graph_holder = GraphHolder()

        self.tfm = LimiXWrapper(load_weights=True)
        for idx in range(12):
            layer = self.tfm.module.transformer_encoder.layers[idx]
            wrapped_layer = GraphPFNLayerWrapper(
                base=layer, graph_holder=self._graph_holder
            )
            self.tfm.module.transformer_encoder.layers[idx] = wrapped_layer

        # >>> Apply dynamic checkpointing
        apply_dynamic_checkpointing(
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
    ) -> Tensor:
        assert features.ndim == 2
        assert y_train.ndim == 1
        assert train_mask.ndim == 1
        assert y_train.shape[0] == train_mask.int().sum().item()

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

        # >>> Apply Backbone
        self._graph_holder.graph = graph

        with (
            torch.autograd.graph.save_on_cpu()
            if self.autograd_cpu_offloading
            else nullcontext()
        ):
            preds, _ = self.tfm(
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

        return pred
