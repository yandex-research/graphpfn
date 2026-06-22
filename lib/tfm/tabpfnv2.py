from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from lib.tfm import TFMBase
from lib.util import TaskType
from vendor.tabpfnv2.model.bar_distribution import FullSupportBarDistribution
from vendor.tabpfnv2.model.layer import PerFeatureEncoderLayer

# >>> Architecture constants
_EMSIZE = 192
_NHEAD = 6
_NLAYERS = 12
_NHID_FACTOR = 4
_NHID = _EMSIZE * _NHID_FACTOR

_CLS_CHECKPOINT = "tabpfn-v2-classifier.ckpt"
_REG_CHECKPOINT = "tabpfn-v2-regressor.ckpt"


class _LayerStack(nn.Module):
    def __init__(
        self,
        *,
        layer_creator: Any,
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(self, x: Tensor, *, half_layers: bool = False, **kwargs: Any) -> Tensor:
        if half_layers:
            assert self.min_num_layers_layer_dropout == self.num_layers
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:  # pyright: ignore[reportArgumentType]
            if self.recompute_each_layer and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    partial(layer, **kwargs), x, use_reentrant=False
                )  # type: ignore[assignment]
            else:
                x = layer(x, **kwargs)

        return x


def _extract_state_dict(state_dict: dict, prefix: str) -> dict:
    return {
        k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)
    }


class _TabPFNModel(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_name: str,
        is_regression: bool,
        checkpointing: bool,
    ) -> None:
        super().__init__()
        state_dict = torch.load(
            Path("checkpoints") / checkpoint_name,
            weights_only=True,
        )["state_dict"]

        output_dim = 5_000 if is_regression else 10
        self.is_regression = is_regression

        # >>> Feature embeddings
        self.feature_embedding = nn.Linear(1, _EMSIZE, bias=False)
        self.feature_embedding.weight.data = state_dict["encoder.5.layer.weight"][
            :, 0
        ].unsqueeze(1)

        # >>> Positional embeddings
        self.pos_embs = nn.Linear(48, _EMSIZE)
        self.pos_embs.load_state_dict(
            _extract_state_dict(state_dict, "feature_positional_embedding_embeddings.")
        )

        # >>> Target embeddings
        layer_key = "1" if is_regression else "2"
        self.y_embedding_weight = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 0]
        )
        self.y_embedding_nan_ind = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 1]
        )
        self.y_embedding_bias = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.bias"]
        )

        # >>> Transformer
        def layer_creator():
            return PerFeatureEncoderLayer(
                d_model=_EMSIZE,
                nhead=_NHEAD,
                dim_feedforward=_NHID,
                activation="gelu",
                zero_init=False,
                precomputed_kv=None,
                multiquery_item_attention_for_test_set=True,
                layer_norm_with_elementwise_affine=False,
            )

        self.transformer_encoder = _LayerStack(
            layer_creator=layer_creator,
            num_layers=_NLAYERS,
            recompute_each_layer=checkpointing,
            min_num_layers_layer_dropout=None,
        )
        self.transformer_encoder.load_state_dict(
            _extract_state_dict(state_dict, "transformer_encoder.")
        )

        # >>> Decoder
        self.decoder = nn.Sequential(
            nn.Linear(_EMSIZE, _NHID),
            nn.GELU(),
            nn.Linear(_NHID, output_dim),
        )
        self.decoder.load_state_dict(
            _extract_state_dict(state_dict, "decoder_dict.standard.")
        )

    def forward(self, x_train: Tensor, y_train: Tensor, x_eval: Tensor) -> Tensor:
        n_train = x_train.shape[0]
        n_eval = x_eval.shape[0]
        n_features = x_train.shape[1]

        # >>> Feature embeddings
        x = torch.cat([x_train, x_eval], dim=0)  # (n_total, n_features)
        x_inp = self.feature_embedding(x.unsqueeze(-1))  # (n_total, n_features, emsize)
        x_inp = x_inp.unsqueeze(0)  # (1, n_total, n_features, emsize)

        x_inp = (
            x_inp
            + self.pos_embs(torch.randn(n_features, _EMSIZE // 4, device=x_inp.device))[
                None, None
            ]
        )

        # >>> Target embeddings
        y_mult = y_train.mean()
        if not self.is_regression:
            y_mult = torch.round(y_mult)
        y_test_placeholder = y_mult * x_inp.new_ones(n_eval)
        y_all = torch.cat([y_train, y_test_placeholder], dim=0).float()  # (n_total,)

        nan_ind = x_inp.new_zeros(n_train + n_eval)
        nan_ind[n_train:] = -2.0

        y_emb = (
            y_all.view(-1, 1, 1) * self.y_embedding_weight.view(1, 1, -1)
            + nan_ind.view(-1, 1, 1) * self.y_embedding_nan_ind.view(1, 1, -1)
            + self.y_embedding_bias.view(1, 1, -1)
        )  # (n_total, 1, emsize)
        y_emb = y_emb.unsqueeze(0)  # (1, n_total, 1, emsize)

        # >>> Transformer forward
        x_inp = torch.cat([x_inp, y_emb], dim=2)  # (1, n_total, n_features+1, emsize)
        encoder_out = self.transformer_encoder(
            x_inp,
            half_layers=False,
            cache_trainset_representation=False,
            single_eval_pos=n_train,
        )

        preds = self.decoder(encoder_out[:, n_train:, -1])  # (1, n_eval, output_dim)
        return preds.squeeze(0)  # (n_eval, output_dim)


class TabPFNv2Wrapper(TFMBase):
    def __init__(self, checkpointing: bool = False) -> None:
        super().__init__()
        self.classifier = _TabPFNModel(
            checkpoint_name=_CLS_CHECKPOINT,
            is_regression=False,
            checkpointing=checkpointing,
        )
        self.regressor = _TabPFNModel(
            checkpoint_name=_REG_CHECKPOINT,
            is_regression=True,
            checkpointing=checkpointing,
        )

        reg_state_dict = torch.load(
            Path("checkpoints") / _REG_CHECKPOINT,
            weights_only=True,
        )["state_dict"]
        borders = reg_state_dict["criterion.borders"]
        self.bar_distribution = FullSupportBarDistribution(borders)

    def forward(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_eval: Tensor,
        task_type: TaskType,
    ) -> Tensor:
        x = torch.cat([x_train, x_eval], dim=0)
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        x_train, x_eval = x[: x_train.shape[0]], x[x_train.shape[0] :]

        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            if task_type == TaskType.REGRESSION:
                logits = self.regressor(x_train, y_train, x_eval).float()
                with torch.autocast(device_type=x_train.device.type, enabled=False):
                    return self.bar_distribution.mean(logits)
            else:
                preds = self.classifier(x_train, y_train, x_eval).float()
                return F.log_softmax(preds, dim=-1)
