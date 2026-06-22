from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch import Tensor

from lib.deep import apply_dynamic_checkpointing
from lib.tfm import TFMBase
from lib.util import TaskType
from vendor.limix.model.encoders import MaskEmbEncoder as _MaskEmbEncoder
from vendor.limix.model.layer import EncoderBaseLayer, MultiheadAttention
from vendor.limix.model.transformer import FeaturesTransformer
from vendor.limix.utils.loading import build_model

# >>> Patch SDPA
_ATTN_CHUNK = 2**15
_orig_flashattn = MultiheadAttention.compute_attention_by_flashattn
_orig_torch_attn = MultiheadAttention.compute_attention_by_torch


def _patched_compute_attention_by_flashattn(
    self: MultiheadAttention,
    qkv: torch.Tensor | None,
    q: torch.Tensor | None,
    kv: torch.Tensor | None,
) -> torch.Tensor:
    B = qkv.shape[0] if qkv is not None else q.shape[0]  # type: ignore[union-attr]
    return torch.cat(
        [
            _orig_flashattn(
                self,
                qkv[s : s + _ATTN_CHUNK] if qkv is not None else None,
                q[s : s + _ATTN_CHUNK] if q is not None else None,
                kv[s : s + _ATTN_CHUNK] if kv is not None else None,
            )
            for s in range(0, B, _ATTN_CHUNK)
        ]
    )


def _patched_compute_attention_by_torch(
    self: MultiheadAttention,
    qkv: torch.Tensor | None,
    q: torch.Tensor | None,
    kv: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    B = qkv.shape[0] if qkv is not None else q.shape[0]  # type: ignore[union-attr]
    return torch.cat(
        [
            _orig_torch_attn(
                self,
                qkv[s : s + _ATTN_CHUNK] if qkv is not None else None,
                q[s : s + _ATTN_CHUNK] if q is not None else None,
                kv[s : s + _ATTN_CHUNK] if kv is not None else None,
                attn_mask[s : s + _ATTN_CHUNK] if attn_mask is not None else None,
            )
            for s in range(0, B, _ATTN_CHUNK)
        ]
    )


MultiheadAttention.compute_attention_by_flashattn = (
    _patched_compute_attention_by_flashattn
)
MultiheadAttention.compute_attention_by_torch = _patched_compute_attention_by_torch
# <<<


# >>> Patch MaskEmbEncoder
_orig_mask_emb_encoder_forward = _MaskEmbEncoder.forward
_MaskEmbEncoder.forward = lambda self, input: _orig_mask_emb_encoder_forward(
    self, input.copy()
)
# <<<


def _download_limix_checkpoint() -> str:
    return hf_hub_download(
        repo_id="stableai-org/LimiX-16M",
        filename="LimiX-16M.ckpt",
        local_dir="./checkpoints",
        cache_dir="./checkpoints",
    )


def load_model(
    model_path,
    mask_prediction: bool = False,
    load_weights: bool = True,
):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = state_dict["config"]
    config["mask_prediction"] = mask_prediction

    model = build_model(config)
    if load_weights:
        model.load_state_dict(state_dict["state_dict"])

    model.eval()
    return model


class LimiXWrapper(TFMBase):
    def __init__(self, load_weights: bool = True, checkpointing: bool = False) -> None:
        super().__init__()
        model_path = _download_limix_checkpoint()
        self.module: FeaturesTransformer = load_model(
            model_path=model_path,
            load_weights=load_weights,
            mask_prediction=True,
        )

        self._encoder_out: Tensor | None = None
        self.module.encoder_out_norm.register_forward_hook(self._capture_encoder_out)

        if checkpointing:
            apply_dynamic_checkpointing(
                self,
                should_checkpoint_fn=lambda x_train, y_train, x_eval, *_, **__: (
                    x_train.numel() + x_eval.numel() > 4_000 * 100
                ),
                submodule_filter_fn=lambda name, submodule: (
                    isinstance(submodule, EncoderBaseLayer)
                    or name.endswith("encoder_x")
                    or name.endswith("x_preprocess")
                    or name.endswith("cls_y_decoder")
                    or name.endswith("reg_y_decoder")
                    or name.endswith("feature_decoder")
                ),
            )

    def _capture_encoder_out(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        self._encoder_out = output

    def forward(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_eval: Tensor,
        task_type: TaskType,
        *,
        return_preds_only: bool = True,
    ) -> Tensor:
        x = torch.cat([x_train, x_eval], dim=0).unsqueeze(0)
        n_train = x_train.shape[0]

        limix_task_type = "reg" if task_type == TaskType.REGRESSION else "cls"

        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            out = self.module(
                x=x,
                y=y_train.unsqueeze(0),
                eval_pos=n_train,
                task_type=limix_task_type,
            )

        out["encoder_out"] = self._encoder_out
        self._encoder_out = None

        if task_type == TaskType.REGRESSION:
            preds = out["reg_output"].float().squeeze(0).squeeze(-1)
        else:
            preds = torch.log_softmax(out["cls_output"].float().squeeze(0), dim=-1)

        if return_preds_only:
            return preds
        else:
            return preds, out  # type: ignore
