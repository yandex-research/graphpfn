import math
from logging import config
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from lib.tabpfn.model.multi_head_attention import MultiHeadAttention
from lib.tabpfn.model.memory import support_save_peak_mem_factor
from rtdl_num_embeddings import LinearEmbeddings
from typing import Literal, NotRequired, TypedDict

try:
    import loralib as lora
except:
    print("Try: uv pip install loralib")
    exit(1)


class LoraConfig(TypedDict):
    linears: list[str] | Literal["all"]
    embeddings: list[str]  # only LinearEmbeddings is supported (m_num)
    attentions: list[str] | Literal["all"]  # does not work yet
    r: int
    alpha: int
    dropout: float
    lora_out: NotRequired[bool]


class LoraLinearEmbeddings(LinearEmbeddings, lora.LoRALayer):
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        r: int = 0,
        lora_alpha: int = 1,
    ) -> None:
        LinearEmbeddings.__init__(self, n_features, d_embedding)
        lora.LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=True,
        )

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, n_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((d_embedding, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        LinearEmbeddings.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        LinearEmbeddings.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0,
                        1,
                    ) * self.scaling
                self.merged = False
        elif self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A).transpose(
                    0,
                    1,
                ) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = LinearEmbeddings.forward(self, x)
            after_A = torch.mul(self.lora_A.T, x[..., None])
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        return LinearEmbeddings.forward(self, x)


def replace_linear_layers(
    model: nn.Module,
    layers: list[str],
    r: int,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
):
    replaced = []
    for nm, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue

        if not any(nm.startswith(x) for x in layers):
            continue

        print("Replacing:", nm)
        replaced.append(nm)
        lora_linear = lora.Linear(
            m.in_features,
            m.out_features,
            r,
            lora_alpha,
            lora_dropout,
            bias=m.bias is not None,
        ).to(m.weight.device)
        with torch.no_grad():
            lora_linear.weight.data = m.weight.data.clone()
            if lora_linear.bias is not None:
                lora_linear.bias.data = m.bias.data.clone()
        model.set_submodule(nm, lora_linear)
    return replaced


def replace_linear_embeds(
    model: nn.Module, layers: list[str], r: int, lora_alpha: int = 1
) -> list[str]:
    replaced = []
    for layer in layers:
        m = model.get_submodule(layer)
        assert isinstance(m, LinearEmbeddings)
        print("Replacing: ", layer)
        replaced.append(layer)
        lora_linear_embeds = LoraLinearEmbeddings(
            m.weight.shape[0],
            m.weight.shape[1],
            r,
            lora_alpha,
        ).to(m.weight.device)

        with torch.no_grad():
            lora_linear_embeds.weight.data = m.weight.data.clone()
            if lora_linear_embeds.bias is not None:
                lora_linear_embeds.bias.data = m.bias.data.clone()
        model.set_submodule(layer, lora_linear_embeds)
    return replaced


class LoraMHA(MultiHeadAttention):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        d_k: int,
        d_v: int,
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        share_kv_across_n_heads: int = 1,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        initialize_output_to_zero: bool = False,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
        recompute: bool = False,
        init_gain: float = 1,
        two_sets_of_queries: bool = False,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_out: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            share_kv_across_n_heads=share_kv_across_n_heads,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            initialize_output_to_zero=initialize_output_to_zero,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            recompute=recompute,
            init_gain=init_gain,
            two_sets_of_queries=two_sets_of_queries,
        )
        self.lora_A_att = nn.Parameter(
            torch.zeros(3, self._nhead, lora_r, self._input_size),
        )
        self.lora_B_att = nn.Parameter(torch.zeros(3, self._nhead, self._d_k, lora_r))
        nn.init.kaiming_uniform_(self.lora_A_att, a=math.sqrt(5))

        self.lora_scaling = lora_alpha / lora_r
        self.lora_dropout = lora_dropout
        self.lora_out = lora_out
        if lora_out:
            self.lora_A_out = nn.Parameter(torch.zeros(self._nhead, lora_r, self._d_k))
            self.lora_B_out = nn.Parameter(
                torch.zeros(
                    self._nhead,
                    self._output_size,
                    lora_r,
                )
            )
            nn.init.kaiming_uniform_(self.lora_A_att, a=math.sqrt(5))

    def compute_qkv(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        use_second_set_of_queries: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        assert not cache_kv
        assert not use_cached_kv
        assert not use_cached_kv
        assert not use_second_set_of_queries

        if x_kv is None:
            x_kv = x

        k = v = kv = None

        assert (k is None) == (v is None)
        assert self._w_qkv is not None

        w_q, w_kv = self._w_qkv[0], self._w_qkv[1:]

        if (
            self._w_qkv is not None
            and x is x_kv
            and kv is None
            and k is None
            and v is None
        ):
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self._w_qkv)
            qkv += self.lora_scaling * torch.einsum(
                "... s, j h r s, j h d r -> ... j h d",
                F.dropout(x, p=self.lora_dropout),
                self.lora_A_att,
                self.lora_B_att,
            )
            q = None
        else:
            qkv = None
            q = torch.einsum("... s, h d s -> ... h d", x, w_q)
            q += self.lora_scaling * torch.einsum(
                "... s, h r s, h d r -> ... h d",
                F.dropout(x, p=self.lora_dropout),
                self.lora_A_att[0],
                self.lora_B_att[0],
            )

        if kv is None and k is None and v is None and qkv is None and w_kv is not None:
            if reuse_first_head_kv:
                orig_num_heads = w_kv.shape[1]
                w_kv = w_kv[:, :1]
                lora_A_kv = self.lora_A_att[1:, :1]
                lora_B_kv = self.lora_B_att[1:, :1]
            else:
                lora_A_kv = self.lora_A_att[1:]
                lora_B_kv = self.lora_B_att[1:]

            kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, w_kv)
            lora_out = self.lora_scaling * torch.einsum(
                "... s, j h r s, j h d r -> ... j h d",
                F.dropout(x_kv, p=self.lora_dropout),
                lora_A_kv,
                lora_B_kv,
            )
            kv += lora_out
            if reuse_first_head_kv:
                expand_shape = [-1 for _ in kv.shape]
                expand_shape[-2] = orig_num_heads
                kv = kv.expand(*expand_shape)
        return q, k, v, kv, qkv

    @support_save_peak_mem_factor  # type: ignore
    def _compute(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        use_second_set_of_queries: bool,
    ) -> torch.Tensor:
        """Attention computation.
        Called by 'forward', potentially on shards, once shapes have been normalized.
        """
        q, k, v, kv, qkv = self.compute_qkv(
            x,
            x_kv,
            k_cache,
            v_cache,
            kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
            use_second_set_of_queries=use_second_set_of_queries,
        )
        attention_head_outputs = MultiHeadAttention.compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout_p,
            self.softmax_scale,
        )
        out = torch.einsum(
            "... h d, h d s -> ... s",
            attention_head_outputs,
            self._w_out,
        )
        if self.lora_out:
            out += self.lora_scaling * torch.einsum(
                "... h d, h r d, h s r -> ... s",
                F.dropout(attention_head_outputs, p=self.lora_dropout),
                self.lora_A_out,
                self.lora_B_out,
            )
        return out


def replace_mha(
    model: nn.Module,
    layers: list[str],
    r: int,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    lora_out: bool = False,
) -> list[str]:
    replaced = []
    for nm, m in model.named_modules():
        if not isinstance(m, MultiHeadAttention):
            continue
        if not any(nm.startswith(x) for x in layers):
            continue
        assert isinstance(m, MultiHeadAttention)
        print("Replacing: ", nm)
        replaced.append(nm)
        lora_mha = LoraMHA(
            input_size=m._input_size,
            output_size=m._output_size,
            d_k=m._d_k,
            d_v=m._d_v,
            nhead=m._nhead,
            device=m._device,
            dtype=m._dtype,
            dropout_p=m.dropout_p,
            softmax_scale=m.softmax_scale,
            lora_r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_out=lora_out,
        )

        assert m._w_qkv is not None
        lora_mha = lora_mha.to(m._w_qkv.data.device)
        assert lora_mha._w_qkv is not None

        with torch.no_grad():
            lora_mha._w_qkv.data = m._w_qkv.data.clone()
            lora_mha._w_out.data = m._w_out.data.clone()
        model.set_submodule(nm, lora_mha)
    return replaced


def replace_with_loras(model: nn.Module, lora_config: LoraConfig):
    linears_to_replace = []
    if lora_config.get("linears", []) == "all":
        linears_to_replace = [""]
    else:
        linears_to_replace = lora_config.get("linears", [])

    assert isinstance(linears_to_replace, list)

    replace_linear_layers(
        model,
        linears_to_replace,
        lora_config["r"],
        lora_config["alpha"],
        lora_config["dropout"],
    )

    replace_linear_embeds(
        model,
        lora_config.get("embeddings", []),
        lora_config["r"],
        lora_config["alpha"],
    )

    mhas_to_replace = []
    if lora_config.get("attentions", []) == "all":
        mhas_to_replace = [""]
    else:
        mhas_to_replace = lora_config.get("attentions", [])

    assert isinstance(mhas_to_replace, list)

    replace_mha(
        model,
        mhas_to_replace,
        lora_config["r"],
        lora_config["alpha"],
        lora_config["dropout"],
        lora_config.get("lora_out", False),
    )
