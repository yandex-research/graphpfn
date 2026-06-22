from functools import partial

import dgl
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import lib.graph.deep

PEARL_BACKBONE_DEFAULT_CONFIG = {
    "type": "BaseGraphBackbone",
    "conv_name": "gcn",
    "norm_name": "none",
    "residual": False,
    "n_layers": 3,
    "d_hidden": 512,
    "dropout": 0.1,
    "activation": "gelu",
}


class PEARL(nn.Module):
    def __init__(
        self,
        *,
        d_output: int,
        batch_size: int = 8,
        backbone: dict = PEARL_BACKBONE_DEFAULT_CONFIG,
        n_features: int = 1,
        inp_out_type: str = "mlp",
        checkpointing: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_features = n_features
        self.checkpointing = checkpointing

        # >>> Input
        d_hidden = backbone["d_hidden"]
        dropout = backbone["dropout"]

        self.input = {
            "mlp": (
                nn.Sequential(
                    nn.Linear(n_features, d_hidden),
                    nn.Dropout(dropout),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_hidden),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(n_features, d_hidden),
                )
            ),
        }[inp_out_type]

        # >>> Backbone
        self.backbone = lib.graph.deep.make_module(
            **backbone,
        )

        # >>> Output
        self.output = {
            "mlp": (
                nn.Sequential(
                    lib.graph.deep.NORM_MODULES[backbone["norm_name"]](d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_output),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_hidden, d_output),
                )
            ),
        }[inp_out_type]

    def forward(
        self,
        graph: dgl.DGLGraph,
        checkpointing: bool | None = None,
    ) -> Tensor:
        x_list = []

        if checkpointing is None:
            checkpointing = self.checkpointing

        def apply(x: torch.Tensor) -> torch.Tensor:
            x = self.input(x)
            x = self.backbone(graph, x)
            x = self.output(x)
            return x

        if checkpointing:
            apply = partial(checkpoint, apply, use_reentrant=False)  # type: ignore

        for _ in range(self.batch_size):
            x = torch.randn(graph.num_nodes(), self.n_features, device=graph.device)
            x = apply(x)
            x_list.append(x)

        return torch.stack(x_list, dim=-1).mean(-1)
