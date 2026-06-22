import torch

from lib.tfm.base import TFMBase
from lib.tfm.limix import LimiXWrapper
from lib.tfm.tabicl import TabICLWrapper
from lib.tfm.tabicl_sklearn import TabICLSklearnWrapper
from lib.tfm.tabpfn import TabPFNWrapper
from lib.tfm.tabpfnv2 import TabPFNv2Wrapper
from lib.util import KWArgs


def load_tfm(
    name: str,
    config: KWArgs | None = None,
    *,
    device: torch.device,
    checkpointing: bool = False,
) -> TFMBase:
    if config is None:
        config = {}
    config = config | {"checkpointing": checkpointing}
    match name:
        case "limix":
            return LimiXWrapper(**config)
        case "tabicl":
            return TabICLWrapper(device=device, **config)
        case "tabicl-sklearn":
            return TabICLSklearnWrapper(device=device, **config)
        case "tabpfn":
            return TabPFNWrapper(device=device, **config)
        case "tabpfnv2":
            return TabPFNv2Wrapper(**config)
        case _:
            raise ValueError(f"Unknown tfm: {name}")
