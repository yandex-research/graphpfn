from huggingface_hub import hf_hub_download
from torch import nn

import lib
import lib.limix.utils.loading


def load_tfm(
    tfm_name: str,
    tfm_config: dict,
) -> nn.Module:
    if tfm_name == "LimiX":
        model_path = hf_hub_download(
            repo_id="stableai-org/LimiX-16M",
            filename="LimiX-16M.ckpt",
            local_dir="./checkpoints",
            cache_dir="./checkpoints",
        )
        return lib.limix.utils.loading.load_model(model_path=model_path)
    else:
        raise ValueError(f"{tfm_name} is not found")
