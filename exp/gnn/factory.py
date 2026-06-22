import itertools
from pathlib import Path

import lib
import lib.graph.constants as constants

DATASET_PRESETS = {
    "classic": [
        "pubmed",
        "facebook",
        "amazon-ratings",
        "questions",
        "wiki-cs",
    ],
    "graphland": [
        "tolokers-2",
        "city-reviews",
        "artnet-exp",
        "hm-prices",
        "avazu-ctr",
        "city-roads-M",
        "twitch-views",
        "artnet-views",
    ],
}

CONV_NAMES = ["gcn", "sage", "gat", "gt"]

SEPARABLE_CONV_NAMES = {
    "gcn",
    "gat",
    "gt",
}


ADD_SELF_LOOP_NAMES = {
    "gat",
    "gt",
}


GRAPHLAND_LR_CONFIG = [
    "_tune_",
    "categorical",
    [
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
    ],
]

GRAPHLAND_DROPOUT_CONFIG = [
    "_tune_",
    "categorical",
    [
        0.0,
        0.1,
        0.2,
    ],
]

TABPFN_LR_CONFIG = [
    "_tune_",
    "categorical",
    [
        4.999998054699972e-06,
        8.340500244230498e-06,
        1.3912793292547576e-05,
        2.320794010302052e-05,
        3.871318040182814e-05,
        6.457747804233804e-05,
        0.00010772173118311912,
        0.00017969068721868098,
        0.00029974215431138873,
        0.0005000000819563866,
    ],
]


def prepare_configs(
    experiment: str | Path,
    *,
    conv_name: str,
    function: str = "bin.base.gnn.main",
    simple: bool = False,
    sep: bool = False,
    gnn_num_layers: int = 3,
    gnn_hidden_dim: int = 512,
    separable_conv_names=SEPARABLE_CONV_NAMES,
    dataset_names=[
        *DATASET_PRESETS["classic"],
        *DATASET_PRESETS["graphland"],
    ],
) -> None:
    experiments_root = Path(experiment)

    for dataset_name in dataset_names:
        experiments_path = experiments_root / dataset_name

        # >>> Data
        _data_config = {
            "cache": False,
            "path": f"data/{dataset_name}",
            "setting": "transductive",
            "add_self_loops": conv_name in ADD_SELF_LOOP_NAMES,
        }
        _transform_config: dict = {
            "labels": False,
        }

        if dataset_name in constants.REGRESSION_DATASETS:
            _transform_config["labels"] = True

        if dataset_name in constants.HETEROGENEOUS_DATASETS:
            _data_config |= {
                "internal_split_name": "RL",
            }
            _transform_config["features"] = {
                "seed": 0,
                "cat_policy": "one-hot",
                "num_policy": "quantile-normal",
                "frac_policy": ["_tune_", "categorical", ["quantile-normal", "none"]],
            }
        else:
            if dataset_name in constants.PYG_DATASETS:
                _data_config["external_split_file"] = "split.npz"
            _transform_config["features"] = {
                "seed": 0,
                "num_policy": ["_tune_", "categorical", ["standard", "none"]],
            }

        # >>> Optimizer
        _optimizer_config = {
            "type": "AdamW",
            "lr": ["_tune_", "categorical", [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]],
            "weight_decay": 0.0,
        }

        # >>> Backbone
        _backbone_config = {
            "type": "BaseGraphBackbone",
            "conv_name": conv_name
            + ("-sep" if (sep and (conv_name in separable_conv_names)) else ""),
            "norm_name": "none" if simple else "layer",
            "residual": not simple,
            "n_layers": gnn_num_layers,
            "d_hidden": gnn_hidden_dim,
            "dropout": ["_tune_", "categorical", [0.0, 0.1, 0.2]],
            "activation": "gelu",
        }

        # >>> GNN
        _gnn_config = {
            "backbone": _backbone_config,
        }

        # >>> Space (base.py config structure)
        _space_config = {
            "seed": 0,
            "data": _data_config,
            "transform": _transform_config,
            "model": _gnn_config,
            "optimizer": _optimizer_config,
            "n_steps": 1000,
            "patience": 100,
        }

        config = {
            "seed": 0,
            "function": function,
            "n_trials": 100,
            "sampler_type": "BruteForceSampler",
            "space": _space_config,
        }

        experiments_path.mkdir(parents=True, exist_ok=True)
        config_path = experiments_path / "tuning"
        lib.dump_config(config_path, config, force=True)


if __name__ == "__main__":
    experiment = Path(__file__).parent.resolve()

    for (conv,) in itertools.product(
        ["gcn", "sage", "gat", "gt"],
    ):
        name = f"{conv}"

        prepare_configs(
            experiment=experiment / name,
            conv_name=conv,
        )
