# LightGBM

import copy
import os
from pathlib import Path
from typing import Any
from typing_extensions import TypedDict

import delu
import dgl
import lightgbm
import numpy as np
import pandas as pd
import torch
from typing import NotRequired
from torch import Tensor
from torch import nn
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger

import lib
from lib import KWArgs


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    pearl: NotRequired[KWArgs]
    fit: KWArgs


class PEARL(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        batch_size: int,
        backbone: dict,
        n_features: int = 1,
        inp_out_type: str = "mlp",
        checkpoint_name: str = "pearl_random",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_features = n_features

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
                    nn.Linear(d_hidden, output_dim),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_hidden, output_dim),
                )
            ),
        }[inp_out_type]

        checkpoint_path = Path("checkpoints") / f"{checkpoint_name}.pt"
        self.load_state_dict(torch.load(checkpoint_path))

    def forward(
        self,
        graph: dgl.DGLGraph,
    ) -> Tensor:
        x_list = []

        for _ in range(self.batch_size):
            x = torch.randn(graph.num_nodes(), self.n_features, device=graph.device)
            x = self.input(x)
            x = self.backbone(graph, x)
            x = self.output(x)
            x_list.append(x)

        return torch.stack(x_list, dim=-1).mean(-1)


def main(
    config: Config, output: str | Path, *, force: bool = False
) -> None | lib.JSONDict:
    # >>> start
    config = copy.deepcopy(config)
    config, output = lib.check(config, output, config_type=Config)
    if not lib.start(main, output, force=force):
        return None

    lib.print_config(config)  # type: ignore[code]
    output = Path(output)
    delu.random.seed(config["seed"])
    report = lib.create_report(main, config)  # type: ignore[code]

    # >>> data
    dataset = lib.graph.data.build_dataset(**config["data"])
    if dataset.task.is_regression:
        dataset.data["labels"], regression_label_stats = (
            lib.graph.data.standardize_labels(
                dataset.data["labels"], dataset.data["masks"]
            )
        )
    else:
        regression_label_stats = None

    # Add PEARL embeds
    if config.get("pearl", None) is not None:
        pearl = PEARL(**config["pearl"]).to("cuda:0")
        with torch.no_grad():
            pearl_embeds = pearl(dataset.data["graph"].to("cuda:0")).cpu().numpy()
        if dataset.data.get("num_features", None) is not None:
            dataset.data["num_features"] = np.concatenate(
                [dataset.data["num_features"], pearl_embeds], axis=-1
            )
        else:
            dataset.data["num_features"] = pearl_embeds

    if dataset.data.get("num_features", None) is not None:
        X = {
            part: pd.DataFrame(dataset.data["num_features"][mask])
            for part, mask in dataset.data["masks"].items()
        }
    else:
        X = {part: pd.DataFrame() for part in dataset.data["masks"]}

    if dataset.data.get("ratio_features", None) is not None:
        # Merge binary features to continuous features.
        X = {
            part: pd.concat(
                [X[part], pd.DataFrame(dataset.data["ratio_features"][mask])], axis=1
            )
            for part, mask in dataset.data["masks"].items()
        }
        # Rename columns
        for part in X:
            X[part].columns = range(X[part].shape[1])

    if dataset.data.get("cat_features", None) is not None:
        # Merge categorical features
        categorical_features = range(
            dataset.n_num_features + dataset.n_ratio_features, dataset.n_features
        )
        X = {
            part: pd.concat(
                [
                    X[part],
                    pd.DataFrame(
                        dataset.data["cat_features"][mask], columns=categorical_features
                    ),
                ],
                axis=1,
            )
            for part, mask in dataset.data["masks"].items()
        }
        dataset.data.pop("cat_features")
    else:
        categorical_features = None

    # >>> model
    fit_extra_kwargs: KWArgs = {}
    stopping_rounds = config["model"].pop("stopping_rounds")
    fit_extra_kwargs["callbacks"] = [
        lightgbm.early_stopping(stopping_rounds=stopping_rounds)
    ]

    model_extra_kwargs: dict[str, Any] = {
        "random_state": config["seed"],
        "categorical_features": list(categorical_features)
        if categorical_features is not None
        else None,
    }

    if dataset.task.is_regression:
        model = LGBMRegressor(**config["model"], **model_extra_kwargs)
        fit_extra_kwargs = {"eval_metric": "rmse"}
        predict = model.predict
    else:
        model = LGBMClassifier(**config["model"], **model_extra_kwargs)
        if dataset.task.is_multiclass:
            predict = model.predict_proba
            fit_extra_kwargs = {"eval_metric": "multi_error"}
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
            fit_extra_kwargs = {
                "eval_metric": "average_precision"
                if dataset.task.score == lib.Score.AP
                else "binary_error"
            }

    report["prediction_type"] = "labels" if dataset.task.is_regression else "probs"

    # >>> training
    logger.info("training...")
    with delu.Timer() as timer:
        model.fit(
            X["train"],
            dataset.data["labels"][dataset.data["masks"]["train"]],
            eval_set=[(X["val"], dataset.data["labels"][dataset.data["masks"]["val"]])],
            **config["fit"],
            **fit_extra_kwargs,
        )
    report["time"] = str(timer)
    report["best_iteration"] = model.booster_.best_iteration

    # >>> finish
    predictions: dict[str, np.ndarray] = {
        k: np.asarray(predict(v)) for k, v in X.items()
    }
    if regression_label_stats is not None:
        predictions = {
            k: v * regression_label_stats.std + regression_label_stats.mean
            for k, v in predictions.items()
        }
    report["metrics"] = dataset.task.calculate_metrics(
        predictions,
        report["prediction_type"],  # type: ignore[code]
    )
    lib.dump_summary(output, lib.summarize(report))
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.init()
    lib.run(main)
