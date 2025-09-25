import datetime
import math
from functools import partial
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import delu
import dgl
import numpy as np
import rtdl_num_embeddings
import scipy
import scipy.special
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor

import lib
import lib.data
import lib.deep
import lib.env
import lib.graph.data
import lib.graph.deep
import lib.graph.util
from lib import KWArgs, PartKey


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        n_num_features: int,
        n_classes: None | int,
        bin_edges: list[Tensor] | None,
        backbone: dict,
        num_features: None | dict = None,
        inp_out_type: str = "mlp",
        pearl: dict | None = None,
    ):
        super().__init__()

        # >>> Continuous (numerical) features
        if num_features is None:
            assert bin_edges is None
            self.num_embedding = None
            d_flat = n_features

        else:
            if num_features["name"] == "piecewise":
                assert bin_edges is not None

            self.num_embedding = lib.graph.deep.make_module(
                **num_features,
                n_features=n_num_features,
                bin_edges=bin_edges,
            )
            d_embed = num_features["d_embed"]
            d_flat = n_features + n_num_features * (d_embed - 1)

        # >>> PEARL
        if pearl is not None:
            self.pearl = PEARL(**pearl)
            d_flat += pearl["output_dim"]
        else:
            self.pearl = None

        # >>> Input
        d_hidden = backbone["d_hidden"]
        dropout = backbone["dropout"]

        self.input = {
            "mlp": (
                nn.Sequential(
                    nn.Linear(d_flat, d_hidden),
                    nn.Dropout(dropout),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_hidden),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_flat, d_hidden),
                )
            ),
        }[inp_out_type]

        # >>> Backbone
        self.backbone = lib.graph.deep.make_module(
            **backbone,
        )

        # >>> Output
        d_out = 1 if n_classes is None else n_classes
        self.output = {
            "mlp": (
                nn.Sequential(
                    lib.graph.deep.NORM_MODULES[backbone["norm_name"]](d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_out),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_hidden, d_out),
                )
            ),
        }[inp_out_type]

    def forward(
        self,
        graph: dgl.DGLGraph,
        x_num: None | Tensor = None,
        x_other: None | Tensor = None,
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_embedding is None else self.num_embedding(x_num))

        if x_other is not None:
            x.append(x_other)

        if self.pearl is not None:
            x.append(self.pearl(graph))

        x = torch.cat(x, dim=1)
        x = self.input(x)
        x = self.backbone(graph, x)
        x = self.output(x)
        return x


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
            model = lambda graph, x: self.output(self.backbone(graph, self.input(x)))
            if self.checkpointing:
                model = partial(
                    torch.utils.checkpoint.checkpoint,
                    model,
                    use_reentrant=False,
                )
            x = model(graph, x)
            x_list.append(x)

        return torch.stack(x_list, dim=-1).mean(-1)


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    optimizer: KWArgs
    n_steps: int
    patience: int
    bins: NotRequired[KWArgs]
    amp_dtype: NotRequired[Literal["bfloat16", "float16"]]


def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
) -> None | lib.JSONDict:
    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    if not lib.start(main, output, force=force):
        return None

    lib.print_config(config)
    print()
    delu.random.seed(config["seed"])
    device = lib.get_device()
    logger.info(f"Device: {device}")
    report = lib.create_report(main, config)

    # >>> Data
    dataset = lib.graph.data.build_dataset(**config["data"])
    assert dataset.task.is_transductive

    if dataset.task.is_regression:
        dataset.data["labels"], regression_label_stats = (
            lib.graph.data.standardize_labels(
                dataset.data["labels"], dataset.data["masks"]
            )
        )
    else:
        regression_label_stats = None

    dataset = dataset.to_torch(device)
    dataset.data["labels"] = (
        dataset.data["labels"].to(torch.long)
        if dataset.task.is_classification
        else dataset.data["labels"].to(torch.float)
    )

    # >>> Model
    if "bins" in config:
        _num_features_train = dataset.data["num_features"][
            dataset.data["masks"]["train"]
        ]
        bin_edges = rtdl_num_embeddings.compute_bins(
            _num_features_train,
            **config["bins"],
        )
        logger.info(f"Bin counts: {[len(x) - 1 for x in bin_edges]}")
    else:
        bin_edges = None

    model = Model(
        n_features=dataset.n_features,
        n_num_features=dataset.n_num_features,
        n_classes=dataset.task.try_compute_n_classes(),
        bin_edges=bin_edges,
        **config["model"],
    )
    report["n_parameters"] = lib.deep.get_n_parameters(model)
    logger.info(f"Number of parameters: {report['n_parameters']}")
    report["prediction_type"] = "labels" if dataset.task.is_regression else "probs"
    model.to(device)
    print(model)

    # >>> Train
    optimizer = lib.deep.make_optimizer(
        **config["optimizer"], params=lib.deep.make_parameter_groups(model)
    )
    base_loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.cross_entropy
    )

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return base_loss_fn(y_pred, y_true)

    step = 0
    training_log = []
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config["patience"], mode="max")
    writer = torch.utils.tensorboard.SummaryWriter(output)

    amp_dtype = config.get("amp_dtype")
    if amp_dtype == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                'amp_dtype is set to "bfloat16" in the config.'
                f" However, the current {device.type.upper()} device"
                " does not support bfloat16"
            )
        amp_dtype = torch.bfloat16

    elif amp_dtype == "float16":
        amp_dtype = torch.float16

    grad_scaler = (
        torch.amp.GradScaler(device.type) if amp_dtype is torch.float16 else None
    )
    logger.info(f"AMP dtype: {amp_dtype}")

    graph = dataset.data["graph"].to(device)
    x_num = dataset.data["num_features"]
    x_other = []
    for data_key in ["cat_features", "ratio_features"]:
        if dataset.data[data_key] is not None:
            x_other.append(dataset.data[data_key])
    x_other = torch.cat(x_other, dim=1) if x_other else None

    @torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    @lib.catch_oom_decorator()
    def apply_model() -> Tensor:
        return model(graph, x_num, x_other).squeeze(-1).float()

    @torch.inference_mode()
    def evaluate() -> tuple[dict[PartKey, Any], dict[PartKey, np.ndarray]]:
        model.eval()
        outputs = apply_model().cpu().numpy()

        if dataset.task.is_regression:
            assert regression_label_stats is not None
            _predictions: np.ndarray = (
                outputs * regression_label_stats.std + regression_label_stats.mean
            )
        else:
            _predictions: np.ndarray = scipy.special.softmax(outputs, axis=-1)
            if dataset.task.is_binclass:
                _predictions = _predictions[..., 1]

        predictions: dict[PartKey, np.ndarray] = {
            part: _predictions[dataset.task.masks[part]] for part in lib.DATA_PARTS
        }
        metrics = (
            dataset.task.calculate_metrics(predictions, report["prediction_type"])
            if lib.are_valid_predictions(predictions)
            else lib.get_default_metrics()
        )
        return metrics, predictions

    def save_checkpoint() -> None:
        lib.dump_checkpoint(
            output,
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "random_state": delu.random.get_state(),
                "early_stopping": early_stopping,
                "report": report,
                "timer": timer,
                "training_log": training_log,
            },
        )

    print()
    timer.run()
    while step < config["n_steps"]:
        step_start_time = timer.elapsed()

        model.train()
        optimizer.zero_grad()

        outputs = apply_model()
        targets = dataset.data["labels"]

        loss = loss_fn(
            outputs[dataset.data["masks"]["train"]],
            targets[dataset.data["masks"]["train"]],
        )
        if grad_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        step += 1
        step_end_time = timer.elapsed()
        step_loss = loss.detach()

        metrics, predictions = evaluate()
        val_score_improved = (
            "metrics" not in report
            or metrics["val"]["score"] > report["metrics"]["val"]["score"]
        )

        training_log.append({"metrics": metrics, "time": timer.elapsed()})
        print(
            f"{'+' if val_score_improved else ' '}"
            f" [step] {step:>4}"
            f" [val] {metrics['val']['score']:.3f}"
            f" [test] {metrics['test']['score']:.3f}"
            f" [loss] {step_loss:.4f}"
            f" [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}"
            f" [s/it] {(step_end_time - step_start_time):.4f}"
        )
        writer.add_scalars("loss", {"train": step_loss}, step, timer.elapsed())

        if val_score_improved:
            report["best_step"] = step
            report["metrics"] = metrics
            save_checkpoint()
            lib.dump_report(output, report)
            # lib.dump_predictions(output, predictions)
            lib.dump_summary(output, lib.summarize(report))

        early_stopping.update(metrics["val"]["score"])
        if not lib.are_valid_predictions(predictions) or early_stopping.should_stop():
            break

    report["time"] = timer.elapsed()
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.init()
    lib.run(main)
