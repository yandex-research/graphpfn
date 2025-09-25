import os
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import delu
import dgl
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from typing_extensions import Callable, NotRequired, TypedDict  # noqa: UP035

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    PowerTransformer,
    FunctionTransformer,
)

import lib
from lib import KWArgs, PartKey
from lib.graph.data import GraphDataset

EvalOut = tuple[dict[PartKey, Any], dict[PartKey, np.ndarray], int]


class Config(TypedDict):
    seed: int
    amp: NotRequired[bool]  # Automatic mixed precision in bfloat16.
    data: KWArgs
    pearl: NotRequired[KWArgs]

    shuffle_targets: bool

    bins: NotRequired[KWArgs]
    optimizer: KWArgs
    epoch_size: int
    batch_size: int
    # seq_len_train: int
    min_train_ratio: float
    seq_len_pred: int
    target_transform: NotRequired[str]

    patience: int
    n_epochs: int
    n_lr_warmup_epochs: NotRequired[int]
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]
    full_finetune: NotRequired[bool]
    finetune_mode: NotRequired[
        Literal[
            "full",
            "ln",
            "head",
            "ln+head",
            "embeds",
            "embeds+head",
            "embeds+ln+head",
            "top_n_layers",
            "lora",
            "none",
        ]
    ]
    finetune_layers: NotRequired[int]
    randperm: NotRequired[bool]


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


class CandidateQueue:
    def __init__(
        self, train_size: int, n_candidates: int | float, device: torch.device
    ) -> None:
        assert train_size > 0
        if isinstance(n_candidates, int):
            assert 0 < n_candidates < train_size
            self._n_candidates = n_candidates
        else:
            assert 0.0 < n_candidates < 1.0
            self._n_candidates = int(n_candidates * train_size)
        self._train_size = train_size
        self._candidate_queue = torch.tensor([], dtype=torch.int64, device=device)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._candidate_queue) < self._n_candidates:
            self._candidate_queue = torch.cat(
                [
                    self._candidate_queue,
                    torch.randperm(
                        self._train_size, device=self._candidate_queue.device
                    ),
                ]
            )
        candidate_indices, self._candidate_queue = self._candidate_queue.split(
            [self._n_candidates, len(self._candidate_queue) - self._n_candidates]
        )
        return candidate_indices


def preprocess_targets(
    dataset: GraphDataset,
    config: Config,
) -> dict:
    if dataset.task.is_regression:
        dataset.data["labels"], regression_label_stats = lib.data.standardize_labels(
            dataset.data["labels"], dataset.data["masks"]
        )

        target_transform = config.get("target_transform", None)

        if target_transform == "power":
            target_transform = Pipeline(
                [("power", PowerTransformer()), ("standard", StandardScaler())]
            ).fit(dataset.data["labels"][dataset.data["masks"]["train"]].reshape(-1, 1))
        elif target_transform == "quantile":
            target_transform = QuantileTransformer(
                output_distribution="normal", random_state=config["seed"]
            ).fit(dataset.data["labels"][dataset.data["masks"]["train"]].reshape(-1, 1))
        elif target_transform is None:
            target_transform = FunctionTransformer(func=None)
        else:
            raise ValueError(f"Unknown target_transform {target_transform}")

        dataset.data["labels"] = (
            target_transform.transform(dataset.data["labels"].reshape(-1, 1))
            .astype(np.float32)
            .squeeze()
        )
    else:
        regression_label_stats = None
        target_transform = None

    return {
        "target_transform": target_transform,
        "regression_label_stats": regression_label_stats,
    }


def preprocess_tfm_features(
    dataset: GraphDataset, config: Config, add_ratio_features_to="num_features"
) -> np.ndarray:
    data = {
        "num_features": dataset.data.get("num_features", None),
        "ratio_features": dataset.data.get("ratio_features", None),
        "cat_features": dataset.data.get("cat_features", None),
    }

    # Convert ratio features
    if data["ratio_features"] is not None:
        x_ratio = data["ratio_features"]
        if data[add_ratio_features_to] is None:
            data[add_ratio_features_to] = np.zeros(
                (x_ratio.shape[0], 0), dtype=np.float32
            )
        data[add_ratio_features_to] = np.column_stack(
            [data[add_ratio_features_to], x_ratio.astype(np.float32)]
        )
        del x_ratio

    # Remove features with just one unique value in the training set.
    for features_type in ["num_features", "cat_features"]:
        features = data.pop(features_type, None)
        if features is None:
            continue
        n_features = features.shape[1]
        good_features_idx = [
            i
            for i in range(n_features)
            if len(np.unique(features[dataset.data["masks"]["train"], i])) > 1
        ]
        if len(good_features_idx) < n_features:
            logger.info(
                f"Deleted {n_features - len(good_features_idx)} features"
                "with a single value in the train set"
            )
            features = features[:, good_features_idx]
        data[features_type] = features

    # concat all features
    tfm_features_list = []
    for features_type in ["num_features", "cat_features"]:
        features = data.get(features_type, None)
        if features is not None:
            tfm_features_list.append(features)
    tfm_features = np.concatenate(tfm_features_list, axis=1)

    return tfm_features


def main(
    config: Config, output: str | Path, *, force: bool = False
) -> None | lib.JSONDict:
    # >>> start
    config, output = lib.check(config, output, config_type=Config)

    if not lib.start(main, output, force=force):
        return None

    lib.print_config(config)
    print()
    output = Path(output)
    delu.random.seed(config["seed"])
    device = lib.get_device()
    logger.info(f"Device: {device}")
    report = lib.create_report(main, config)

    # >>> dataset
    dataset = lib.graph.data.build_dataset(**config["data"])
    assert dataset.task.is_transductive

    preprocess_targets_results = preprocess_targets(dataset, config)
    regression_label_stats = preprocess_targets_results["regression_label_stats"]

    tfm_features = preprocess_tfm_features(dataset, config)

    dataset = dataset.to_torch(device)
    tfm_features = torch.tensor(tfm_features, device=device)
    Y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(
        dtype=torch.float32, device=device
    )

    # Print info about dataset
    logger.info(f"Final: {dataset.size()=} {dataset.size('train')=}")
    logger.info(
        f"Final: {dataset.n_num_features=} {dataset.n_ratio_features=}"
        f" {dataset.n_cat_features=}"
    )

    # >>> model
    models = nn.ModuleList([])

    logger.info(
        f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    tfm = lib.tfm.load_tfm(tfm_name="LimiX", tfm_config={})

    models.append(tfm)

    if "pearl" in config:
        pearl = PEARL(**config["pearl"])
        models.append(pearl)

    report["n_parameters"] = lib.deep.get_n_parameters(models)
    logger.info(f"n_parameters = {report['n_parameters']}")
    report["prediction_type"] = "labels" if dataset.task.is_regression else "probs"
    models.to(device)
    if torch.cuda.device_count() > 1:
        models = nn.DataParallel(models)

    # >>> training

    # Add trainable params for TFM
    if config.get("finetune_mode", "") == "none":
        params = []
    elif (
        config.get("full_finetune", False)
        or config.get("finetune_mode", None) == "full"
    ):
        params = lib.deep.make_parameter_groups(tfm)
    else:
        raise NotImplementedError("Currently, only full finetune is supported")

    # Add params for other models
    assert models[0] is tfm
    if isinstance(params, dict):
        params = [params]
    for m in models[1:]:
        params.append({"params": m.parameters()})

    optimizer = lib.deep.make_optimizer(
        **config["optimizer"],
        params=params,
    )
    if dataset.task.is_regression:
        assert regression_label_stats is not None
        loss_fn = F.mse_loss
        pred_transform = (  # noqa: E731
            lambda tensor: tensor * regression_label_stats.std
            + regression_label_stats.mean
        )
    else:
        loss_fn = F.cross_entropy
    gradient_clipping_norm = config.get("gradient_clipping_norm")

    epoch_size = config["epoch_size"]
    eval_batch_size = 4096
    chunk_size = None
    generator = torch.Generator(device).manual_seed(config["seed"])

    report["metrics"] = {"val": {"score": -math.inf}}
    if "n_lr_warmup_epochs" in config:
        n_warmup_steps = min(10000, config["n_lr_warmup_epochs"] * epoch_size)
        n_warmup_steps = max(1, math.trunc(n_warmup_steps / epoch_size)) * epoch_size
        logger.info(f"{n_warmup_steps=}")
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=n_warmup_steps
        )
    else:
        lr_scheduler = None

    timer = delu.tools.Timer()
    parameter_statistics = config.get("parameter_statistics", config["seed"] == 1)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    logger.info(f"AMP enabled: {amp_enabled}")

    @torch.autocast(
        device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
    )
    def apply_model(part: PartKey, idx_train: Tensor, idx: Tensor) -> Tensor:
        # TFM input and targets
        tfm_input = [tfm_features]
        if "pearl" in config:
            tfm_input.append(pearl(dataset.data["graph"]))
        tfm_input = torch.cat(tfm_input, dim=-1)
        tfm_input = torch.cat(
            [
                tfm_input[dataset.data["masks"]["train"]][idx_train],
                tfm_input[dataset.data["masks"][part]][idx],
            ],
            dim=1,
        )
        tfm_targets_train = Y_train[idx_train]

        # Target preprocessing
        if config["shuffle_targets"] and dataset.task.is_multiclass:
            targets_permutation = torch.randperm(
                dataset.task.compute_n_classes(), device=tfm_targets_train.device
            )
            tfm_targets_train = targets_permutation[tfm_targets_train.long()].float()

        # For flash attention reproducibility
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            out = tfm.forward(
                x=tfm_input,
                y=tfm_targets_train,
                eval_pos=tfm_targets_train.shape[1],
                task_type="reg" if dataset.task.is_regression else "cls",
            ).float()

            if dataset.task.is_regression:
                out = out.squeeze(-1)

        # Target postprocessing
        if config["shuffle_targets"] and dataset.task.is_multiclass:
            out = out[..., targets_permutation]

        return out

    @torch.inference_mode()
    def evaluate(parts: list[PartKey], eval_batch_size: int) -> EvalOut:
        models.eval()
        predictions: dict[PartKey, np.ndarray] = {}
        # using the whole train on evaluation
        idx_train = torch.arange(dataset.size("train"), device=device)

        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(
                                    part, idx_train.unsqueeze(0), idx.unsqueeze(0)
                                ).squeeze(0)
                                for idx in tqdm(
                                    torch.arange(
                                        dataset.size(part),
                                        device=device,
                                    ).split(eval_batch_size)
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    delu.cuda.free_memory()
                    logger.warning(f"eval_batch_size = {eval_batch_size}")
                else:
                    break
            if not eval_batch_size:
                RuntimeError("Not enough memory even for eval_batch_size=1")

        if regression_label_stats is not None:
            predictions = {
                k: pred_transform(torch.from_numpy(v).to(device)).cpu().numpy()  # pyright: ignore
                for k, v in predictions.items()
            }
        else:
            predictions = {
                k: scipy.special.softmax(
                    v[..., : dataset.task.compute_n_classes()], axis=-1
                )
                for k, v in predictions.items()
            }
            if dataset.task.is_binclass:
                predictions = {k: v[..., 1] for k, v in predictions.items()}

        metrics = (
            dataset.task.calculate_metrics(predictions, report["prediction_type"])
            if lib.are_valid_predictions(predictions)
            else {x: {"score": -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def train_loop(
        *,
        step_fn: Callable[[Tensor, Tensor], Tensor],
        eval_fn: Callable[..., tuple],
        n_steps: int,
        patience: int,
        report_key: str,
        chunk_size=None,
        eval_batch_size=eval_batch_size,
    ):
        def save_checkpoint(step) -> None:
            lib.dump_checkpoint(
                output,
                {
                    "step": step,
                    "model": models.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.get_state(),
                    "random_state": delu.random.get_state(),
                    "early_stopping": early_stopping,
                    "report": report,
                    "timer": timer,
                    "training_log": training_log,
                }
                | (
                    {}
                    if lr_scheduler is None
                    else {"lr_scheduler": lr_scheduler.state_dict()}
                ),
            )
            lib.dump_report(output, report)
            lib.backup_output(output)

        step = 0
        early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        report[report_key] = {"metrics": {"val": {"score": -math.inf}}}

        if n_steps == 0:
            save_checkpoint(step)

        while n_steps == -1 or step < n_steps:
            print(f"[...] {output} | {timer}")

            # >>>
            models.train()
            epoch_losses = []
            logs_train = defaultdict(list)

            n_candidates = config["batch_size"] * min(
                config["seq_len_pred"],
                int(dataset.size("train") * (1 - config["min_train_ratio"])),
            )

            idx_queue = CandidateQueue(
                dataset.size("train"),
                n_candidates=n_candidates,
                device=device,
            )
            delu.cuda.free_memory()

            for _ in tqdm(
                range(epoch_size),
                desc=f"Epoch {step // epoch_size} Step {step}",
            ):
                if config.get("randperm", False):
                    idx = torch.randperm(dataset.size("train"), device=device)
                    idx = idx[:n_candidates]
                else:
                    idx = next(idx_queue)
                idx = idx.view(config["batch_size"], -1)

                mask = idx.new_ones(
                    (config["batch_size"], dataset.size("train")), dtype=torch.bool
                )
                mask[
                    torch.arange(config["batch_size"], device=mask.device).unsqueeze(
                        -1
                    ),
                    idx,
                ] = False
                idx_train = (
                    torch.arange(dataset.size("train"), device=idx.device)
                    .expand(config["batch_size"], dataset.size("train"))[mask]
                    .view(config["batch_size"], -1)
                )

                optimizer.zero_grad()
                loss = step_fn(idx_train, idx)
                loss.backward()

                for k, v in log_dict.items():
                    logs_train[k].append(v)

                if parameter_statistics and (
                    step % epoch_size == 0  # The first batch of the epoch.
                    or step // epoch_size == 0  # The first epoch.
                ):
                    for k, v in lib.deep.compute_parameter_stats(models).items():
                        writer.add_scalars(
                            f"{report_key}/{k}", v, step, timer.elapsed()
                        )
                        del k, v

                if gradient_clipping_norm is not None:
                    nn.utils.clip_grad.clip_grad_norm_(
                        models.parameters(), gradient_clipping_norm
                    )
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                step += 1
                epoch_losses.append(loss.detach())

            epoch_losses = torch.stack(epoch_losses).tolist()
            mean_loss = statistics.mean(epoch_losses)

            metrics, predictions, eval_batch_size = eval_fn(["val"], eval_batch_size)
            metrics["train"] = {}
            for k, v in logs_train.items():
                metrics["train"][k] = np.mean(v).item()

            training_log.append(
                {
                    "epoch-losses": epoch_losses,
                    "metrics": metrics,
                    "time": timer.elapsed(),
                }
            )
            lib.print_metrics(mean_loss, metrics)
            writer.add_scalars(
                f"{report_key}/loss", {"train": mean_loss}, step, timer.elapsed()
            )
            for part in metrics:
                for k in metrics[part].keys():
                    if k != "score":
                        continue
                    writer.add_scalars(
                        f"{report_key}/{k}",
                        {part: metrics[part][k]},
                        step,
                        timer.elapsed(),
                    )

            if metrics["val"]["score"] > report[report_key]["metrics"]["val"]["score"]:
                print("🌸 New best epoch! 🌸")
                report[report_key]["best_step"] = step
                report[report_key]["metrics"] = metrics
                save_checkpoint(step)
                lib.dump_predictions(output, predictions)

            early_stopping.update(metrics["val"]["score"])
            if early_stopping.should_stop() or not lib.are_valid_predictions(
                predictions
            ):
                break

            print()
        return chunk_size, eval_batch_size

    def step_fn(idx_train, idx):
        "idx is big set of datasets"
        if dataset.task.is_classification:
            return loss_fn(
                apply_model("train", idx_train, idx).permute(0, 2, 1),
                Y_train[idx].long(),
            )
        else:
            return loss_fn(
                apply_model("train", idx_train, idx).permute(1, 0),
                Y_train[idx].transpose(0, 1),
            ).mean()

    log_dict = {}
    timer.run()
    chunk_size, eval_batch_size = train_loop(
        step_fn=step_fn,
        eval_fn=evaluate,
        n_steps=config["n_epochs"],
        patience=config["patience"],
        report_key="train",
        chunk_size=chunk_size,
    )
    report["time"] = str(timer)

    # >>> finish
    models.load_state_dict(lib.load_checkpoint(output)["model"])
    logger.info("Final Eval")
    report["metrics"], predictions, _ = evaluate(
        ["train", "val", "test"], eval_batch_size
    )
    report["chunk_size"] = chunk_size
    report["eval_batch_size"] = eval_batch_size
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    # to free-up space
    if config["seed"] > 0:
        os.remove(output / "checkpoint.pt")
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.configure_torch()
    lib.run(main)
