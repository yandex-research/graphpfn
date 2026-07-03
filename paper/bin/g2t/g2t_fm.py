import math
import os
import statistics
from functools import partial
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
from typing_extensions import NotRequired, TypedDict  # noqa: UP035

import lib
from lib import KWArgs, PartKey
from lib.ensemble import (
    Ensemble,
    apply_feature_transform,
    apply_target_transform,
    create_ensemble,
    reverse_target_perm,
)
from lib.graph.data import GraphDataset
from lib.graph.pearl import PEARL
from lib.tfm import TFMBase, load_tfm
from lib.util import TaskType


class EnsembleConfig(TypedDict):
    n_members: int
    shuffle_features: bool
    shuffle_targets: bool
    max_columns: NotRequired[int]


class TFMConfig(TypedDict):
    name: str
    config: NotRequired[dict]


class Config(TypedDict):
    seed: int
    amp: NotRequired[bool]  # Automatic mixed precision in bfloat16.
    data: KWArgs
    nfa: NotRequired[KWArgs]
    encodings: NotRequired[KWArgs]
    d_reduction: NotRequired[KWArgs]
    transform: lib.graph.data.TransformConfig
    pearl: NotRequired[KWArgs]
    tfm: TFMConfig

    ensemble: EnsembleConfig

    bins: NotRequired[KWArgs]
    optimizer: KWArgs
    epoch_size: int
    min_train_ratio: float
    seq_len_pred: int
    target_transform: NotRequired[str]

    patience: int
    n_epochs: int
    n_lr_warmup_epochs: NotRequired[int]
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]
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


class G2T(nn.Module):
    def __init__(self, tfm: TFMBase, pearl: PEARL | None) -> None:
        super().__init__()
        self.tfm = tfm
        self.pearl = pearl

    def forward(
        self,
        features: Tensor,
        y_train: Tensor,
        train_mask: Tensor,
        eval_mask: Tensor,
        graph: dgl.DGLGraph,
        *,
        task_type: TaskType,
    ) -> Tensor:
        if self.pearl is not None:
            pearl_features = self.pearl(graph)
            features = torch.cat([features, pearl_features], dim=-1)

        return self.tfm(
            x_train=features[train_mask],
            y_train=y_train,
            x_eval=features[eval_mask],
            task_type=task_type,
        )


def _batched_forward(
    g2t: G2T,
    features: Tensor,
    y_train: Tensor,
    train_mask: Tensor,
    part_mask: Tensor,
    eval_batch_size: int,
    graph: dgl.DGLGraph,
    task_type: TaskType,
    amp_enabled: bool,
    device: torch.device,
) -> Tensor:
    part_indices = torch.where(part_mask)[0]
    chunks = part_indices.split(eval_batch_size)
    results = []
    for chunk in chunks:
        eval_mask = torch.zeros(part_mask.shape[0], dtype=torch.bool, device=device)
        eval_mask[chunk] = True
        with torch.autocast(
            device.type,
            enabled=amp_enabled,
            dtype=torch.bfloat16 if amp_enabled else None,
        ):
            preds = g2t(
                features=features,
                y_train=y_train,
                train_mask=train_mask,
                eval_mask=eval_mask,
                graph=graph,
                task_type=task_type,
            )
        results.append(preds)
    return torch.cat(results, dim=0)


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


@torch.inference_mode()
def evaluate(
    g2t: G2T,
    dataset: GraphDataset,
    feature_groups: list[Tensor],
    y_train: Tensor,
    prediction_type: str,
    regression_label_stats: Any,
    device: torch.device,
    amp_enabled: bool,
    ensemble: Ensemble,
    eval_batch_size: int,
    parts: list[PartKey] = ["val", "test"],
) -> tuple[dict[PartKey, Any], dict[PartKey, np.ndarray], int]:
    g2t.eval()
    graph = dataset.data["graph"]
    train_mask = dataset.data["masks"]["train"]

    ensemble_predictions: dict[PartKey, list[Tensor]] = {part: [] for part in parts}

    for member in tqdm(ensemble):
        features_t = apply_feature_transform(feature_groups, member)
        y_train_t = apply_target_transform(y_train, member)

        for part in parts:
            part_mask = dataset.data["masks"][part]

            while eval_batch_size:
                try:
                    preds = _batched_forward(
                        g2t,
                        features_t,
                        y_train_t,
                        train_mask,  # type: ignore
                        part_mask,  # type: ignore
                        eval_batch_size,
                        graph,
                        dataset.task.type_,
                        amp_enabled,
                        device,
                    )
                    break
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    delu.cuda.free_memory()
                    logger.warning(f"eval_batch_size = {eval_batch_size}")
            if not eval_batch_size:
                raise RuntimeError("Not enough memory even for eval_batch_size=1")

            preds = reverse_target_perm(preds, member)  # type: ignore
            ensemble_predictions[part].append(preds)

    predictions: dict[PartKey, np.ndarray] = {}
    for part in parts:
        aggregated = ensemble.aggregate_predictions(
            ensemble_predictions[part], dataset.task.type_
        )

        if regression_label_stats is not None:
            part_pred = (
                (aggregated * regression_label_stats.std + regression_label_stats.mean)
                .cpu()
                .numpy()
            )
        else:
            part_pred = scipy.special.softmax(
                aggregated.cpu().numpy()[..., : dataset.task.compute_n_classes()],
                axis=-1,
            )
            if dataset.task.is_binclass:
                part_pred = part_pred[..., 1]

        predictions[part] = part_pred

    metrics = (
        dataset.task.calculate_metrics(predictions, prediction_type)
        if lib.are_valid_predictions(predictions)
        else {x: {"score": -999999.0} for x in predictions}
    )
    return metrics, predictions, eval_batch_size


def compute_loss(
    g2t: G2T,
    dataset: GraphDataset,
    feature_groups: list[Tensor],
    y_train: Tensor,
    context_idx: Tensor,
    batch_idx: Tensor,
    device: torch.device,
    amp_enabled: bool,
    ensemble: Ensemble,
) -> Tensor:
    _, member = ensemble.sample_member()
    features_t = apply_feature_transform(feature_groups, member)
    y_train_t = apply_target_transform(y_train, member)

    train_node_idx = torch.where(dataset.data["masks"]["train"])[0]  # type: ignore
    train_mask = torch.zeros(dataset.size(), dtype=torch.bool, device=device)
    train_mask[train_node_idx[context_idx]] = True
    eval_mask = torch.zeros(dataset.size(), dtype=torch.bool, device=device)
    eval_mask[train_node_idx[batch_idx]] = True

    with torch.autocast(
        device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
    ):
        preds = g2t(
            features=features_t,
            y_train=y_train_t[context_idx],
            train_mask=train_mask,
            eval_mask=eval_mask,
            graph=dataset.data["graph"],
            task_type=dataset.task.type_,
        )

    if dataset.task.is_regression:
        loss = F.mse_loss(input=preds, target=y_train_t[batch_idx])
    else:
        loss = F.cross_entropy(input=preds, target=y_train_t[batch_idx].long())

    return loss


def main(
    config: Config, output: str | Path, *, force: bool = False
) -> None | lib.JSONDict:
    # >>> start
    config, output = lib.check(config, output, config_type=Config)

    if not lib.start(main, output, force=force):
        return None

    lib.print_config(config)  # type: ignore
    print()
    output = Path(output)
    delu.random.seed(config["seed"])
    device = lib.get_device()
    logger.info(f"Device: {device}")
    report = lib.create_report(main, config)  # type: ignore

    # For memory footprint benchmarking
    delu.cuda.free_memory()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    timer = delu.tools.Timer()
    timer.run()

    # >>> dataset
    dataset_timer = delu.tools.Timer()
    dataset_timer.run()

    dataset = lib.graph.data.GraphDataset.from_dir(**config["data"])
    assert dataset.task.is_transductive

    d_reduction_config = config.get("d_reduction", {})

    def transform_and_reduce(
        features: dict[str, np.ndarray | None], transform_key: str
    ) -> dict[str, np.ndarray | None]:
        features = lib.graph.data.transform_features(
            features, dataset.task, **config["transform"].get(transform_key, {})
        )
        return lib.graph.data.apply_d_reduction(
            features, dataset.task, **d_reduction_config
        )

    feature_sources = [transform_and_reduce(dataset.features, "features")]

    if "nfa" in config:
        nfa = lib.graph.data.apply_nfa(dataset, **config["nfa"])
        feature_sources.append(transform_and_reduce(nfa, "nfa"))

    if "encodings" in config:
        for name, params in config["encodings"].items():
            enc_arr = lib.graph.data.get_structural_encodings(
                dataset.data["graph"],
                name=name,
                params=params,  # type: ignore
            )
            enc_dict: dict[str, np.ndarray | None] = dict.fromkeys(
                lib.graph.data.GRAPH_FEATURE_KEYS
            )
            enc_dict["num_features"] = enc_arr
            feature_sources.append(transform_and_reduce(enc_dict, "encodings"))

    regression_label_stats = lib.graph.data.prepare_labels(
        dataset, config["transform"]["labels"]
    )

    for key in lib.graph.data.GRAPH_FEATURE_KEYS:
        dataset.data[key] = None
    dataset = dataset.to_torch(device)

    train_mask = dataset.data["masks"]["train"]
    feature_groups: list[Tensor] = []
    for source in feature_sources:
        group_np = lib.graph.data.flatten_features(source)
        if group_np is None:
            continue
        group = torch.tensor(group_np, device=device)
        group = lib.graph.data.drop_constant_features(group, train_mask)  # type: ignore
        if group.shape[1] == 0:
            continue
        feature_groups.append(group)

    y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(  # type: ignore
        dtype=torch.float32, device=device
    )

    feature_group_sizes = [g.shape[1] for g in feature_groups]
    ensemble = create_ensemble(
        n_members=config["ensemble"]["n_members"],
        n_features=sum(feature_group_sizes),
        n_classes=dataset.task.try_compute_n_classes(),
        feature_group_sizes=feature_group_sizes,
        shuffle_features=config["ensemble"]["shuffle_features"],
        shuffle_targets=config["ensemble"]["shuffle_targets"],
        max_columns=config["ensemble"].get("max_columns"),
        device=device,
        seed=config["seed"],
    )

    report["dataset_time"] = str(dataset_timer.elapsed())

    # >>> model
    logger.info(
        f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    tfm = load_tfm(**config["tfm"], device=device, checkpointing=True)

    pearl: PEARL | None = None
    if "pearl" in config:
        pearl = PEARL(**config["pearl"])
        pearl.load_state_dict(torch.load("checkpoints/pearl_random.pt"))

    g2t = G2T(tfm=tfm, pearl=pearl)

    report["n_parameters"] = lib.deep.get_n_parameters(g2t)
    logger.info(f"n_parameters = {report['n_parameters']}")
    report["prediction_type"] = "labels" if dataset.task.is_regression else "probs"
    g2t.to(device)

    # >>> training

    if config.get("finetune_mode", "") == "none":
        params: list[dict[str, Any]] = []
    elif config.get("finetune_mode", None) == "full":
        params = lib.deep.make_parameter_groups(g2t.tfm)
    else:
        raise ValueError("Unknown finetune mode")

    if g2t.pearl is not None:
        params.append({"params": g2t.pearl.parameters()})

    optimizer = lib.deep.make_optimizer(
        **config["optimizer"],
        params=params,
    )
    gradient_clipping_norm = config.get("gradient_clipping_norm")

    epoch_size = config["epoch_size"]
    eval_batch_size = 2**15
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

    parameter_statistics = config.get("parameter_statistics", config["seed"] == 1)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    logger.info(f"AMP enabled: {amp_enabled}")

    prediction_type = report["prediction_type"]

    step_fn = partial(
        compute_loss,
        g2t=g2t,
        dataset=dataset,
        feature_groups=feature_groups,
        y_train=y_train,
        device=device,
        amp_enabled=amp_enabled,
        ensemble=ensemble,
    )

    eval_fn = partial(
        evaluate,
        g2t=g2t,
        dataset=dataset,
        feature_groups=feature_groups,
        y_train=y_train,
        prediction_type=prediction_type,
        regression_label_stats=regression_label_stats,
        device=device,
        amp_enabled=amp_enabled,
        ensemble=ensemble,
    )

    def train_loop(
        *,
        n_steps: int,
        patience: int,
        report_key: str,
        eval_batch_size: int = eval_batch_size,
    ):
        def save_checkpoint(step) -> None:
            lib.dump_checkpoint(
                output,
                {
                    "step": step,
                    "model": g2t.state_dict(),
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
            lib.backup(output)

        step = 0
        early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        report[report_key] = {"metrics": {"val": {"score": -math.inf}}}

        if n_steps == 0:
            save_checkpoint(step)

        while n_steps == -1 or step < n_steps:
            print(f"[...] {output} | {timer}")

            # >>>
            g2t.train()
            epoch_losses = []

            n_candidates = min(
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
                    batch_idx = torch.randperm(dataset.size("train"), device=device)
                    batch_idx = batch_idx[:n_candidates]
                else:
                    batch_idx = torch.unique(next(idx_queue))

                mask = torch.ones(
                    dataset.size("train"), dtype=torch.bool, device=device
                )
                mask[batch_idx] = False
                context_idx = torch.where(mask)[0]

                optimizer.zero_grad()
                loss = step_fn(context_idx=context_idx, batch_idx=batch_idx)
                loss.backward()

                if parameter_statistics and (
                    step % epoch_size == 0  # The first batch of the epoch.
                    or step // epoch_size == 0  # The first epoch.
                ):
                    for k, v in lib.deep.compute_parameter_stats(g2t).items():
                        writer.add_scalars(
                            f"{report_key}/{k}", v, step, timer.elapsed()
                        )
                        del k, v

                if gradient_clipping_norm is not None:
                    nn.utils.clip_grad.clip_grad_norm_(
                        g2t.parameters(), gradient_clipping_norm
                    )
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                step += 1
                epoch_losses.append(loss.detach())

            epoch_losses = torch.stack(epoch_losses).tolist()
            mean_loss = statistics.mean(epoch_losses)

            metrics, predictions, eval_batch_size = eval_fn(
                eval_batch_size=eval_batch_size, parts=["val"]
            )
            metrics["train"] = {}

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
        return eval_batch_size

    timer.run()
    eval_batch_size = train_loop(
        n_steps=config["n_epochs"],
        patience=config["patience"],
        report_key="train",
    )

    # >>> finish
    g2t.load_state_dict(lib.load_checkpoint(output)["model"])
    logger.info("Final Eval")
    inference_timer = delu.tools.Timer()
    inference_timer.run()
    report["metrics"], predictions, _ = eval_fn(
        eval_batch_size=eval_batch_size, parts=["train", "val", "test"]
    )
    report["inference_time"] = str(inference_timer.elapsed())
    report["eval_batch_size"] = eval_batch_size
    report["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    # to free-up space
    if config["seed"] > 0:
        os.remove(output / "checkpoint.pt")

    report["time"] = timer.elapsed()
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.configure_torch()
    lib.run(main)
