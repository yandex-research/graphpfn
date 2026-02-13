import math
import os
import statistics
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import delu
import dgl
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import FunctionTransformer
from torch import Tensor, nn
from tqdm.auto import tqdm
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
from lib.graphpfn.model import GraphPFN
from lib.util import Checkpoint

EvalOut = tuple[dict[str, Any], dict[PartKey, np.ndarray]]


class EnsembleConfig(TypedDict):
    n_members: int
    shuffle_features: bool
    shuffle_targets: bool
    max_columns: NotRequired[int]


class Config(TypedDict):
    seed: int
    amp: NotRequired[bool]  # Automatic mixed precision in bfloat16.
    data: KWArgs

    model: KWArgs
    checkpoint_name: str | Path
    unfreeze_all: NotRequired[bool]

    n_steps: int
    epoch_size: int
    patience: int
    n_lr_warmup_epochs: NotRequired[int]

    optimizer: KWArgs
    seq_len_pred: int
    min_train_ratio: float

    ensemble: EnsembleConfig


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
) -> dict:
    if dataset.task.is_regression:
        dataset.data["labels"], regression_label_stats = lib.data.standardize_labels(
            dataset.data["labels"], dataset.data["masks"]
        )
        # Can be replaced with something else
        target_transform = FunctionTransformer(func=None)
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


def preprocess_features(
    dataset: GraphDataset, add_ratio_features_to="num_features"
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


@torch.inference_mode()
def evaluate(
    graphpfn: GraphPFN,
    dataset: GraphDataset,
    graph: dgl.DGLGraph,
    features: Tensor,
    y_train: Tensor,
    prediction_type: str,
    regression_label_stats,
    device: torch.device,
    amp_enabled: bool,
    ensemble: Ensemble,
    parts: list[PartKey] = ["val", "test"],
) -> EvalOut:
    graphpfn.eval()

    ensemble_member_predictions: list[Tensor] = []

    for member in tqdm(ensemble):
        features_transformed = apply_feature_transform(features, member)
        y_train_transformed = apply_target_transform(y_train, member)

        with torch.autocast(
            device.type,
            enabled=amp_enabled,
            dtype=torch.bfloat16 if amp_enabled else None,
        ):
            out = graphpfn(
                graph=graph,
                features=features_transformed,
                y_train=y_train_transformed,
                train_mask=dataset.data["masks"]["train"],
                task_type=dataset.task.type_,
                checkpointing=False,
                batched_attn=(dataset.size() >= 2**16 - 1),
            )

        member_pred = out["predictions"]
        member_pred = reverse_target_perm(member_pred, member)
        ensemble_member_predictions.append(member_pred)

    aggregated = ensemble.aggregate_predictions(
        ensemble_member_predictions, dataset.task.type_
    )

    predictions: dict[PartKey, np.ndarray] = {}
    for part in parts:
        part_pred = aggregated[dataset.data["masks"][part], ...].cpu().numpy()
        if part_pred.shape[0] == 0:
            continue
        predictions[part] = part_pred

    if regression_label_stats is not None:
        pred_transform = (  # noqa: E731
            lambda tensor: (
                tensor * regression_label_stats.std + regression_label_stats.mean
            )
        )
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
        dataset.task.calculate_metrics(predictions, prediction_type)
        if lib.are_valid_predictions(predictions)
        else {x: {"score": -999999.0} for x in predictions}
    )

    return metrics, predictions


def compute_loss(
    graphpfn: GraphPFN,
    dataset: GraphDataset,
    graph: dgl.DGLGraph,
    features: Tensor,
    y_train: Tensor,
    context_mask: Tensor,
    batch_mask: Tensor,
    device: torch.device,
    amp_enabled: bool,
    ensemble: Ensemble,
) -> Tensor:
    graphpfn.train()

    _, member = ensemble.sample_member()
    features_transformed = apply_feature_transform(features, member)
    y_train_transformed = apply_target_transform(y_train, member)

    with torch.autocast(
        device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
    ):
        train_mask = features_transformed.new_zeros([dataset.size()], dtype=bool)
        train_idx = torch.where(dataset.data["masks"]["train"])[0]
        train_mask[train_idx[context_mask]] = True

        out = graphpfn(
            graph=graph,
            features=features_transformed,
            y_train=y_train_transformed[context_mask],
            train_mask=train_mask,
            task_type=dataset.task.type_,
            checkpointing=True,
            batched_attn=(dataset.size() >= 2**16 - 1),
        )

    preds = out["predictions"][dataset.data["masks"]["train"]][batch_mask]
    if dataset.task.is_regression:
        loss = F.mse_loss(input=preds, target=y_train_transformed[batch_mask])
    else:
        loss = F.cross_entropy(
            input=preds, target=y_train_transformed[batch_mask].long()
        )

    return loss


def main(
    config,
    output: str | Path,
    *,
    force: bool = False,
    continue_: bool = False,
) -> None | lib.JSONDict:
    # >>> start
    exp = lib.Experiment()
    config, output = exp.check(config, output, config_type=Config)
    if not exp.start(main, force=force, continue_=continue_, exp_tracker=False):
        return None

    # TODO: clean up this?
    warnings.filterwarnings(
        "ignore", message=".*pkg_resources is deprecated as an AP.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated.*",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)

    output = Path(output)
    device = lib.get_device(exp.device_id)  # TODO: refactor to exp.device
    report = lib.create_report(main, config)
    timer = delu.tools.Timer()

    assert not exp.ddp, "You do not need >1 GPU for inference..."

    # For memory footprint benchmarking
    delu.cuda.free_memory()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # >>> dataset
    dataset_timer = delu.tools.Timer()
    dataset_timer.run()

    dataset = lib.graph.data.build_dataset(**config["data"])
    assert dataset.task.is_transductive

    prediction_type = "labels" if dataset.task.is_regression else "probs"
    preprocess_targets_results = preprocess_targets(dataset)
    regression_label_stats = preprocess_targets_results["regression_label_stats"]
    features = preprocess_features(dataset)

    dataset = dataset.to_torch(device)
    features = torch.tensor(features, device=device)
    y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(
        dtype=torch.float32, device=device
    )

    ensemble_config = config["ensemble"]
    ensemble = create_ensemble(
        n_members=ensemble_config["n_members"],
        n_features=features.shape[1],
        n_classes=dataset.task.try_compute_n_classes(),
        shuffle_features=ensemble_config["shuffle_features"],
        shuffle_targets=ensemble_config["shuffle_targets"],
        max_columns=ensemble_config.get("max_columns"),
        device=device,
        seed=config["seed"],
    )

    report["dataset_time"] = str(dataset_timer.elapsed())

    # >>> model
    # TODO: double-check edge head
    if config["checkpoint_name"] != "none":
        checkpoint = torch.load(
            Path("checkpoints") / f"{config['checkpoint_name']}.ckpt"
        )
        if "model_ema" not in checkpoint:
            logger.error("Loading model instead of model_ema")
            state_dict = checkpoint["model"]
        else:
            state_dict = {
                k[7:]: v
                for k, v in checkpoint["model_ema"].items()
                if k.startswith("module.")
            }
    else:
        checkpoint = None
        state_dict = None

    graphpfn = GraphPFN(**config.get("model", dict())).to(device)
    if state_dict is not None:
        graphpfn.load_state_dict(state_dict, strict=False)

    if config.get("unfreeze_all", True):
        for param in graphpfn.parameters():
            param.requires_grad = True

    # >>> prepare for training

    params = lib.deep.make_parameter_groups(graphpfn)

    optimizer = lib.deep.make_optimizer(
        **config["optimizer"],
        params=params,
    )
    gradient_clipping_norm = config.get("gradient_clipping_norm")

    epoch_size = config["epoch_size"]

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

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    logger.info(f"AMP enabled: {amp_enabled}")

    step_fn = partial(
        compute_loss,
        dataset=dataset,
        graph=dataset.data["graph"],
        features=features,
        device=device,
        amp_enabled=amp_enabled,
        ensemble=ensemble,
    )

    eval_fn = partial(
        evaluate,
        dataset=dataset,
        graph=dataset.data["graph"],
        features=features,
        y_train=y_train,
        prediction_type=prediction_type,
        regression_label_stats=regression_label_stats,
        device=device,
        amp_enabled=amp_enabled,
        ensemble=ensemble,
    )

    def prepare_checkpoint(step: int) -> Checkpoint:
        return {
            "run_uid": exp.run_uid if exp.master_process else None,
            "step": step,
            "report": report,
            "timer": timer,
            "model": graphpfn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "random_state": delu.random.get_state(),
            "model_ema": dict(),
            "extra": dict(),
        }

    timer = delu.tools.Timer()

    # >>> train loop

    timer.run()

    step = 0
    early_stopping = delu.tools.EarlyStopping(config["patience"], mode="max")
    report["train"] = {"metrics": {"val": {"score": -math.inf}}}

    if config["n_steps"] == 0:
        exp.save_checkpoint(prepare_checkpoint(step))

    while config["n_steps"] == -1 or step < config["n_steps"]:
        print(f"[...] {output} | {timer}")

        # >>>
        graphpfn.train()
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
            batch_idx = next(idx_queue)
            batch_mask = batch_idx.new_zeros([dataset.size("train")], dtype=torch.bool)
            batch_mask[batch_idx] = True
            context_mask = ~batch_mask

            optimizer.zero_grad()
            loss = step_fn(
                graphpfn=graphpfn,
                y_train=y_train,
                context_mask=context_mask,
                batch_mask=batch_mask,
            )

            loss.backward()

            if gradient_clipping_norm is not None:
                nn.utils.clip_grad.clip_grad_norm_(
                    graphpfn.parameters(), gradient_clipping_norm
                )
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            step += 1
            epoch_losses.append(loss.detach())

        epoch_losses = torch.stack(epoch_losses).tolist()
        mean_loss = statistics.mean(epoch_losses)

        metrics, predictions = eval_fn(graphpfn)

        lib.print_metrics(mean_loss, metrics)

        if metrics["val"]["score"] > report["train"]["metrics"]["val"]["score"]:
            print("🌸 New best epoch! 🌸")
            report["train"]["best_step"] = step
            report["train"]["metrics"] = metrics
            exp.save_checkpoint(prepare_checkpoint(step))

        early_stopping.update(metrics["val"]["score"])
        if early_stopping.should_stop() or not lib.are_valid_predictions(predictions):
            break

        print()

    report["time"] = timer.elapsed()

    # >>> finish
    graphpfn.load_state_dict(exp.load_checkpoint()["model"])
    logger.info("Final Eval")
    inference_timer = delu.tools.Timer()
    inference_timer.run()
    report["metrics"], _ = eval_fn(graphpfn)
    report["inference_time"] = str(inference_timer.elapsed())
    report["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    # to free-up space
    if config["seed"] > 0:
        os.remove(output / "checkpoint.ckpt")
    exp.finish()
    lib.dump_summary(output, lib.summarize(report))
    lib.print_summary(output)
    return report


if __name__ == "__main__":
    lib.configure_torch()
    lib.run(main)
