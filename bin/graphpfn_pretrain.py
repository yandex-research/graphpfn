import math
import warnings
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
from loguru import logger
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict  # noqa: UP035

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    PowerTransformer,
    FunctionTransformer,
)

from torch.nn.parallel import DistributedDataParallel

import lib
import lib.graphpfn
from lib import KWArgs, PartKey
from lib.util import Checkpoint
from lib.graph.data import GraphDataset
from lib.graphpfn.util import SimpleEdgeSampler
from lib.graphpfn.model import GraphPFN, GraphPFNOutput
from lib.graphpfn.prior.dataset import PriorDataset as GraphPrior

EvalOut = dict[str, Any]


class Config(TypedDict):
    # Misc
    seed: int
    amp: NotRequired[bool]  # Automatic mixed precision in bfloat16.

    # Learning
    n_steps: int
    n_lr_warmup_epochs: NotRequired[int]
    gradient_clipping_norm: NotRequired[float]
    optimizer: KWArgs
    epoch_size: int
    batch_size: int
    prior: KWArgs
    ssl: NotRequired[KWArgs]

    # Evaluation
    evaluation_data: list[KWArgs]


class GraphPriorWrapper:
    def __init__(
        self,
        device: str,
        structure_features: bool = False,
        nfa: bool = False,
    ):
        scm_fixed_hp = lib.graphpfn.prior.prior_config.DEFAULT_FIXED_HP
        scm_sampled_hp = lib.graphpfn.prior.prior_config.DEFAULT_SAMPLED_HP

        if structure_features:
            scm_sampled_hp |= {
                "structure_features_flag": {
                    "distribution": "meta_choice",
                    "choice_values": [True, False],
                },
            }

        if nfa:
            self.nfa = {"num_modes": ["mean"]}
        else:
            self.nfa = None

        self.prior = GraphPrior(
            scm_fixed_hp=scm_fixed_hp,
            scm_sampled_hp=scm_sampled_hp,
            device=device,
        )

    def try_get_dataset(self) -> GraphDataset:
        # >>> Sample from prior
        graphs, x, y, d, seq_lens, train_size = next(self.prior)
        x = x[..., : d.item()]

        graph = graphs[0].cpu()
        features = x[...].squeeze(0).to()
        targets = y.squeeze(0).to()

        # >>> Transform to desired format
        features = features.cpu().numpy()
        targets = targets.cpu().numpy()

        masks = dict()
        masks["train"] = np.zeros(features.shape[0], dtype=bool)
        masks["val"] = np.zeros(features.shape[0], dtype=bool)
        masks["test"] = np.zeros(features.shape[0], dtype=bool)

        masks["train"][:train_size] = True
        masks["test"][train_size:] = True

        task_type = (
            "multiclass" if len(np.unique(targets)) > 2 else "binclass"
        )

        dataset = GraphDataset.from_data(
            name="GraphPrior",
            graph=graph,
            features={"num_features": features},
            labels=targets,
            masks=masks,
            task_type=task_type,
        )

        policies = {
            "num_policy": "noisy-quantile-normal",
            "ratio_policy": "noisy-quantile-uniform",
            "cat_policy": "ordinal",
        }

        return lib.graph.data.build_dataset(
            path=None,
            dataset=dataset,
            nfa=self.nfa,
            setting=dataset.task.setting,
            **policies,
        )

    def __iter__(self):
        return self

    def __next__(self) -> GraphDataset:
        while True:
            dataset = self.try_get_dataset()
            features = preprocess_tfm_features(dataset)
            if features.shape[-1] >= 4:
                break

        self.prior.prior.current_graph_idx += 1
        return dataset


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


def preprocess_tfm_features(
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
    evaluation_data: list[KWArgs],
    device: str,
    amp_enabled: bool,
    parts: list[str] = ["val", "test"],
) -> EvalOut:
    graphpfn.eval()

    metrics: EvalOut = dict()

    for dataset in evaluation_data:
        if not isinstance(dataset, GraphDataset):
            dataset = lib.graph.data.build_dataset(**dataset)
        name = dataset.data["name"]
        prediction_type = "labels" if dataset.task.is_regression else "probs"

        preprocess_targets_results = preprocess_targets(dataset)
        regression_label_stats = preprocess_targets_results["regression_label_stats"]

        assert dataset.task.is_transductive
        features = preprocess_tfm_features(dataset)

        dataset = dataset.to_torch(device)
        features = torch.tensor(features, device=device)
        y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(
            dtype=torch.float32, device=device
        )

        with torch.autocast(
            device.type,
            enabled=amp_enabled,
            dtype=torch.bfloat16 if amp_enabled else None,
        ):
            out = graphpfn(
                graph=dataset.data["graph"],
                features=features,
                y_train=y_train,
                train_mask=dataset.data["masks"]["train"],
                task_type=dataset.task.type_,
                checkpointing=False,
            )

        predictions: dict[PartKey, np.ndarray] = {}

        for part in parts:
            predictions[part] = (
                out["predictions"][dataset.data["masks"][part], ...].cpu().numpy()
            )

        if regression_label_stats is not None:
            pred_transform = (  # noqa: E731
                lambda tensor: tensor * regression_label_stats.std
                + regression_label_stats.mean
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

        for part in list(predictions.keys()):
            if predictions[part].shape[0] == 0:
                predictions.pop(part)

        dataset_metrics = (
            dataset.task.calculate_metrics(predictions, prediction_type)
            if lib.are_valid_predictions(predictions)
            else {x: {"score": -999999.0} for x in predictions}
        )

        for k, v in dataset_metrics.items():
            metrics[f"{name} ({k})"] = v

    return metrics


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
    if not exp.start(main, force=force, continue_=continue_):
        return None

    warnings.filterwarnings(
        "ignore", message=".*pkg_resources is deprecated as an AP.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated.*",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)

    step = 0
    output = Path(output)
    device = lib.get_device(exp.device_id)
    report = lib.create_report(main, config)
    timer = delu.tools.Timer()

    # >>> model
    logger.info(
        f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    edge_head = config.get("ssl", dict()).get("coef_edge", None) is not None
    graphpfn = GraphPFN(edge_head=edge_head).to(device)

    report["n_parameters"] = lib.deep.get_n_parameters(graphpfn)

    logger.info(f"n_parameters = {report['n_parameters']}")

    graphpfn_without_ddp = graphpfn
    if exp.ddp:
        graphpfn = DistributedDataParallel(
            graphpfn, device_ids=[exp.rank], output_device=exp.rank
        )
        graphpfn_without_ddp = graphpfn.module

    # >>> datasets
    graph_prior = GraphPriorWrapper(device=device, **config["prior"])
    evaluation_data: list[KWArgs | GraphDataset] = config["evaluation_data"]

    # >>> prepare training

    params = lib.deep.make_parameter_groups(graphpfn_without_ddp)

    optimizer = lib.deep.make_optimizer(
        **config["optimizer"],
        params=params,
    )
    gradient_clipping_norm = config.get("gradient_clipping_norm")

    epoch_size = config["epoch_size"]

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

    def prepare_checkpoint(step: int) -> Checkpoint:
        return {
            "run_uid": exp.run_uid if exp.master_process else None,
            "step": step,
            "report": report,
            "timer": timer,
            "model": graphpfn_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "random_state": delu.random.get_state(),
            "extra": {
                "current_graph_idx": graph_prior.prior.prior.current_graph_idx
            },
        }

    def load_from_checkpoint(
        checkpoint: Checkpoint,
    ) -> tuple[int, dict, delu.tools.Timer]:
        graphpfn_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # delu.random.set_state(checkpoint["random_state"])
        logger.warning(
            "Currently not loading random state, as it is saved only for master proc"
        )
        graph_prior.prior.prior.current_graph_idx = (
            checkpoint["extra"]["current_graph_idx"]
        )
        logger.info(
            f"Continuing from step={checkpoint['step']} "
            f"for run_uid={checkpoint['run_uid']}"
        )

        return checkpoint["step"], checkpoint["report"], checkpoint["timer"]

    # >>> eval / step functions

    eval_fn = partial(
        evaluate,
        evaluation_data=evaluation_data,
        device=device,
        amp_enabled=amp_enabled,
        parts=["val", "test"],
    )

    def step_fn():
        dataset = next(graph_prior)
        coef_feat_loss = config.get("ssl", dict()).get("coef_feat", None)
        coef_edge_loss = config.get("ssl", dict()).get("coef_edge", None)

        assert dataset.task.is_transductive
        features = preprocess_tfm_features(dataset)

        dataset = dataset.to_torch(device)
        graph = dataset.data["graph"]
        train_mask = dataset.data["masks"]["train"]
        features = torch.tensor(features, device=device)
        y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(
            dtype=torch.float32, device=device
        )

        features_mean = features[train_mask].mean(-2)
        features_std = features[train_mask].std(-2)
        features = (features - features_mean) / features_std

        # >>> Masking for SSL
        if coef_feat_loss is not None:
            feat_masking_rate = config["ssl"].get("feat_masking_rate", 0.1)
            original_features = features
            features = torch.where(
                torch.rand_like(features) < feat_masking_rate, torch.nan, features
            )

        if coef_edge_loss is not None:
            edge_masking_rate = config["ssl"].get("edge_masking_rate", 0.1)
            edge_sampler = SimpleEdgeSampler(prob=edge_masking_rate)
            negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1)
            mask = edge_sampler(graph)
            edges_pos = graph.find_edges(mask)
            edges_neg = negative_sampler(graph, mask)
            edges = (
                torch.cat([edges_pos[0], edges_neg[0]], dim=-1),
                torch.cat([edges_pos[1], edges_neg[1]], dim=-1),
            )
            edges_labels = torch.cat(
                [
                    torch.ones_like(edges_pos[0]),
                    torch.zeros_like(edges_neg[0]),
                ],
                dim=-1,
            ).float()
            graph.remove_edges(mask)
        else:
            edges = None

        # <<< Masking for SSL

        with torch.autocast(
            device.type,
            enabled=amp_enabled,
            dtype=torch.bfloat16 if amp_enabled else None,
        ):
            out = graphpfn(
                graph=graph,
                features=features,
                y_train=y_train,
                train_mask=train_mask,
                task_type=dataset.task.type_,
                checkpointing=True,
                edges=edges,
            )

        pred = out["predictions"]
        pred = pred[~dataset.data["masks"]["train"]].unsqueeze(0)

        labels = dataset.data["labels"]
        if dataset.task.is_classification:
            sup_loss = F.cross_entropy(
                pred.permute(0, 2, 1),
                labels[~dataset.data["masks"]["train"]].unsqueeze(0).long(),
            )
        else:
            raise NotImplementedError("Currently, regression is not supported")

        loss = sup_loss

        if coef_feat_loss is not None:
            feat_mask = torch.isnan(features)
            feat_loss = F.mse_loss(
                input=out["features_pred"][feat_mask],
                target=original_features[feat_mask],
            )
            loss = loss + coef_feat_loss * feat_loss

        if coef_edge_loss is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                input=out["edge_predictions"].squeeze(-1),
                target=edges_labels,
            )
            loss = loss + coef_edge_loss * edge_loss

        return loss

    # >>> training

    report["train"] = {"metrics": {"val": {"score": -math.inf}}}
    n_steps = config["n_steps"]

    checkpoint = exp.load_checkpoint() if continue_ else None
    if checkpoint is not None:
        step, report, timer = load_from_checkpoint(checkpoint)

    timer.run()

    while n_steps == -1 or step < n_steps:
        logger.info(f"[...] {output} | {timer}")

        # >>>
        graphpfn.train()
        epoch_losses = []
        delu.cuda.free_memory()

        iterator = range(epoch_size)
        if exp.master_process:
            iterator = tqdm(iterator, desc=f"Epoch {step // epoch_size} Step {step}")

        for _ in iterator:
            optimizer.zero_grad()
            try:
                loss = step_fn()
            except RuntimeError as e:
                logger.error(e)
                delu.cuda.free_memory()
                continue
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

        metrics = eval_fn(graphpfn)

        report["train"]["best_step"] = step
        report["train"]["metrics"] = metrics
        report["metrics"] = metrics
        report["lr"] = lr_scheduler.get_last_lr()

        exp.save_checkpoint(prepare_checkpoint(step))

    # >>> finish
    exp.finish()
    return report


if __name__ == "__main__":
    lib.configure_torch(deterministic=False)
    lib.run(main)
