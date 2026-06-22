import math
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, NotRequired

import delu
import dgl
import numpy as np
import scipy
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from typing_extensions import TypedDict

import lib
from lib import KWArgs, PartKey
from lib.graph.data import GraphDataset, TaskType
from lib.graphpfn.model import GraphPFN
from lib.graphpfn.prior import GraphPriorSampler, GraphPriorSamplerDDP
from lib.graphpfn.prior.prior_typings import DistributionConfig
from lib.graphpfn.util import SimpleEdgeSampler
from lib.util import backup, make_seed, random_float, tracker

EvalOut = dict[str, Any]


class EvaluationDataConfig(TypedDict):
    real: list[KWArgs]
    n_synthetic_per_gpu: int


class Config(TypedDict):
    # Misc
    seed: int
    amp: NotRequired[bool]  # Automatic mixed precision in bfloat16.
    tracker: NotRequired[KWArgs]

    # Learning
    base_checkpoint: NotRequired[str | Path]
    n_steps: int
    n_gradient_accumulation_steps: int
    lr_scheduler: NotRequired[KWArgs]
    ema: NotRequired[KWArgs]
    gradient_clipping_norm: NotRequired[float]
    optimizer: KWArgs
    epoch_size: int
    batch_size: int
    prior: KWArgs
    sampler: NotRequired[KWArgs]
    model: NotRequired[KWArgs]
    ssl: NotRequired[KWArgs]

    # Evaluation
    evaluation_data: EvaluationDataConfig


def evaluate_dataset(
    graphpfn: nn.Module,
    dataset: GraphDataset,
    device: torch.device,
    amp_enabled: bool,
    parts: list[str] = ["val", "test"],
) -> dict[str, Any]:
    assert dataset.task.is_transductive
    prediction_type = "labels" if dataset.task.is_regression else "probs"

    regression_label_stats = lib.graph.data.prepare_labels(dataset, True)

    dataset = dataset.to_torch(device)
    features = lib.graph.data.flatten_features(dataset.features)
    assert features is not None
    features = lib.graph.data.drop_constant_features(
        features,
        dataset.data["masks"]["train"],  # type: ignore
    )
    y_train = dataset.data["labels"][dataset.data["masks"]["train"]].to(  # type: ignore
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
            n_random_features=8,  # TODO: do not hardcode
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

    return (
        dataset.task.calculate_metrics(predictions, prediction_type)
        if lib.are_valid_predictions(predictions)
        else {x: {"score": -999999.0} for x in predictions}
    )


def get_real_eval_dataset(config: KWArgs) -> GraphDataset:
    dataset = lib.graph.data.GraphDataset.from_dir(**config["data"])
    features = lib.graph.data.transform_features(
        dataset.features, dataset.task, **config["transform"]["features"]
    )
    features = lib.graph.data.apply_d_reduction(
        features,
        dataset.task,
        **config.get("d_reduction", {}),
    )
    dataset.data.update(features)  # type: ignore
    return dataset


@delu.random.preserve_state()
def get_synthetic_eval_dataset(
    idx: int,
    config: DistributionConfig,
    base_seed: int,
) -> GraphDataset:
    delu.random.seed(make_seed(base_seed, "eval", lib.get_rank(), idx))
    batch = lib.graphpfn.prior.sample_batch(config, 1, torch.device("cpu"))
    datasets = lib.graphpfn.prior.unbatch_prior_dataset(batch)
    assert len(datasets) == 1
    name = f"synthetic-{idx:03d}"
    return lib.graphpfn.prior.convert_to_graph_dataset(datasets[0], name)


def test_synthetic_eval_dataset_determinism(
    n_datasets: int,
    config: DistributionConfig,
    base_seed: int,
) -> None:
    """Checks that synthetic dataset generation for evaluation is deterministic"""
    for idx in range(n_datasets):
        first = get_synthetic_eval_dataset(idx, config, base_seed).data
        second = get_synthetic_eval_dataset(idx, config, base_seed).data

        assert first["name"] == second["name"]
        np.testing.assert_array_equal(first["labels"], second["labels"])
        for key in first["masks"]:
            np.testing.assert_array_equal(first["masks"][key], second["masks"][key])
        for key in lib.graph.data.GRAPH_FEATURE_KEYS:
            if first[key] is None:
                assert second[key] is None
            else:
                np.testing.assert_array_equal(first[key], second[key])

        g1, g2 = first["graph"], second["graph"]
        assert g1.num_nodes() == g2.num_nodes()
        assert g1.num_edges() == g2.num_edges()
        np.testing.assert_array_equal(g1.edges()[0].numpy(), g2.edges()[0].numpy())
        np.testing.assert_array_equal(g1.edges()[1].numpy(), g2.edges()[1].numpy())


@torch.inference_mode()
def evaluate(
    graphpfn: nn.Module,
    evaluation_data_config: EvaluationDataConfig,
    device: torch.device,
    amp_enabled: bool,
    get_synthetic_eval_dataset_fn: Callable[[int], GraphDataset],
    parts: list[str] = ["val", "test"],
) -> EvalOut:
    graphpfn.eval()

    metrics: EvalOut = dict()

    for config in evaluation_data_config["real"]:
        dataset = get_real_eval_dataset(config)
        dataset_metrics = evaluate_dataset(
            graphpfn,
            dataset,
            device=device,
            amp_enabled=amp_enabled,
            parts=parts,
        )
        name = dataset.data["name"]
        for k, v in dataset_metrics.items():
            metrics[f"{name} ({k})"] = v

    synthetic_scores = []
    for idx in range(evaluation_data_config["n_synthetic_per_gpu"]):
        dataset = get_synthetic_eval_dataset_fn(idx)
        dataset_metrics = evaluate_dataset(
            graphpfn,
            dataset,
            device=device,
            amp_enabled=amp_enabled,
            parts=["test"],
        )
        synthetic_scores.append(dataset_metrics["test"]["score"])

    metrics["synthetic"] = {
        "score": lib.allreduce_float(np.mean(synthetic_scores).item())
    }

    return metrics


def main(
    config,
    output: str | Path,
    *,
    force: bool = False,
    continue_: bool = False,
    profiler: torch.profiler.profile | None = None,
    tiny: bool = False,
) -> None | lib.JSONDict:
    warnings.filterwarnings(
        "ignore", message=".*pkg_resources is deprecated as an AP.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated.*",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)

    # >>> start
    if lib.is_ddp():
        lib.configure_ddp(timeout_minutes=30)
    lib.configure_logging(enqueue=True)

    logger.info(f"Launching on {torch.cuda.device_count()} devices")

    config, output = lib.check(config, output, config_type=Config)

    if lib.is_master_process():
        start = lib.start(main, output, force=force, continue_=continue_)
        lib.broadcast_bool(start)
    else:
        start = lib.broadcast_bool()

    if not start:
        return None

    output = Path(output)
    rank = lib.get_rank()
    world_size = lib.get_world_size()

    if tiny:
        config["n_steps"] = 9
        config["n_gradient_accumulation_steps"] = 4
        config["epoch_size"] = 3
        config["evaluation_data"]["real"] = config["evaluation_data"]["real"][:2]
        config["evaluation_data"]["n_synthetic_per_gpu"] = 2

    timer = delu.tools.Timer()
    timer.run()

    # >>> seeding & tracker
    loaded_checkpoint = (
        lib.load_checkpoint(output)
        if continue_ and lib.get_checkpoint_path(output).exists()
        else None
    )

    if lib.is_master_process():
        name = str(output.parent.relative_to(lib.env.get_exp_dir()))
        tracker.init(
            config,  # type: ignore
            name,
            run_uid=loaded_checkpoint["run_uid"] if loaded_checkpoint else None,
            step=loaded_checkpoint["step"] + 1 if loaded_checkpoint else 0,
            **config.get("tracker", {}),
        )
        lib.print_config(config)  # type: ignore
        print()

    resume_step = loaded_checkpoint["step"] if loaded_checkpoint is not None else 0
    delu.random.seed(make_seed(config["seed"], "main", rank, resume_step))
    del loaded_checkpoint

    step = 0
    device = lib.get_device()
    report = lib.create_report(main, config)  # type: ignore

    # >>> model
    logger.info(
        f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    graphpfn = GraphPFN(**config.get("model", dict())).to(device)

    if "base_checkpoint" in config:
        checkpoint = torch.load(config["base_checkpoint"])
        state_dict = {
            k[7:]: v
            for k, v in checkpoint["model_ema"].items()
            if k.startswith("module.")
        }
        assert all([k in graphpfn.state_dict() for k in state_dict.keys()])
        graphpfn.load_state_dict(state_dict, strict=False)

    report["n_parameters"] = lib.deep.get_n_parameters(graphpfn)

    logger.info(f"n_parameters = {report['n_parameters']}")

    graphpfn_without_ddp = graphpfn
    if lib.is_ddp():
        graphpfn = DistributedDataParallel(
            graphpfn,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

    if "ema" in config:
        multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(
            decay=config["ema"]["decay"]
        )
        graphpfn_ema = torch.optim.swa_utils.AveragedModel(
            graphpfn_without_ddp,
            device,
            multi_avg_fn=multi_avg_fn,
        )
    else:
        graphpfn_ema = graphpfn_without_ddp

    # >>> datasets
    sampler_cls = GraphPriorSamplerDDP if lib.is_ddp() else GraphPriorSampler
    graph_prior = sampler_cls(
        config=config["prior"],
        seed=make_seed(config["seed"], "sampler", resume_step),
        **config.get("sampler", {}),
    )
    test_synthetic_eval_dataset_determinism(
        config["evaluation_data"]["n_synthetic_per_gpu"],
        config["prior"],
        config["seed"],
    )

    # >>> prepare training

    params = lib.deep.make_parameter_groups(graphpfn_without_ddp)

    optimizer = lib.deep.make_optimizer(
        **config["optimizer"],
        params=params,
    )
    gradient_clipping_norm = config.get("gradient_clipping_norm")

    epoch_size = config["epoch_size"]

    if "lr_scheduler" in config:
        lr_scheduler = lib.deep.get_lr_scheduler(
            optimizer=optimizer,
            n_steps=config["n_steps"],
            **config["lr_scheduler"],
        )
    else:
        lr_scheduler = None

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    logger.info(f"AMP enabled: {amp_enabled}")

    def prepare_checkpoint(step: int) -> dict:
        return {
            "run_uid": tracker.get_uid(),
            "step": step,
            "report": report,
            "timer": timer,
            "model": graphpfn_without_ddp.state_dict(),
            "model_ema": graphpfn_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
            if (lr_scheduler is not None)
            else None,
            "random_state": delu.random.get_state(),
            "extra": {},
        }

    def save_checkpoint(checkpoint: dict) -> None:
        lib.barrier()
        if lib.is_master_process():
            info = {
                k: v["score"] for k, v in checkpoint["report"]["metrics"].items()
            } | {
                "lr": (
                    checkpoint["report"]["lr"][0]
                    if "lr" in checkpoint["report"]
                    else None
                ),
                "loss": (
                    checkpoint["report"]["loss"]
                    if "loss" in checkpoint["report"]
                    else None
                ),
            }
            logger.info(f"{info=}")
            tracker.log(info, step=checkpoint["step"])
            lib.dump_checkpoint(output, checkpoint)
            backup(output)
        lib.barrier()

    def load_from_checkpoint(
        checkpoint: dict,
    ) -> tuple[int, dict, delu.tools.Timer]:
        graphpfn_without_ddp.load_state_dict(checkpoint["model"])
        graphpfn_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            assert checkpoint["lr_scheduler"] is not None
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # delu.random.set_state(checkpoint["random_state"])
        logger.warning(
            "Currently not loading random state, as it is saved only for master proc"
        )
        logger.info(
            f"Continuing from step={checkpoint['step']} "
            f"for run_uid={checkpoint['run_uid']}"
        )

        return checkpoint["step"], checkpoint["report"], checkpoint["timer"]

    # >>> eval / step functions

    eval_fn = partial(
        evaluate,
        evaluation_data_config=config["evaluation_data"],
        device=device,
        amp_enabled=amp_enabled,
        get_synthetic_eval_dataset_fn=partial(
            get_synthetic_eval_dataset,
            config=config["prior"],
            base_seed=config["seed"],
        ),
        parts=["val", "test"],
    )

    # TODO: move this outside of main?
    def step_fn():
        batch = next(graph_prior)
        ssl_feat_config = config.get("ssl", dict()).get("feat", None)
        ssl_edge_config = config.get("ssl", dict()).get("edge", None)

        n_nodes = int(batch["n_nodes"][0].item())
        n_features = int(batch["n_features"][0].item())
        n_edges = int(batch["n_edges"][0].item())
        n_train_nodes = batch["n_train_nodes"]
        task_type = batch["task_type"]

        edges = batch["edges"]
        graph = dgl.graph(
            (edges[0, 0, :n_edges], edges[0, 1, :n_edges]),
            num_nodes=n_nodes,
            device=device,
        )
        features = batch["features"][0, :n_nodes, :n_features].to(device)
        labels = batch["labels"][0, :n_nodes].to(device)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[:n_train_nodes] = True
        y_train = labels[train_mask].to(dtype=torch.float32)

        features_mean = features[train_mask].mean(-2)
        features_std = features[train_mask].std(-2)
        features = (features - features_mean) / features_std

        # >>> Masking for SSL
        if ssl_feat_config is not None:
            feat_masking_rate = random_float(
                a=ssl_feat_config.get("min_mask_rate", 0.1),
                b=ssl_feat_config.get("max_mask_rate", 0.1),
            )
            original_features = features
            feat_mask = torch.rand_like(features) < feat_masking_rate

            ssl_feat_mask_type = ssl_feat_config.get("mask", "permute")
            if ssl_feat_mask_type == "permute":
                assert features.ndim == 2
                permutation = torch.randint(
                    low=0,
                    high=n_nodes,
                    size=features.shape,
                    device=features.device,
                )
                features_masked = torch.gather(input=features, dim=0, index=permutation)
            elif ssl_feat_mask_type == "nan":
                features_masked = torch.nan
            else:
                raise ValueError(f"Unknown {ssl_feat_mask_type=}")

            features = torch.where(feat_mask, features_masked, features)

            ssl_feat_predict_type = ssl_feat_config.get("predict", "all")
            if ssl_feat_predict_type == "all":
                feat_mask = torch.ones_like(feat_mask)
            elif ssl_feat_predict_type != "masked":
                raise ValueError(f"Unknown {ssl_feat_predict_type=}")

        if ssl_edge_config is not None:
            edge_masking_rate = random_float(
                a=ssl_edge_config.get("min_mask_rate", 0.1),
                b=ssl_edge_config.get("max_mask_rate", 0.1),
            )
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
                task_type=task_type,
                edges=edges,
                n_random_features=0,
            )

        pred = out["predictions"]
        pred = pred[~train_mask].unsqueeze(0)

        is_classification = task_type in [TaskType.BINCLASS, TaskType.MULTICLASS]
        if is_classification:
            sup_loss = F.cross_entropy(
                pred.permute(0, 2, 1),
                labels[~train_mask].unsqueeze(0).long(),
            )
        else:
            loss_fn = F.mse_loss
            sup_loss = loss_fn(
                pred,
                labels[~train_mask].unsqueeze(0),
            ).mean()

        loss = sup_loss

        if ssl_feat_config is not None:
            feat_loss = F.mse_loss(
                input=out["features_pred"][feat_mask],  # type: ignore
                target=original_features[feat_mask],  # type: ignore
            )
            baseline_loss = F.mse_loss(
                input=torch.zeros_like(out["features_pred"][feat_mask]),  # type: ignore
                target=original_features[feat_mask],  # type: ignore
            )
            logger.info(f"{feat_loss.item()=} {baseline_loss.item()=}")
            loss = loss + ssl_feat_config["coef"] * feat_loss

        if ssl_edge_config is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                input=out["edge_predictions"].squeeze(-1),
                target=edges_labels,  # type: ignore
            )
            loss = loss + ssl_edge_config["coef"] * edge_loss

        return loss

    # >>> training

    report["train"] = {"metrics": {"val": {"score": -math.inf}}}
    n_steps = config["n_steps"]

    lib.barrier()
    checkpoint = (
        lib.load_checkpoint(output)
        if continue_ and lib.get_checkpoint_path(output).exists()
        else None
    )
    is_resuming = checkpoint is not None
    lib.barrier()
    if checkpoint is not None:
        step, report, timer = load_from_checkpoint(checkpoint)
        del checkpoint

    timer.run()

    if not is_resuming:
        delu.cuda.free_memory()
        metrics = eval_fn(graphpfn_ema)
        report["train"]["best_step"] = step  # type: ignore
        report["train"]["metrics"] = metrics
        report["metrics"] = metrics
        report["lr"] = (
            lr_scheduler.get_last_lr()
            if lr_scheduler is not None
            else config["optimizer"]["lr"]
        )
        report["loss"] = float("nan")
        save_checkpoint(prepare_checkpoint(step))

    while n_steps == -1 or step < n_steps:
        logger.info(f"[...] {output} | {timer}")

        # >>>
        graphpfn.train()
        epoch_losses = []
        delu.cuda.free_memory()

        iterator = range(epoch_size)
        if lib.is_master_process():
            iterator = tqdm(iterator, desc=f"Epoch {step // epoch_size} Step {step}")

        for _ in iterator:
            step_loss = []
            for inner_step in range(config["n_gradient_accumulation_steps"]):
                is_gradient_step = (inner_step + 1) == config[
                    "n_gradient_accumulation_steps"
                ]

                with (
                    graphpfn.no_sync()
                    if lib.is_ddp() and not is_gradient_step
                    else nullcontext()
                ):
                    loss = step_fn()
                    (loss / config["n_gradient_accumulation_steps"]).backward()
                    step_loss.append(loss.detach())

                if profiler is not None:
                    profiler.step()

            if gradient_clipping_norm is not None:
                nn.utils.clip_grad.clip_grad_norm_(
                    graphpfn.parameters(), gradient_clipping_norm
                )
            optimizer.step()
            if isinstance(graphpfn_ema, torch.optim.swa_utils.AveragedModel):
                graphpfn_ema.update_parameters(graphpfn_without_ddp)

            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            step += 1
            epoch_losses.append(torch.mean(torch.stack(step_loss)))

        epoch_loss_mean = torch.stack(epoch_losses).mean()
        if lib.is_ddp():
            torch.distributed.all_reduce(epoch_loss_mean)
            epoch_loss_mean = epoch_loss_mean / world_size

        delu.cuda.free_memory()
        metrics = eval_fn(graphpfn_ema)

        report["train"]["best_step"] = step  # type: ignore
        report["train"]["metrics"] = metrics
        report["metrics"] = metrics
        report["lr"] = (
            lr_scheduler.get_last_lr()
            if lr_scheduler is not None
            else config["optimizer"]["lr"]
        )
        report["loss"] = epoch_loss_mean.item()

        save_checkpoint(prepare_checkpoint(step))

    # >>> finish
    lib.barrier()
    report["time"] = timer.elapsed()
    if lib.is_master_process():
        lib.finish(output, report)
    if lib.is_ddp():
        torch.distributed.destroy_process_group()
    return report


if __name__ == "__main__":
    lib.configure_torch(deterministic=False)
    lib.run(main)
