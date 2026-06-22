"""GraphPriorSampler: iterator yielding batches of synthetic graph datasets."""

from collections.abc import Iterator
from functools import partial
from typing import Self

import delu
import torch
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset

from lib.graphpfn.prior import priors
from lib.graphpfn.prior.checks import SanityCheckError, check_dataset
from lib.graphpfn.prior.config import sample_configs
from lib.graphpfn.prior.prior_typings import (
    DistributionConfig,
    PriorDataset,
    PriorDatasetBatch,
    SampledConfig,
)
from lib.util import (
    TaskType,
    configure_logging,
    get_device,
    get_world_size,
    is_master_process,
)

TASK_TYPE_CODES = {TaskType.BINCLASS: 0, TaskType.MULTICLASS: 1, TaskType.REGRESSION: 2}
TASK_TYPE_CODES_INV = {v: k for k, v in TASK_TYPE_CODES.items()}

# >>> Dataset sampling


def _sample_dataset_with_retry(
    config: SampledConfig,
    max_retries: int = 3,
) -> PriorDataset:
    min_features = config["sanity_check"]["min_features"]
    task_config = config["prior"]["task"]
    n_classes = (
        task_config["n_classes"] if task_config["_type_"] == "multiclass" else None
    )

    for attempt in range(max_retries):
        try:
            dataset = priors.sample_dataset(config["prior"])

            check_dataset(
                features=dataset["features"],
                labels=dataset["labels"],
                n_train_nodes=dataset["n_train_nodes"],
                task_type=dataset["task_type"],
                min_features=min_features,
                n_classes=n_classes,
            )

            return dataset

        except SanityCheckError as e:
            logger.warning(f"[Attempt {attempt + 1}/{max_retries}] {e} {config}")
            if attempt == max_retries - 1:
                raise e

    raise SanityCheckError(f"Failed after {max_retries} attempts")


# >>> Batch assembly


def _pad_and_batch(datasets: list[PriorDataset]) -> PriorDatasetBatch:
    batch_size = len(datasets)

    node_counts = [d["features"].shape[0] for d in datasets]
    feature_counts = [d["features"].shape[1] for d in datasets]
    edge_counts = [d["edges"].shape[1] for d in datasets]

    max_nodes = max(node_counts)
    max_features = max(feature_counts)
    max_edges = max(edge_counts)

    features = torch.zeros(batch_size, max_nodes, max_features, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_nodes, dtype=torch.float32)
    edges = torch.zeros(batch_size, 2, max_edges, dtype=torch.int64)

    task_type = datasets[0]["task_type"]
    n_train_nodes = datasets[0]["n_train_nodes"]

    for i, d in enumerate(datasets):
        n, f = d["features"].shape
        e = d["edges"].shape[1]
        features[i, :n, :f] = d["features"]
        labels[i, :n] = d["labels"]
        edges[i, :, :e] = d["edges"]

        assert d["n_train_nodes"] == n_train_nodes
        assert d["task_type"] == task_type

    return PriorDatasetBatch(
        features=features,
        labels=labels,
        edges=edges,
        n_nodes=torch.tensor(node_counts, dtype=torch.int64),
        n_features=torch.tensor(feature_counts, dtype=torch.int64),
        n_edges=torch.tensor(edge_counts, dtype=torch.int64),
        n_train_nodes=n_train_nodes,
        task_type=task_type,
    )


# >>> Public functional API


def sample_batch(
    base_config: DistributionConfig,
    batch_size: int,
    device: torch.device,
) -> PriorDatasetBatch:
    """Sample a batch of datasets and move to device.

    This is the main functional API for sampling. Use this directly when you
    need a single batch, or use GraphPriorSampler for an infinite iterator.

    Args:
        base_config: Distribution config (may contain _distribution_ specs).
        batch_size: Number of datasets in the batch.
        device: Target device for tensors.

    Returns:
        PriorDatasetBatch with tensors on the specified device.
    """

    while True:
        try:
            configs = sample_configs(base_config, batch_size)
            datasets = [_sample_dataset_with_retry(config) for config in configs]
            batch = _pad_and_batch(datasets)
        except SanityCheckError:
            continue

        batch["features"] = batch["features"].to(device)
        batch["labels"] = batch["labels"].to(device)
        batch["n_nodes"] = batch["n_nodes"].to(device)
        batch["n_features"] = batch["n_features"].to(device)

        return batch


# >>> Public iterator interface


class _BatchIterableDataset(IterableDataset[PriorDatasetBatch]):
    def __init__(
        self,
        config: DistributionConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> PriorDatasetBatch:
        return sample_batch(self.config, self.batch_size, self.device)


def _identity(x: list[PriorDatasetBatch]) -> PriorDatasetBatch:
    return x[0]


def _worker_init(worker_id: int, seed: int) -> None:
    configure_logging(enqueue=True)
    delu.random.seed(seed + worker_id)


class GraphPriorSampler:
    """
    Iterator yielding batches of synthetic graph datasets.

    Usage:
        config = get_default_config()
        sampler = GraphPriorSampler(config, batch_size=16, device=device)
        for batch in sampler:
            train_on_batch(batch)

    With multiprocessing:
        sampler = GraphPriorSampler(
            config, batch_size=16, device=device,
            seed=42, n_workers=4
        )
    """

    def __init__(
        self,
        config: DistributionConfig,
        batch_size: int,
        seed: int | None = None,
        n_workers: int = 0,
        prefetch_factor: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            config: Distribution config for sampling.
            batch_size: Number of datasets per batch.
            device: Target device for tensors.
            seed: Random seed. Required if n_workers > 0.
            n_workers: Number of worker processes (0 = main process only).
            verbose: Whether to log batch info.

        Raises:
            ValueError: if n_workers > 0 and seed is None.
        """
        if n_workers > 0 and seed is None:
            raise ValueError("seed is required when n_workers > 0")

        self.verbose = verbose
        self._iterator: Iterator[PriorDatasetBatch]

        dataset = _BatchIterableDataset(config, batch_size, torch.device("cpu"))

        if n_workers == 0:
            self._iterator = iter(dataset)
        else:
            assert seed is not None
            loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=n_workers,
                worker_init_fn=partial(_worker_init, seed=seed),
                multiprocessing_context=torch.multiprocessing.get_context("spawn"),
                collate_fn=_identity,
                prefetch_factor=prefetch_factor,
            )
            self._iterator = iter(loader)

    def __iter__(self) -> Self:
        return self

    def log_stats(self, batch: PriorDatasetBatch) -> None:
        n_nodes_avg = batch["n_nodes"].float().mean().item()
        n_features_avg = batch["n_features"].float().mean().item()
        n_edges_avg = batch["n_edges"].float().mean().item()
        logger.info(
            f"batch: n_nodes={n_nodes_avg:.0f}, "
            f"n_edges={n_edges_avg:.0f}, "
            f"n_features={n_features_avg:.0f}, "
            f"n_train_nodes={batch['n_train_nodes']}"
        )

    def __next__(self) -> PriorDatasetBatch:
        batch = next(self._iterator)
        if self.verbose:
            self.log_stats(batch)
        return batch


class GraphPriorSamplerDDP:
    """
    DDP wrapper for GraphPriorSampler
    """

    def __init__(
        self,
        config: DistributionConfig,
        batch_size: int,
        seed: int | None = None,
        n_workers: int = 0,
        prefetch_factor: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.verbose = verbose

        if is_master_process():
            world_size = get_world_size()
            self.prior = GraphPriorSampler(
                config=config,
                batch_size=world_size * batch_size,
                seed=seed,
                n_workers=world_size * n_workers,
                prefetch_factor=prefetch_factor,
                verbose=False,
            )
        else:
            self.prior = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> PriorDatasetBatch:
        device = get_device()
        local_batch_size = self.batch_size

        if self.prior is not None:
            # >>> Sample
            global_batch = next(self.prior)
            for k, v in global_batch.items():
                if isinstance(v, torch.Tensor):
                    global_batch[k] = v.to(device)
            # >>> Step 1: metadata
            global_batch_size, max_nodes, max_features = global_batch["features"].shape
            max_edges = global_batch["edges"].shape[-1]
            n_train_nodes = global_batch["n_train_nodes"]
            task_type_code = TASK_TYPE_CODES[global_batch["task_type"]]
            metadata = torch.tensor(
                [
                    global_batch_size,
                    max_nodes,
                    max_features,
                    max_edges,
                    n_train_nodes,
                    task_type_code,
                ],
                dtype=torch.int64,
                device=device,
            )
            torch.distributed.broadcast(metadata, src=0)  # type: ignore
            # >>> Step 2: allocate
            features = torch.empty(
                [local_batch_size, max_nodes, max_features],
                dtype=torch.float32,
                device=device,
            )
            labels = torch.empty(
                [local_batch_size, max_nodes],
                dtype=torch.float32,
                device=device,
            )
            edges = torch.empty(
                [local_batch_size, 2, max_edges],
                dtype=torch.int64,
                device=device,
            )
            n_nodes = torch.empty(
                [local_batch_size],
                dtype=torch.int64,
                device=device,
            )
            n_features = torch.empty(
                [local_batch_size],
                dtype=torch.int64,
                device=device,
            )
            n_edges = torch.empty(
                [local_batch_size],
                dtype=torch.int64,
                device=device,
            )
            # >>> Step 3: scatter
            features_list = global_batch["features"].split(self.batch_size, dim=0)
            labels_list = global_batch["labels"].split(self.batch_size, dim=0)
            edges_list = global_batch["edges"].split(self.batch_size, dim=0)
            n_nodes_list = global_batch["n_nodes"].split(self.batch_size, dim=0)
            n_features_list = global_batch["n_features"].split(self.batch_size, dim=0)
            n_edges_list = global_batch["n_edges"].split(self.batch_size, dim=0)
            torch.distributed.scatter(features, list(features_list), src=0)  # type: ignore
            torch.distributed.scatter(labels, list(labels_list), src=0)  # type: ignore
            torch.distributed.scatter(edges, list(edges_list), src=0)  # type: ignore
            torch.distributed.scatter(n_nodes, list(n_nodes_list), src=0)  # type: ignore
            torch.distributed.scatter(n_features, list(n_features_list), src=0)  # type: ignore
            torch.distributed.scatter(n_edges, list(n_edges_list), src=0)  # type: ignore
        else:
            # >>> Step 1: metadata
            metadata = torch.tensor([0] * 6, dtype=torch.int64, device=device)
            torch.distributed.broadcast(metadata, src=0)  # type: ignore
            (
                global_batch_size,
                max_nodes,
                max_features,
                max_edges,
                n_train_nodes,
                task_type_code,
            ) = metadata.cpu().tolist()
            # >>> Step 2: allocate
            features = torch.empty(
                [local_batch_size, max_nodes, max_features],
                dtype=torch.float32,
                device=device,
            )
            labels = torch.empty(
                [local_batch_size, max_nodes],
                dtype=torch.float32,
                device=device,
            )
            edges = torch.empty(
                [local_batch_size, 2, max_edges],
                dtype=torch.int64,
                device=device,
            )
            n_nodes = torch.empty(
                [local_batch_size],
                dtype=torch.int64,
                device=device,
            )
            n_features = torch.empty(
                [local_batch_size],
                dtype=torch.int64,
                device=device,
            )
            n_edges = torch.empty(
                [local_batch_size],
                dtype=torch.int64,
                device=device,
            )
            # >>> Step 3: scatter
            torch.distributed.scatter(features, None, src=0)  # type: ignore
            torch.distributed.scatter(labels, None, src=0)  # type: ignore
            torch.distributed.scatter(edges, None, src=0)  # type: ignore
            torch.distributed.scatter(n_nodes, None, src=0)  # type: ignore
            torch.distributed.scatter(n_features, None, src=0)  # type: ignore
            torch.distributed.scatter(n_edges, None, src=0)  # type: ignore

        # >>> Step 4: collect & return
        local_batch: PriorDatasetBatch = {
            "features": features,
            "labels": labels,
            "edges": edges,
            "n_nodes": n_nodes,
            "n_features": n_features,
            "n_edges": n_edges,
            "n_train_nodes": n_train_nodes,
            "task_type": TASK_TYPE_CODES_INV[task_type_code],
        }

        if self.verbose and self.prior is not None:
            self.prior.log_stats(local_batch)

        return local_batch
