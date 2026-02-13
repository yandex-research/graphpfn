import time
from typing import Self

import delu
import dgl
import lib.graphpfn.prior.prior_config
import numpy as np
import torch
from lib.graph.data import GraphDataset
from lib.graphpfn.prior.dataset import PriorDataset
from lib.util import KWArgs
from loguru import logger
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset


def _worker_init_fn(worker_id: int):
    delu.random.seed(worker_id + 1000 * int(time.time()))


def _identity(args):
    return args


class PriorDatasetWrapper(IterableDataset):
    def __init__(
        self,
        *,
        fixed_hp: KWArgs,
        sampled_hp: KWArgs,
        extra_kwargs: KWArgs,
        device: torch.device,
        n_datasets: int,
    ):
        super().__init__()
        self.n_datasets = n_datasets

        scm_fixed_hp = lib.graphpfn.prior.prior_config.DEFAULT_FIXED_HP | fixed_hp
        scm_sampled_hp = lib.graphpfn.prior.prior_config.DEFAULT_SAMPLED_HP | sampled_hp

        self.prior = PriorDataset(
            batch_size=n_datasets,
            scm_fixed_hp=scm_fixed_hp,
            scm_sampled_hp=scm_sampled_hp,
            device=device,
            **extra_kwargs,
        )

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> list[GraphDataset]:
        """Samples list of datsets that share the same graph"""
        # >>> Sample from prior
        graph, x, y, d, seq_lens, train_size = next(self.prior)

        assert x.ndim == 3
        assert y.ndim == 2
        assert d.ndim == 1
        assert train_size.ndim == 1

        assert x.shape[0] == self.n_datasets
        assert y.shape[0] == self.n_datasets
        assert d.shape[0] == self.n_datasets
        assert train_size.shape[0] == self.n_datasets

        return [
            self.prepare_dataset(
                graph=graph,
                x=x[i, ...],
                y=y[i, ...],
                n_features=int(d[i]),
                n_train_samples=int(train_size[i]),
            )
            for i in range(self.n_datasets)
        ]

    def prepare_dataset(
        self,
        graph: dgl.DGLGraph,
        x: Tensor,
        y: Tensor,
        n_features: int,
        n_train_samples: int,
    ) -> GraphDataset:
        assert x.ndim == 2
        assert y.ndim == 1

        features = x[..., :n_features]
        targets = y

        # >>> Transform to desired format
        features = features.cpu().numpy()
        targets = targets.cpu().numpy()

        masks: dict[str, np.ndarray] = dict()
        masks["train"] = np.zeros(features.shape[0], dtype=bool)
        masks["val"] = np.zeros(features.shape[0], dtype=bool)
        masks["test"] = np.zeros(features.shape[0], dtype=bool)

        masks["train"][:n_train_samples] = True
        masks["test"][n_train_samples:] = True

        n_unique_targets = len(np.unique(targets))
        if n_unique_targets > 10:
            task_type = "regression"
        elif n_unique_targets > 2:
            task_type = "multiclass"
        else:
            task_type = "binclass"

        # TODO: are you sure we do not have cat features?
        dataset = GraphDataset.from_data(
            name="GraphPrior",
            graph=graph,
            features={"num_features": features},  # TODO: optimize this
            labels=targets,
            masks=masks,
            task_type=task_type,
        )

        # TODO: refactor
        # TODO: move this dict into params
        # TODO: and do not hard-code this...
        dataset = lib.graph.data.build_dataset(
            path=None,
            dataset=dataset,
            nfa=None,
            setting=dataset.task.setting,
            num_policy="noisy-quantile-normal",
            ratio_policy="noisy-quantile-uniform",
            cat_policy="ordinal",
        )

        # TODO: remove
        self.postprocess_dataset(dataset)
        self.sanity_check(dataset)
        return dataset

    def postprocess_dataset(
        self,
        dataset: GraphDataset,
        transform_targets: bool = True,
    ) -> None:
        if dataset.task.is_regression and transform_targets:
            target_transform = QuantileTransformer(
                n_quantiles=dataset.size(),
                output_distribution="normal",
                subsample=None,  # type: ignore
            )
            dataset.data["labels"] = (
                target_transform.fit_transform(dataset.data["labels"].reshape(-1, 1))
                .squeeze(-1)
                .astype(np.float32)
            )

    @staticmethod
    def sanity_check(dataset: GraphDataset) -> None:
        labels = dataset.data["labels"]

        if np.isnan(labels).any():
            raise ValueError("Bad labels")
        if dataset.task.is_binclass:
            if not np.all((labels == 0.0) | (labels == 1.0)):
                raise ValueError("Bad binclass labels")


class GraphPrior:
    def __init__(
        self,
        *,
        fixed_hp: KWArgs,
        sampled_hp: KWArgs,
        extra_kwargs: KWArgs,
        device: torch.device,
        n_workers: int = 8,
        verbose: bool = True,
    ):
        self.verbose = verbose

        n_datasets = 1

        dataset = PriorDatasetWrapper(
            fixed_hp=fixed_hp,
            sampled_hp=sampled_hp,
            extra_kwargs=extra_kwargs,
            device=device,
            n_datasets=n_datasets,
        )

        self.prior = iter(
            DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=n_workers,
                worker_init_fn=_worker_init_fn,
                multiprocessing_context=(
                    torch.multiprocessing.get_context("spawn")
                    if n_workers > 0
                    else None
                ),
                collate_fn=_identity,
            )
        )

    def __iter__(self):
        return self

    # TODO: batched version
    def __next__(self) -> GraphDataset:
        dataset = next(self.prior)[0][0]
        if self.verbose:
            logger.info(
                f"{dataset.size()=} avg degree={dataset.n_edges / dataset.size()}"
                f" {dataset.n_features=} {dataset.task.type_=}"
            )
        return dataset
