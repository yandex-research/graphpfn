from typing import Protocol

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch import Tensor

from graphpfn.data import GraphDataset
from graphpfn.inference.preprocessing import RegressionTargetStats
from graphpfn.model.graphpfn import GraphPFN
from graphpfn.util import TaskType


class ScoreFn(Protocol):
    def __call__(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float: ...


class LossFn(Protocol):
    def __call__(
        self,
        *,
        y_true: Tensor,
        y_pred: Tensor,
    ) -> Tensor: ...


def apply_model(
    *,
    model: GraphPFN,
    dataset: GraphDataset,
    graph: dgl.DGLGraph,
    features: Tensor,
    y_train: Tensor,
    train_mask: Tensor,
    regression_target_stats: RegressionTargetStats | None,
    amp: bool,
    device: torch.device,
) -> np.ndarray:
    with torch.autocast(
        device.type,
        enabled=amp,
        dtype=torch.bfloat16 if amp else None,
    ):
        preds = model(
            graph=graph,
            features=features,
            y_train=y_train,
            train_mask=train_mask,
            task_type=dataset.task_type,
        )

    if dataset.task_type == "binclass":
        preds = preds[:, 1]
    elif dataset.task_type == "multiclass":
        n_classes = dataset.n_classes
        assert n_classes is not None
        preds = preds[:, :n_classes]
    elif dataset.task_type == "regression":
        assert regression_target_stats is not None
        preds = regression_target_stats.mean + regression_target_stats.std * preds
    else:
        raise ValueError(f"Unknown TaskType={dataset.task_type}")

    return preds.detach().cpu().numpy()


def compute_loss(
    *,
    y_true: Tensor,
    y_pred: Tensor,
    task_type: TaskType,
) -> Tensor:
    if task_type == "regression":
        return F.mse_loss(y_pred, y_true)
    else:
        return F.cross_entropy(y_pred, y_true.long())


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType,
) -> dict[str, float]:
    if task_type == "binclass":
        return {
            "average_precision": average_precision_score(y_true=y_true, y_score=y_pred),
            "roc_auc": roc_auc_score(y_true=y_true, y_score=y_pred),
        }

    elif task_type == "multiclass":
        return {
            "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred.argmax(-1)),
        }

    if task_type == "regression":
        return {
            "r2": r2_score(y_true=y_true, y_pred=y_pred),
            "mae": mean_absolute_error(y_true=y_true, y_pred=y_pred),
        }

    else:
        raise ValueError(f"Unknown TaskType={task_type}")


def compute_score(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType,
) -> float:
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, task_type=task_type)
    score_key = {
        "binclass": "average_precision",
        "multiclass": "accuracy",
        "regression": "r2",
    }[task_type]
    # NOTE: we assume that score is always maximized.
    #       If you want, for example, to minimize MAE,
    #       consider multiplying score by -1.0.
    return metrics[score_key]


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
