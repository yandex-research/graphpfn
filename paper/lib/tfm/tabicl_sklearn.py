from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lib.tfm import TFMBase
from lib.util import TaskType
from vendor.tabicl.sklearn import TabICLClassifier, TabICLRegressor


class TabICLSklearnWrapper(TFMBase):
    """Wraps TabICLv2 sklearn estimators behind TFMBase.

    Uses sklearn fit/predict interface. No gradient flow -- intentional.
    TabICLv2 has separate models for classification and regression.
    """

    def __init__(
        self,
        device: torch.device,
        use_builtin_ensembling: bool = False,
        checkpoints_path: str | Path = "checkpoints",
        classifier_version: str = "tabicl-classifier-v2-20260212.ckpt",
        regressor_version: str = "tabicl-regressor-v2-20260212.ckpt",
        **tabicl_config: Any,
    ) -> None:
        super().__init__()
        self._device = device
        self._tabicl_config = tabicl_config
        self._use_builtin_ensembling = use_builtin_ensembling
        self._checkpoints_path = Path(checkpoints_path)
        self._classifier_version = classifier_version
        self._regressor_version = regressor_version
        self._classifier: TabICLClassifier | None = None
        self._regressor: TabICLRegressor | None = None

    def forward(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_eval: Tensor,
        task_type: TaskType,
    ) -> Tensor:
        x_train_np = x_train.detach().cpu().float().numpy()
        y_train_np = y_train.detach().cpu().float().numpy()
        x_eval_np = x_eval.detach().cpu().float().numpy()

        if task_type == TaskType.REGRESSION:
            self.regressor.fit(x_train_np, y_train_np)
            preds = self.regressor.predict(x_eval_np, output_type="mean")
            return torch.tensor(preds, device=x_train.device, dtype=torch.float32)
        elif task_type in [TaskType.BINCLASS, TaskType.MULTICLASS]:
            self.classifier.fit(x_train_np, y_train_np)
            probs_np = self.classifier.predict_proba(x_eval_np)
            probs = torch.tensor(probs_np, device=x_train.device, dtype=torch.float32)
            return torch.log(probs.clamp(min=1e-30))
        else:
            raise ValueError(f"Unknown {task_type=}")

    @property
    def classifier(self) -> TabICLClassifier:
        if self._classifier is None:
            config = {
                **self._tabicl_config,
                "device": self._device,
                "model_path": self._checkpoints_path / self._classifier_version,
            }
            if not self._use_builtin_ensembling:
                config = {
                    **config,
                    "n_estimators": 1,
                    "feat_shuffle_method": "none",
                    "class_shuffle_method": "none",
                    "norm_methods": ["none"],
                }
            self._classifier = TabICLClassifier(**config)
        return self._classifier

    @property
    def regressor(self) -> TabICLRegressor:
        if self._regressor is None:
            config = {
                **self._tabicl_config,
                "device": self._device,
                "model_path": self._checkpoints_path / self._regressor_version,
            }
            if not self._use_builtin_ensembling:
                config = {
                    **config,
                    "n_estimators": 1,
                    "feat_shuffle_method": "none",
                    "norm_methods": ["none"],
                }
            self._regressor = TabICLRegressor(**config)
        return self._regressor
