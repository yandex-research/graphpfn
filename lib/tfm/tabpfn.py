from typing import Any

import torch
from torch import Tensor

from lib.tfm import TFMBase
from lib.util import TaskType

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
except ImportError:
    TabPFNClassifier = None
    TabPFNRegressor = None
    ModelVersion = None


class TabPFNWrapper(TFMBase):
    """No gradient flow -- intentional."""

    def __init__(
        self,
        device: torch.device,
        use_builtin_ensembling: bool = False,
        checkpointing: bool = False,
        version: str = "v3",
        **config: Any,
    ) -> None:
        super().__init__()
        self._classifier: TabPFNClassifier | None = None
        self._regressor: TabPFNRegressor | None = None

        self.device = device
        self.config = config
        self.version = ModelVersion(version)
        if not use_builtin_ensembling:
            self.config["n_estimators"] = 1

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

        with torch.autocast(device_type=x_train.device.type, enabled=False):
            if task_type == TaskType.REGRESSION:
                self.regressor.fit(x_train_np, y_train_np)
                preds = self.regressor.predict(x_eval_np, output_type="mean")
                return torch.tensor(preds, device=x_train.device, dtype=torch.float32)
            elif task_type in [TaskType.BINCLASS, TaskType.MULTICLASS]:
                self.classifier.fit(x_train_np, y_train_np)
                probs_np = self.classifier.predict_proba(x_eval_np)
                probs = torch.tensor(
                    probs_np,
                    device=x_train.device,
                    dtype=torch.float32,
                )
                return torch.log(probs.clamp(min=1e-30))
            else:
                raise ValueError(f"Unknown {task_type=}")

    @property
    def classifier(self) -> TabPFNClassifier:
        if self._classifier is None:
            self._classifier = TabPFNClassifier.create_default_for_version(
                self.version,
                device=self.device,
                **self.config,
            )
        return self._classifier

    @property
    def regressor(self) -> TabPFNRegressor:
        if self._regressor is None:
            self._regressor = TabPFNRegressor.create_default_for_version(
                self.version,
                device=self.device,
                **self.config,
            )
        return self._regressor
