from torch import Tensor, nn

from lib.util import TaskType


class TFMBase(nn.Module):
    """Common interface for tabular foundation models."""

    def forward(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_eval: Tensor,
        task_type: TaskType,
    ) -> Tensor:
        """Run in-context learning: predict on x_eval given (x_train, y_train).

        Returns:
            Classification: (n_eval, n_classes) log-probs
            Regression: (n_eval,) predictions
        """
        raise NotImplementedError
