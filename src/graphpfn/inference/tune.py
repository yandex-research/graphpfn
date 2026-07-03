from typing import Any

import numpy as np
import optuna
import torch

from graphpfn.data import GraphDataset
from graphpfn.inference.finetune import predict_ft
from graphpfn.inference.util import LossFn, ScoreFn
from graphpfn.util import KWArgs

DEFAULT_LR_GRID = [
    4.999998054699972e-06,
    8.340500244230498e-06,
    1.3912793292547576e-05,
    2.320794010302052e-05,
    3.871318040182814e-05,
    6.457747804233804e-05,
    0.00010772173118311912,
    0.00017969068721868098,
    0.00029974215431138873,
    0.0005000000819563866,
]


def tune_and_predict_ft(
    dataset: GraphDataset,
    *,
    score_fn: ScoreFn,
    loss_fn: LossFn,
    lr_grid: list[float] = DEFAULT_LR_GRID,
    n_steps: int = -1,
    steps_per_epoch: int = 10,
    patience: int = 4,
    train_batch_size: int = 1024,
    min_context_ratio: float = 0.75,
    model_kwargs: KWArgs = {},
    preprocessing_kwargs: KWArgs = {},
    amp: bool = True,
    device: str | torch.device,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    predictions: dict[int, np.ndarray] = {}

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_categorical("lr", lr_grid)
        y_pred = predict_ft(
            dataset,
            score_fn=score_fn,
            loss_fn=loss_fn,
            lr=lr,
            n_steps=n_steps,
            steps_per_epoch=steps_per_epoch,
            patience=patience,
            train_batch_size=train_batch_size,
            min_context_ratio=min_context_ratio,
            model_kwargs=model_kwargs,
            preprocessing_kwargs=preprocessing_kwargs,
            amp=amp,
            device=device,
            verbose=verbose,
        )
        predictions[trial.number] = y_pred
        val_mask = dataset.masks["val"]
        return score_fn(y_true=dataset.targets[val_mask], y_pred=y_pred[val_mask])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.BruteForceSampler(),
    )
    study.optimize(objective, n_trials=len(lr_grid))

    best_trial = study.best_trial
    return predictions[best_trial.number], best_trial.params
