import random
from functools import partial

import dgl
import numpy as np
import torch

from graphpfn import GraphDataset, tune_and_predict_ft
from graphpfn.inference.util import compute_loss, compute_metrics, compute_score


def main(
    dataset_name: str = "tolokers-2",
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)

    # >>> Data loading

    dataset = GraphDataset.from_graphland(dataset_name)

    # >>> Prediction

    preds, best_hparams = tune_and_predict_ft(
        dataset=dataset,
        score_fn=partial(compute_score, task_type=dataset.task_type),
        loss_fn=partial(compute_loss, task_type=dataset.task_type),
        device="cuda:0",
    )

    print(f"{best_hparams=}")

    # >>> Metrics computation

    test_mask = dataset.masks["test"]
    test_metrics = compute_metrics(
        y_true=dataset.targets[test_mask],
        y_pred=preds[test_mask],
        task_type=dataset.task_type,
    )
    print(f"Final test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
