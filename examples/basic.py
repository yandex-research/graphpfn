import random

import dgl
import numpy as np
import torch
from torch_geometric.data import Data as PygData
from torch_geometric.datasets import HeterophilousGraphDataset

from graphpfn import GraphDataset, predict_icl
from graphpfn.inference.util import compute_metrics


def main(
    dataset_name: str = "amazon-ratings",
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)

    # >>> Data loading

    pyg_dataset: PygData = HeterophilousGraphDataset(
        f"data/{dataset_name}",
        dataset_name,
    )[0]  # type: ignore
    dataset = GraphDataset.from_pyg(
        pyg_dataset,
        task_type="multiclass",
        name=dataset_name,
    )

    # NOTE: You can easily change, e.g., splits:
    # custom_splits = dict(np.load(f"data/{dataset_name}/split.npz"))
    # dataset.masks = custom_splits

    # >>> Prediction

    preds = predict_icl(
        dataset=dataset,
        model_kwargs={
            "checkpoint": "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt",
        },
        preprocessing_kwargs={
            "d_pca": 64
        },  # PCA is recommended for datasets with neural embeddings features
        device="cuda:0",
    )

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
