import numpy as np
import torch

from graphpfn.data import GraphDataset
from graphpfn.inference.preprocessing import (
    prepare_features_tensor,
    prepare_graph,
    standardize_targets,
)
from graphpfn.inference.util import apply_model
from graphpfn.model.graphpfn import GraphPFN
from graphpfn.util import KWArgs


@torch.no_grad()
def predict_icl(
    dataset: GraphDataset,
    *,
    model_kwargs: KWArgs = {},
    preprocessing_kwargs: KWArgs = {},
    amp: bool = True,
    device: str | torch.device,
) -> np.ndarray:
    device = torch.device(device)

    # >>> Data

    graph = prepare_graph(dataset.graph).to(device)
    features = prepare_features_tensor(
        dataset.features,
        dataset.masks["train"],
        **preprocessing_kwargs,
    ).to(device)
    train_mask = torch.tensor(dataset.masks["train"]).to(device)
    y_train = torch.tensor(dataset.targets).to(device)[train_mask]
    task_type = dataset.task_type

    if task_type == "regression":
        y_train, regression_target_stats = standardize_targets(y_train=y_train)
    else:
        regression_target_stats = None

    # >>> Model

    model = GraphPFN.from_pretrained(
        n_features=features.shape[-1],
        n_classes=dataset.n_classes,
        device=device,
        **model_kwargs,
    )
    model = model.eval().to(device)

    # >>> Final inference

    return apply_model(
        model=model,
        dataset=dataset,
        graph=graph,
        features=features,
        y_train=y_train,
        train_mask=train_mask,
        regression_target_stats=regression_target_stats,
        amp=amp,
        device=device,
    )
