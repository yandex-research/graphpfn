from __future__ import annotations

import dgl
import numpy as np
import torch

from lib.graph.data import GraphData, GraphDataset, GraphTask, get_score
from lib.graph.util import Setting
from lib.util import PartKey, Score

from .prior_typings import PriorDataset, PriorDatasetBatch


def unbatch_prior_dataset(batch: PriorDatasetBatch) -> list[PriorDataset]:
    batch_size = batch["features"].shape[0]
    n_nodes = batch["n_nodes"]
    n_features = batch["n_features"]
    n_edges = batch["n_edges"]

    datasets: list[PriorDataset] = []
    for i in range(batch_size):
        n, f, e = n_nodes[i].item(), n_features[i].item(), n_edges[i].item()
        datasets.append(
            PriorDataset(
                features=batch["features"][i, :n, :f],
                labels=batch["labels"][i, :n],
                edges=batch["edges"][i, :, :e],
                n_train_nodes=batch["n_train_nodes"],
                task_type=batch["task_type"],
            )
        )
    return datasets


def convert_to_graph_dataset(
    dataset: PriorDataset,
    name: str,
    setting: Setting = Setting.TRANSDUCTIVE,
    score: Score | None = None,
) -> GraphDataset[np.ndarray]:
    n_nodes = dataset["features"].shape[0]
    n_train = dataset["n_train_nodes"]
    task_type = dataset["task_type"]

    # >>> Build masks
    masks: dict[PartKey, np.ndarray] = {
        "train": np.arange(n_nodes) < n_train,
        "val": np.zeros(n_nodes, dtype=bool),
        "test": np.arange(n_nodes) >= n_train,
    }

    # >>> Build graph
    edges = dataset["edges"]
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=n_nodes,
        idtype=torch.int32,
    )

    labels = dataset["labels"].numpy().astype(np.float32)
    features = dataset["features"].numpy().astype(np.float32)

    data = GraphData(
        name=name,
        graph=graph,
        labels=labels,
        masks=masks,
        num_features=features,
        cat_features=None,
        frac_features=None,
    )
    task = GraphTask(
        labels=labels,
        masks=masks,
        type_=task_type,
        setting=setting,
        score=score if score is not None else get_score(task_type),
    )
    return GraphDataset(data, task)
