"""Main pipeline: attributes -> graph -> postprocessing."""

from __future__ import annotations

import dgl
import torch

from lib.util import TaskType

from ..attributes import sample_attributes_mlp
from ..checks import SanityCheckError
from ..graphs.sbm import sample_sbm_from_groups
from ..graphs.util import extract_largest_component, to_simple
from ..postprocessing import (
    apply_task,
    compute_n_train_nodes,
    drop_constant_features,
    process_features,
)
from ..prior_typings import AttributesThenGraphPriorConfig, PriorDataset, unpack


def sample_graph_from_attributes(
    labels: torch.Tensor,
    avg_degree: float,
    offdiagonal_coef: float,
) -> tuple[dgl.DGLGraph, torch.Tensor]:
    graph = sample_sbm_from_groups(labels, avg_degree, offdiagonal_coef)
    graph = to_simple(graph)
    graph, mask = extract_largest_component(graph)

    graph = to_simple(graph)
    return graph, mask


def sample_dataset(config: AttributesThenGraphPriorConfig) -> PriorDataset:
    n_nodes = config["graph"]["n_nodes"]
    sampler = config["graph"]["sampler"]
    # TODO: refactor GraphConfig so that the sampler can be restricted at the type level
    if sampler["_type_"] != "sbm-simple":
        raise ValueError(
            f"attributes_then_graph only supports sbm-simple sampler,"
            f" got: {sampler['_type_']}"
        )

    # >>> Sample attributes (MLP SCM, no graph needed)
    features, labels = sample_attributes_mlp(n_nodes, config["scm"])

    # >>> Apply task
    task_type = TaskType(config["task"]["_type_"])
    assert task_type == TaskType.MULTICLASS
    labels = apply_task(
        labels, config["task"], config["postprocessing"]["permute_labels"]
    )

    # >>> Generate attribute-aware graph
    graph, mask = sample_graph_from_attributes(
        labels.to(torch.int32),
        avg_degree=config["graph"]["avg_degree"],
        **unpack(sampler),
    )
    features = features[mask]
    labels = labels[mask]

    # >>> Shuffle node ordering + extract edges
    actual_n_nodes = graph.num_nodes()
    perm = torch.randperm(actual_n_nodes)
    src, dst = graph.edges()
    edges = torch.stack([perm[src], perm[dst]], dim=0)
    inverse_perm = torch.argsort(perm)
    features = features[inverse_perm]
    labels = labels[inverse_perm]

    # >>> Compute n_train_nodes
    n_train_nodes = compute_n_train_nodes(n_nodes, config["train_ratio"])
    if n_train_nodes >= actual_n_nodes:
        raise SanityCheckError(
            f"n_train_nodes ({n_train_nodes}) >= actual_n_nodes ({actual_n_nodes}). "
            "Graph became too small after extracting largest component."
        )

    # >>> Postprocess features
    features = process_features(
        features,
        p_cat=config["postprocessing"]["p_cat"],
        max_categories=config["postprocessing"]["max_categories"],
        do_permute_features=config["postprocessing"]["permute_features"],
    )

    train_features = features[:n_train_nodes, :]
    _, col_mask = drop_constant_features(train_features)
    features = features[:, col_mask]

    return {
        "features": features,
        "labels": labels,
        "edges": edges,
        "n_train_nodes": n_train_nodes,
        "task_type": task_type,
    }
