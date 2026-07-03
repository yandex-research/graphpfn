from pathlib import Path

import dgl
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from graphpfn.model.ensemble import (
    apply_feature_transform,
    apply_target_transform,
    create_ensemble,
    reverse_target_transform,
)
from graphpfn.model.light import GraphPFNLight
from graphpfn.util import KWArgs, TaskType


class GraphPFN(nn.Module):
    def __init__(
        self,
        base: GraphPFNLight,
        n_members: int = 10,
        *,
        n_features: int,
        n_classes: int | None,
        device: torch.device | str,
        shuffle_features: bool | None = None,
        shuffle_targets: bool | None = None,
        max_columns: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        if shuffle_features is None:
            shuffle_features = n_members > 1
        if shuffle_targets is None:
            shuffle_targets = (
                (n_members > 1) and (n_classes is not None) and (n_classes > 2)
            )

        self.verbose = verbose
        self.base = base
        self.ensemble = create_ensemble(
            n_members=n_members,
            n_features=n_features,
            n_classes=n_classes,
            shuffle_features=shuffle_features,
            shuffle_targets=shuffle_targets,
            max_columns=max_columns,
            device=torch.device(device),
        )

    @classmethod
    def from_pretrained(
        cls,
        n_features: int,
        n_classes: int | None,
        *,
        device: torch.device | str,
        checkpoint: str | Path = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt",
        model_kwargs: KWArgs = dict(),
        ensemble_kwargs: KWArgs = dict(),
    ) -> "GraphPFN":
        graphpfn_light = GraphPFNLight(**model_kwargs)
        path = (
            resolve_checkpoint(checkpoint)
            if isinstance(checkpoint, str)
            else checkpoint
        )
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        assert all([k in graphpfn_light.state_dict() for k in state.keys()])
        graphpfn_light.load_state_dict(state, strict=False)
        return GraphPFN(
            graphpfn_light,
            n_features=n_features,
            n_classes=n_classes,
            device=device,
            **ensemble_kwargs,
        )

    def train_step_forward(
        self,
        graph: dgl.DGLGraph,
        features: Tensor,
        y_context: Tensor,
        y_query: Tensor,
        context_node_mask: Tensor,
        query_node_mask: Tensor,
        task_type: TaskType,
    ) -> tuple[Tensor, Tensor]:
        """
        Run one randomly-sampled ensemble member for a single training step.

        Args:
            graph: Input graph.
            features: Node features, shape [n_nodes, n_features].
            y_context: Context targets, shape [n_context].
            y_query: Query targets, shape [n_query].
            context_node_mask: Boolean mask over all nodes, True = context node.
            query_node_mask: Boolean mask over all nodes, True = query node.
            task_type: Task type string.

        Returns:
            Tuple of (query_preds, query_targets) both in the transformed
            (permuted / ECOC-mapped) target space.
        """
        _, member = self.ensemble.sample_member()
        features_transformed = apply_feature_transform(features, member)
        y_context_transformed = apply_target_transform(y_context, member)
        y_query_transformed = apply_target_transform(y_query, member)

        preds = self.base(
            graph=graph,
            features=features_transformed,
            y_train=y_context_transformed,
            train_mask=context_node_mask,
            task_type=task_type,
        )

        return preds[query_node_mask], y_query_transformed

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: Tensor,
        y_train: Tensor,
        train_mask: Tensor,
        task_type: TaskType,
        *,
        n_random_features: int | None = None,
    ) -> Tensor:
        ensemble_member_predictions: list[Tensor] = []

        for member in tqdm(self.ensemble, disable=not self.verbose):
            features_transformed = apply_feature_transform(features, member)
            y_train_transformed = apply_target_transform(y_train, member)

            member_pred = self.base(
                graph=graph,
                features=features_transformed,
                y_train=y_train_transformed,
                train_mask=train_mask,
                task_type=task_type,
                n_random_features=n_random_features,
            )

            member_pred = reverse_target_transform(member_pred, member)
            ensemble_member_predictions.append(member_pred)

        aggregated = self.ensemble.aggregate_predictions(
            ensemble_member_predictions, task_type
        )

        return aggregated


def resolve_checkpoint(
    name: str,
    *,
    local_dir: str | Path | None = None,
) -> Path:
    if not name.startswith("hf://"):
        return Path(name)
    parts = name.removeprefix("hf://").split("/")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, "config.json", local_dir=local_dir)
    local_path = hf_hub_download(repo_id, filename=filename, local_dir=local_dir)
    return Path(local_path)
