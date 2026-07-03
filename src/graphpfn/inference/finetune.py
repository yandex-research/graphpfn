import warnings

import delu
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from graphpfn.data import GraphDataset
from graphpfn.inference.preprocessing import (
    prepare_features_tensor,
    prepare_graph,
    standardize_targets,
)
from graphpfn.inference.util import CandidateQueue, LossFn, ScoreFn, apply_model
from graphpfn.model.graphpfn import GraphPFN
from graphpfn.util import KWArgs

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    module=r"torch\.utils\.checkpoint",
)


def predict_ft(
    dataset: GraphDataset,
    *,
    score_fn: ScoreFn,
    loss_fn: LossFn,
    lr: float,
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
    train_size = y_train.shape[0]

    if task_type == "regression":
        y_train, regression_target_stats = standardize_targets(y_train=y_train)
    else:
        regression_target_stats = None

    # >>> Model & Optimizer

    model = GraphPFN.from_pretrained(
        n_features=features.shape[-1],
        n_classes=dataset.n_classes,
        device=device,
        **model_kwargs,
    )
    model = model.eval().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # >>> Train setup

    def step_fn(*, context_mask: Tensor, query_mask: Tensor) -> Tensor:
        n_nodes = train_mask.shape[0]
        train_idx = train_mask.nonzero().squeeze(1)

        context_node_mask = train_mask.new_zeros([n_nodes], dtype=torch.bool)
        context_node_mask[train_idx[context_mask]] = True

        query_node_mask = train_mask.new_zeros([n_nodes], dtype=torch.bool)
        query_node_mask[train_idx[query_mask]] = True

        with torch.autocast(
            device.type,
            enabled=amp,
            dtype=torch.bfloat16 if amp else None,
        ):
            query_preds, query_targets = model.train_step_forward(
                graph=graph,
                features=features,
                y_context=y_train[context_mask],
                y_query=y_train[query_mask],
                context_node_mask=context_node_mask,
                query_node_mask=query_node_mask,
                task_type=task_type,
            )
        return loss_fn(y_true=query_targets, y_pred=query_preds)

    def apply_fn() -> np.ndarray:
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

    @torch.no_grad()
    def eval_fn() -> tuple[np.ndarray, float]:
        model.eval()
        y_pred = apply_fn()
        val_mask = dataset.masks["val"]
        y_val = dataset.targets[val_mask]
        return y_pred, score_fn(y_true=y_val, y_pred=y_pred[val_mask])

    # >>> Train loop

    step = 0
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best_y_pred, best_val_score = eval_fn()

    while n_steps == -1 or step < n_steps:
        # >>>
        model.train()
        epoch_losses = []

        n_candidates = min(
            train_batch_size,
            int(train_size * (1 - min_context_ratio)),
        )

        idx_queue = CandidateQueue(
            train_size,
            n_candidates=n_candidates,
            device=device,
        )
        delu.cuda.free_memory()

        for _ in tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {step // steps_per_epoch} Step {step}",
        ):
            query_idx = next(idx_queue)
            query_mask = query_idx.new_zeros([train_size], dtype=torch.bool)
            query_mask[query_idx] = True
            context_mask = ~query_mask

            optimizer.zero_grad()
            loss = step_fn(context_mask=context_mask, query_mask=query_mask)

            loss.backward()
            optimizer.step()

            step += 1
            epoch_losses.append(loss.detach())

        y_pred, val_score = eval_fn()

        if verbose:
            print(f"{step=}: {val_score=:.4f}")

        if val_score > best_val_score:
            if verbose:
                print("🌸 New best epoch! 🌸")
            best_val_score = val_score
            best_y_pred = y_pred

        early_stopping.update(val_score)
        if early_stopping.should_stop() or not np.isfinite(best_y_pred).all():
            break

    # >>> Return

    return best_y_pred
