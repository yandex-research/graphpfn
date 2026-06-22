import math
import random
from collections.abc import Iterator
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import TypedDict

from lib.data import TaskType


class EnsembleMember(TypedDict):
    feature_perm: Tensor | None
    target_perm: Tensor | None
    column_indices: Tensor | None
    ecoc_targets: Tensor | None


# >>> ECOC Functions


def get_n_estimators(
    n_classes: int,
    alphabet_size: int,
    n_estimators_redundancy: int = 8,
) -> int:
    assert alphabet_size >= 2
    min_estimators = max(
        math.ceil(math.log(n_classes, alphabet_size)),
        math.ceil(n_classes / (alphabet_size - 1)),
    )
    return min_estimators * n_estimators_redundancy


def prepare_codebook(
    n_classes: int,
    n_estimators: int,
    alphabet_size: int,
    n_retries: int = 50,
) -> Tensor:
    """
    Generate ECOC codebook with retry logic to maximize class separation.

    Args:
        n_classes: Number of original classes.
        n_estimators: Number of ECOC estimators (columns in codebook).
        alphabet_size: Number of meta-classes per estimator.
        n_retries: Number of random codebooks to try, keeping the best.

    Returns:
        Codebook tensor of shape (n_classes, n_estimators).
    """
    prepare_codebook_single = partial(
        _prepare_codebook_single,
        n_classes=n_classes,
        n_estimators=n_estimators,
        alphabet_size=alphabet_size,
    )

    best_codebook = prepare_codebook_single()
    best_score = _get_codebook_score(best_codebook)

    for _ in range(n_retries):
        codebook = prepare_codebook_single()
        score = _get_codebook_score(codebook)
        if best_score < score:
            best_codebook = codebook
            best_score = score

    assert best_score[0] > 0
    return best_codebook


def _prepare_codebook_single(
    n_classes: int,
    n_estimators: int,
    alphabet_size: int,
) -> Tensor:
    assert n_classes > alphabet_size
    assert alphabet_size > 1

    codebook = torch.zeros((n_estimators, n_classes), dtype=torch.int64)
    class_indices = torch.arange(n_classes)
    highlight_rows = min(n_classes, n_estimators)

    for row in range(highlight_rows):
        focus = row % n_classes
        others = class_indices[class_indices != focus]
        codebook[row, focus] = 0
        others = others[torch.randperm(others.shape[0])]
        groups = torch.tensor_split(others, alphabet_size - 1)
        for offset, group in enumerate(groups, start=1):
            codebook[row, group] = offset

    for row in range(highlight_rows, n_estimators):
        class_indices = class_indices[torch.randperm(class_indices.shape[0])]
        groups = torch.tensor_split(class_indices, alphabet_size)
        for code, group in enumerate(groups):
            codebook[row, group] = code

    return codebook.T


def _get_codebook_score(
    codebook: Tensor,
) -> tuple[float, float]:
    codebook_ohe = F.one_hot(codebook, int(codebook.max() + 1))
    distances = (codebook_ohe[None] - codebook_ohe[:, None]).abs().sum([2, 3]).float()
    diagonal = torch.eye(distances.shape[0], dtype=torch.bool)
    distances_offdiagonal = distances[~diagonal]
    return distances_offdiagonal.min().item(), distances_offdiagonal.mean().item()


def aggregate_log_probs(
    log_probs: Tensor,
    codebook: Tensor,
) -> Tensor:
    """
    Aggregate log probabilities from ECOC estimators into class predictions.

    Args:
        log_probs: Log probabilities, shape (..., alphabet_size, n_estimators).
        codebook: ECOC codebook, shape (n_classes, n_estimators).

    Returns:
        Aggregated log probabilities, shape (..., n_classes).
    """
    assert log_probs.ndim >= 3
    log_probs_batch_shape = log_probs.shape[:-2]
    log_probs = log_probs.reshape(-1, log_probs.shape[-2], log_probs.shape[-1])
    assert log_probs.shape[2] == codebook.shape[1]
    alphabet_size = log_probs.shape[1]
    codebook_ohe = F.one_hot(codebook, num_classes=alphabet_size).float()
    scores = torch.einsum("nae, cea -> nc", log_probs, codebook_ohe)
    return F.log_softmax(scores, dim=1).reshape(*log_probs_batch_shape, -1)


def get_min_ensemble_members(
    n_classes: int,
    alphabet_size: int,
) -> int:
    """
    Compute minimum ensemble members needed for ECOC.

    Returns 1 if n_classes <= alphabet_size (no ECOC needed).
    """
    if n_classes <= alphabet_size:
        return 1
    return max(
        math.ceil(math.log(n_classes, alphabet_size)),
        math.ceil(n_classes / (alphabet_size - 1)),
    )


# >>> Column Subsampler


class ColumnSubsampler:
    def __init__(
        self,
        n_features: int,
        max_columns: int,
    ) -> None:
        self.n_features = n_features
        self.max_columns = max_columns

    def sample_indices(self, rng: torch.Generator) -> Tensor:
        perm = torch.randperm(self.n_features, generator=rng, device=rng.device)
        return perm[: self.max_columns].sort().values

    @property
    def enabled(self) -> bool:
        return self.n_features > self.max_columns


# >>> Ensemble Transform Functions


def apply_feature_transform(
    features: Tensor | list[Tensor], member: EnsembleMember
) -> Tensor:
    if isinstance(features, list):
        features = torch.cat(features, dim=-1)
    if member["feature_perm"] is not None:
        features = features[:, member["feature_perm"]]
    if member["column_indices"] is not None:
        features = features[:, member["column_indices"]]
    return features


def apply_target_transform(
    targets: Tensor,
    member: EnsembleMember,
) -> Tensor:
    if member["ecoc_targets"] is not None:
        targets = member["ecoc_targets"][targets.long()].float()
    if member["target_perm"] is not None:
        targets = member["target_perm"][targets.long()].float()
    return targets


def reverse_target_perm(
    predictions: Tensor,
    member: EnsembleMember,
) -> Tensor:
    if member["target_perm"] is not None:
        predictions = predictions[..., member["target_perm"]]
    return predictions


# >>> Ensemble Class


class Ensemble:
    def __init__(
        self,
        n_members: int,
        n_features: int,
        n_classes: int | None,
        *,
        feature_group_sizes: list[int] | None = None,
        shuffle_features: bool = True,
        shuffle_targets: bool = True,
        max_columns: int | None = None,
        ecoc_codebook: Tensor | None = None,
        max_effective_classes: int = 10,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        """
        Create an ensemble with all members pre-computed.

        Args:
            n_members: Total number of ensemble members (subtasks).
            n_features: Number of input features.
            n_classes: Number of classes (None for regression).
            feature_group_sizes: Sizes of feature groups for group-aware shuffling.
                When provided, features are shuffled within groups and group order
                is permuted. Must sum to n_features. When None, all features are
                treated as a single group.
            shuffle_features: Whether to randomly permute features.
            shuffle_targets: Whether to randomly permute target labels.
            max_columns: If n_features > max_columns, subsample to max_columns.
                When shuffle_features is True, subsampling is done by truncation
                after the group-aware permutation. When shuffle_features is False,
                a random subset of columns is selected.
            ecoc_codebook: Precomputed ECOC codebook, shape (n_classes, n_members).
            max_effective_classes: Max number of classes in each ensemble member.
            device: Torch device for ECOC codebook and permutations.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If ecoc_codebook.shape[1] != n_members.
        """
        if ecoc_codebook is not None and ecoc_codebook.shape[1] != n_members:
            raise ValueError(
                f"ecoc_codebook.shape[1]={ecoc_codebook.shape[1]} "
                f"!= n_members={n_members}"
            )

        if feature_group_sizes is not None:
            assert sum(feature_group_sizes) == n_features
        else:
            feature_group_sizes = [n_features]

        self._n_members = n_members
        self._n_features = n_features
        self._n_classes = n_classes
        self._feature_group_sizes = feature_group_sizes
        self._shuffle_features = shuffle_features
        self._shuffle_targets = shuffle_targets
        self._ecoc_codebook = ecoc_codebook
        self._max_effective_classes = max_effective_classes
        self._device = device
        self._seed = seed

        self._column_subsampler: ColumnSubsampler | None = None
        if max_columns is not None and n_features > max_columns:
            self._column_subsampler = ColumnSubsampler(n_features, max_columns)

        self._members: list[EnsembleMember] = []
        self._build_members()

    def _build_group_perm(
        self, rng: torch.Generator, device: torch.device | None
    ) -> Tensor:
        group_sizes = self._feature_group_sizes
        n_groups = len(group_sizes)

        starts: list[int] = []
        offset = 0
        for size in group_sizes:
            starts.append(offset)
            offset += size

        group_order = torch.randperm(n_groups, generator=rng, device=device)

        perm_parts: list[Tensor] = []
        for g_idx in group_order.tolist():
            size = group_sizes[g_idx]
            if size == 0:
                continue
            start = starts[g_idx]
            intra_perm = torch.randperm(size, generator=rng, device=device)
            group_indices = torch.arange(start, start + size, device=device)
            perm_parts.append(group_indices[intra_perm])

        return torch.cat(perm_parts)

    def _build_members(self) -> None:
        device = self._device
        rng = torch.Generator(device).manual_seed(self._seed)

        for i in range(self._n_members):
            feature_perm: Tensor | None = None
            if self._shuffle_features:
                feature_perm = self._build_group_perm(rng, device)

            column_indices: Tensor | None = None
            if self._column_subsampler is not None:
                column_indices = self._column_subsampler.sample_indices(rng)

            target_perm: Tensor | None = None
            if self._shuffle_targets and self._n_classes is not None:
                effective_n_classes = self.effective_n_classes
                assert effective_n_classes is not None
                target_perm = torch.randperm(
                    effective_n_classes, generator=rng, device=device
                )

            ecoc_targets: Tensor | None = None
            if self._ecoc_codebook is not None:
                ecoc_targets = self._ecoc_codebook[:, i].to(device)

            self._members.append(
                EnsembleMember(
                    feature_perm=feature_perm,
                    target_perm=target_perm,
                    column_indices=column_indices,
                    ecoc_targets=ecoc_targets,
                )
            )

    @property
    def n_members(self) -> int:
        return self._n_members

    @property
    def uses_ecoc(self) -> bool:
        return self._ecoc_codebook is not None

    @property
    def ecoc_codebook(self) -> Tensor | None:
        return self._ecoc_codebook

    @property
    def effective_n_classes(self) -> int | None:
        if self.uses_ecoc:
            return self._max_effective_classes
        else:
            return self._n_classes

    def get_member(self, idx: int) -> EnsembleMember:
        return self._members[idx]

    def sample_member(self) -> tuple[int, EnsembleMember]:
        idx = random.randint(0, self._n_members - 1)
        return idx, self._members[idx]

    def __iter__(self) -> Iterator[EnsembleMember]:
        return iter(self._members)

    def __len__(self) -> int:
        return self._n_members

    def aggregate_predictions(
        self,
        predictions: list[Tensor],
        task_type: str | TaskType,
    ) -> Tensor:
        """
        Aggregate predictions from all ensemble members.

        For regression: average predictions.
        For classification with ECOC: aggregate log-probs using codebook.
        For classification without ECOC: average probabilities and return log-probs.
        """
        task_type = TaskType(task_type)

        if task_type == TaskType.REGRESSION:
            return torch.stack(predictions, dim=0).mean(dim=0)

        if self._ecoc_codebook is not None:
            log_probs = F.log_softmax(torch.stack(predictions, dim=-1), dim=-2)
            return aggregate_log_probs(log_probs, self._ecoc_codebook)

        log_probs = torch.stack([F.log_softmax(p, dim=-1) for p in predictions], dim=0)
        return torch.logsumexp(log_probs, dim=0) - math.log(len(predictions))


def create_ensemble(
    n_members: int,
    n_features: int,
    n_classes: int | None,
    *,
    feature_group_sizes: list[int] | None = None,
    shuffle_features: bool = True,
    shuffle_targets: bool = True,
    max_columns: int | None = None,
    max_effective_classes: int = 10,
    device: torch.device | None = None,
    seed: int = 0,
) -> Ensemble:
    """
    Factory function to create an Ensemble with proper ECOC setup.

    Handles:
    - ECOC codebook creation when n_classes > max_effective_classes
    - Validation that n_members >= min_ecoc_estimators
    """
    ecoc_codebook: Tensor | None = None
    if n_classes is not None and n_classes > max_effective_classes:
        min_members = get_min_ensemble_members(n_classes, max_effective_classes)
        if n_members < min_members:
            raise ValueError(
                f"n_members={n_members} < min_members={min_members} "
                f"for n_classes={n_classes}"
            )
        ecoc_codebook = prepare_codebook(
            n_classes=n_classes,
            n_estimators=n_members,
            alphabet_size=max_effective_classes,
        )
        if device is not None:
            ecoc_codebook = ecoc_codebook.to(device)

    return Ensemble(
        n_members=n_members,
        n_features=n_features,
        n_classes=n_classes,
        feature_group_sizes=feature_group_sizes,
        shuffle_features=shuffle_features,
        shuffle_targets=shuffle_targets,
        max_columns=max_columns,
        ecoc_codebook=ecoc_codebook,
        max_effective_classes=max_effective_classes,
        device=device,
        seed=seed,
    )
