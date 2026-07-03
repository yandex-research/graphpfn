from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

import torch

from lib.util import TaskType


def unpack(config: Mapping[str, Any]) -> dict[str, Any]:
    """Strip _-prefixed keys for unpacking config into function args."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


# >>> Distribution Config Types


class UniformDistribution(TypedDict):
    _distribution_: Literal["uniform"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: float
    max: float


class LogUniformDistribution(TypedDict):
    _distribution_: Literal["log_uniform"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: float
    max: float


class LogUniformIntDistribution(TypedDict):
    _distribution_: Literal["log_uniform_int"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: int
    max: int


class UniformIntDistribution(TypedDict):
    _distribution_: Literal["uniform_int"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: int
    max: int


class ChoiceDistribution(TypedDict):
    """Uniform random selection from values, with optional weights.

    The optional `weights` dict specifies non-default weights for choices.
    Keys are string representations of choice values, values are weight multipliers.
    Default weight for unlisted choices is 1.0.

    Example:
        {"_distribution_": "choice",
         "values": ["relu", "tanh", "random_fourier"],
         "weights": {"random_fourier": 10}}  # random_fourier is 10x more likely
    """

    _distribution_: Literal["choice"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    values: list[ConfigValue]
    weights: NotRequired[dict[str, float]]


class BetaDistribution(TypedDict):
    _distribution_: Literal["beta"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    alpha: float
    beta: float


class BernoulliDistribution(TypedDict):
    """Sample True with probability p, False with probability 1-p."""

    _distribution_: Literal["bernoulli"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    p: float


class MixedLogUniformDistribution(TypedDict):
    """Bimodal log-uniform: samples from first range with prob p_first, else second."""

    _distribution_: Literal["mixed_log_uniform"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min_first: float
    max_first: float
    min_second: float
    max_second: float
    p_first: float


class UniformIntWithDefaultDistribution(TypedDict):
    """Uniform int in [min, max] with prob (1-p_default), else default."""

    _distribution_: Literal["uniform_int_with_default"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min: int
    max: int
    default: int
    p_default: float


class MetaBetaDistribution(TypedDict):
    """Meta-beta: samples alpha/beta from uniform [min, max], then beta * scale."""

    _distribution_: Literal["meta_beta"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    scale: float
    min: float
    max: float


class MetaTruncNormLogScaledDistribution(TypedDict):
    """Log-scaled truncated normal: samples mean/std on log scale.

    Samples:
        log_mean ~ uniform(log(min_mean), log(max_mean))
        log_std ~ uniform(log(min_std), log(max_std))
        mu = exp(log_mean)
        sigma = mu * exp(log_std)
        sample ~ TruncNorm(mu, sigma) + lower_bound
    """

    _distribution_: Literal["meta_trunc_norm_log_scaled"]
    _shared_: NotRequired[bool]
    _runtime_: NotRequired[bool | DistributionSpec]
    min_mean: float
    max_mean: float
    min_std: float
    max_std: float
    lower_bound: float
    round: bool


type DistributionSpec = (
    UniformDistribution
    | LogUniformDistribution
    | ChoiceDistribution
    | BetaDistribution
    | BernoulliDistribution
    | MixedLogUniformDistribution
    | MetaBetaDistribution
    | MetaTruncNormLogScaledDistribution
    | LogUniformIntDistribution
    | UniformIntDistribution
    | UniformIntWithDefaultDistribution
)


type ConfigValue = (
    None | bool | int | float | str | DistributionSpec | dict[str, "ConfigValue"]
)

type DistributionConfig = dict[str, ConfigValue]


# >>> Sampled Config Types


# Graph sampler configs
class SBMSamplerConfig(TypedDict):
    _type_: Literal["sbm"]
    n_groups: int
    offdiagonal_coef: float


class GeometricSamplerConfig(TypedDict):
    _type_: Literal["geometric"]
    n_latent_features: int


class PASamplerConfig(TypedDict):
    _type_: Literal["preferential-attachment"]


class ERSamplerConfig(TypedDict):
    _type_: Literal["erdos-renyi"]


class MultiLevelSBMWithPASamplerConfig(TypedDict):
    _type_: Literal["multi-level-sbm-with-pa"]
    pa_nodes_ratio: float
    pa_max_degree: int
    n_first_level_subgraphs: int
    first_level_degree_ratio: float
    min_first_level_n_nodes: int
    n_groups: int
    offdiagonal_coef: float


GraphSamplerConfig = (
    SBMSamplerConfig
    | GeometricSamplerConfig
    | PASamplerConfig
    | ERSamplerConfig
    | MultiLevelSBMWithPASamplerConfig
)


class GraphConfig(TypedDict):
    n_nodes: int
    avg_degree: float
    sampler: GraphSamplerConfig


# SCM sub-configs (shared between GNN and MLP SCMs)


class CausesConfig(TypedDict):
    strategy: Literal["normal", "uniform", "mixed"]
    pre_sample_stats: bool


class NoiseConfig(TypedDict):
    std: float
    pre_sample_std: bool


class InitConfig(TypedDict):
    std: float
    block_wise_dropout: bool
    p_dropout: float
    scale_std_by_dropout: bool


class CausalConfig(TypedDict):
    enabled: bool
    y_is_effect: bool
    in_clique: bool
    sort_features: bool


class StructuralConfig(TypedDict):
    use_degree: bool
    use_pagerank: bool
    lappe_k: int


# SCM configs (discriminated by _type_)


class MLPSCMConfig(TypedDict):
    _type_: Literal["mlp"]
    n_features: int
    n_causes: int
    n_layers: int
    hidden_dim: int
    activation_type: str | DistributionSpec
    causes: CausesConfig
    noise: NoiseConfig
    init: InitConfig
    causal: CausalConfig


class GNNSCMConfig(TypedDict):
    """GNN SCM config nests MLPSCMConfig as `base`, adding graph-specific params."""

    _type_: Literal["gnn"]
    base: MLPSCMConfig
    conv_type: Literal["gcn", "sage-mean", "sage-min", "sage-max", "gt"]
    graph_conv_ratio: float
    structural: StructuralConfig


class RegressionTaskConfig(TypedDict):
    _type_: Literal["regression"]


class BinclassTaskConfig(TypedDict):
    _type_: Literal["binclass"]
    quantile: float
    p_reverse: float


class MulticlassTaskConfig(TypedDict):
    _type_: Literal["multiclass"]
    n_classes: int
    multiclass_type: Literal["rank", "value"]
    p_ordered: float
    p_reverse: float


TaskConfig = RegressionTaskConfig | BinclassTaskConfig | MulticlassTaskConfig


class PostprocessingConfig(TypedDict):
    p_cat: float
    max_categories: int
    permute_features: bool
    permute_labels: bool


class GraphThenAttributesPriorConfig(TypedDict):
    _type_: Literal["graph_then_attributes"]
    graph: GraphConfig
    scm: GNNSCMConfig
    task: TaskConfig
    postprocessing: PostprocessingConfig
    train_ratio: float


class AttributesThenGraphPriorConfig(TypedDict):
    _type_: Literal["attributes_then_graph"]
    scm: MLPSCMConfig
    graph: GraphConfig
    task: TaskConfig
    postprocessing: PostprocessingConfig
    train_ratio: float


PriorConfig = GraphThenAttributesPriorConfig | AttributesThenGraphPriorConfig


class SanityCheckConfig(TypedDict):
    min_features: int


class SampledConfig(TypedDict):
    sanity_check: SanityCheckConfig
    prior: PriorConfig


# >>> Prior Data Types
# NOTE: Split is contiguous: train=[0, n_train_nodes), test=[n_train_nodes, n_nodes)


class PriorDataset(TypedDict):
    features: torch.Tensor
    labels: torch.Tensor
    edges: torch.Tensor
    n_train_nodes: int
    task_type: TaskType


class PriorDatasetBatch(TypedDict):
    features: torch.Tensor
    labels: torch.Tensor
    edges: torch.Tensor
    n_nodes: torch.Tensor
    n_features: torch.Tensor
    n_edges: torch.Tensor
    n_train_nodes: int
    task_type: TaskType
