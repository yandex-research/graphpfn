import math
from typing import cast

import numpy as np
import scipy.stats as stats

from lib.graphpfn.prior.prior_typings import (
    BernoulliDistribution,
    BetaDistribution,
    ChoiceDistribution,
    ConfigValue,
    DistributionConfig,
    DistributionSpec,
    LogUniformDistribution,
    LogUniformIntDistribution,
    MetaBetaDistribution,
    MetaTruncNormLogScaledDistribution,
    MixedLogUniformDistribution,
    SampledConfig,
    UniformDistribution,
    UniformIntDistribution,
    UniformIntWithDefaultDistribution,
)


# >>> Individual distribution samplers


def _sample_uniform(spec: UniformDistribution) -> float:
    return float(np.random.uniform(spec["min"], spec["max"]))


def _sample_log_uniform(spec: LogUniformDistribution) -> float:
    log_min = np.log(spec["min"])
    log_max = np.log(spec["max"])
    return float(np.exp(np.random.uniform(log_min, log_max)))


def _sample_log_uniform_int(spec: LogUniformIntDistribution) -> int:
    log_min = np.log(spec["min"])
    log_max = np.log(spec["max"])
    return round(np.exp(np.random.uniform(log_min, log_max)))


def _sample_uniform_int(spec: UniformIntDistribution) -> int:
    return int(np.random.randint(spec["min"], spec["max"] + 1))


def _sample_choice(spec: ChoiceDistribution) -> ConfigValue:
    values = spec["values"]
    weight_overrides = spec.get("weights", {})

    if not weight_overrides:
        idx = np.random.randint(len(values))
        return values[idx]

    # Build weights array
    weights = []
    for val in values:
        key = str(val) if not isinstance(val, str) else val
        weights.append(weight_overrides.get(key, 1.0))

    # Normalize to probabilities and sample
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = int(np.random.choice(len(values), p=probs))
    return values[idx]


def _sample_beta(spec: BetaDistribution) -> float:
    return float(np.random.beta(spec["alpha"], spec["beta"]))


def _sample_bernoulli(spec: BernoulliDistribution) -> bool:
    return bool(np.random.random() < spec["p"])


def _sample_mixed_log_uniform(spec: MixedLogUniformDistribution) -> float:
    if np.random.random() < spec["p_first"]:
        log_min = np.log(spec["min_first"])
        log_max = np.log(spec["max_first"])
    else:
        log_min = np.log(spec["min_second"])
        log_max = np.log(spec["max_second"])
    return float(np.exp(np.random.uniform(log_min, log_max)))


def _sample_uniform_int_with_default(spec: UniformIntWithDefaultDistribution) -> int:
    if np.random.random() < spec["p_default"]:
        return spec["default"]
    return int(np.random.randint(spec["min"], spec["max"] + 1))


def _sample_meta_beta(spec: MetaBetaDistribution) -> float:
    b = np.random.uniform(spec["min"], spec["max"])
    k = np.random.uniform(spec["min"], spec["max"])
    return float(spec["scale"] * np.random.beta(b, k))


def _sample_trunc_norm(mu: float, sigma: float) -> float:
    """Sample from truncated normal distribution with mean mu and std sigma."""
    lower = 0
    upper = 1_000_000
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return float(stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(1)[0])


def _sample_meta_trunc_norm_log_scaled(
    spec: MetaTruncNormLogScaledDistribution,
) -> int | float:
    log_mean = np.random.uniform(math.log(spec["min_mean"]), math.log(spec["max_mean"]))
    log_std = np.random.uniform(math.log(spec["min_std"]), math.log(spec["max_std"]))
    mu = math.exp(log_mean)
    sigma = mu * math.exp(log_std)
    sample = _sample_trunc_norm(mu, sigma)
    result = spec["lower_bound"] + (round(sample) if spec["round"] else sample)
    return int(result) if spec["round"] else float(result)


# >>> Dispatcher


def sample_value(spec: DistributionSpec) -> ConfigValue:
    """Sample a single value from a distribution specification.

    For choice distributions with dict values, returns the selected dict
    which may contain nested distributions to be sampled recursively.
    """
    match spec["_distribution_"]:
        case "uniform":
            return _sample_uniform(spec)
        case "log_uniform":
            return _sample_log_uniform(spec)
        case "log_uniform_int":
            return _sample_log_uniform_int(spec)
        case "uniform_int":
            return _sample_uniform_int(spec)
        case "choice":
            return _sample_choice(spec)
        case "beta":
            return _sample_beta(spec)
        case "bernoulli":
            return _sample_bernoulli(spec)
        case "mixed_log_uniform":
            return _sample_mixed_log_uniform(spec)
        case "uniform_int_with_default":
            return _sample_uniform_int_with_default(spec)
        case "meta_beta":
            return _sample_meta_beta(spec)
        case "meta_trunc_norm_log_scaled":
            return _sample_meta_trunc_norm_log_scaled(spec)
        case _:
            raise ValueError(f"Unknown distribution type: {spec['_distribution_']}")


def _sample_recursive(
    value: ConfigValue,
    path: tuple[str, ...],
    shared_values: dict[tuple[str, ...], ConfigValue],
) -> ConfigValue:
    if path in shared_values:
        cached = shared_values[path]
        # If cached value is a dict (result of sampling a choice), still process its
        # contents to resolve any nested shared values or distributions
        if isinstance(cached, dict) and "_distribution_" not in cached:
            resolved: dict[str, ConfigValue] = {}
            for key, val in cached.items():
                resolved[key] = _sample_recursive(val, (*path, key), shared_values)
            return resolved
        return cached

    if isinstance(value, dict):
        if "_distribution_" in value:
            spec = cast(DistributionSpec, value)

            # >>> Handle _runtime_ flag (may itself be a distribution)
            runtime_flag = spec.get("_runtime_", False)
            if isinstance(runtime_flag, dict) and "_distribution_" in runtime_flag:
                runtime_flag = bool(sample_value(cast(DistributionSpec, runtime_flag)))

            if runtime_flag:
                # Pass through for runtime sampling with resolved _runtime_ flag
                result_spec = cast(dict[str, ConfigValue], dict(spec.items()))
                result_spec["_runtime_"] = True
                return result_spec
            # <<<

            # Normal config-time sampling
            sampled = sample_value(spec)

            if isinstance(sampled, dict):
                return _sample_recursive(sampled, path, shared_values)

            return sampled
        else:
            result: dict[str, ConfigValue] = {}
            for key, val in value.items():
                result[key] = _sample_recursive(val, (*path, key), shared_values)
            return result

    return value


def sample_config(
    base_config: DistributionConfig,
    shared_values: dict[tuple[str, ...], ConfigValue] | None = None,
) -> SampledConfig:
    """Recursively sample concrete values from a distribution config.

    - Traverses nested dicts
    - For each DistributionSpec, samples a value
    - For "_distribution_": "choice" with dict values, selects one dict
      and recursively samples any nested distributions
    - Uses shared_values for params marked with "_shared_": True
    """
    if shared_values is None:
        shared_values = {}

    result = _sample_recursive(base_config, (), shared_values)
    return cast(SampledConfig, result)


def resolve_or_sample(value: ConfigValue) -> ConfigValue:
    """Sample if value is an unresolved (runtime) distribution, otherwise return as-is.

    Use this for config fields that may be either:
    - A concrete value (already sampled at config-time)
    - A runtime distribution spec (needs sampling now)

    Example:
        for layer_idx in range(n_layers):
            activation_name = resolve_or_sample(config["activation_type"])
            activation = make_activation(str(activation_name))
    """
    if isinstance(value, dict) and "_distribution_" in value:
        sampled = sample_value(cast(DistributionSpec, value))
        # Recursively resolve if choice returned a nested distribution
        if isinstance(sampled, dict) and "_distribution_" in sampled:
            return resolve_or_sample(sampled)
        return sampled
    return value


def _collect_and_validate_shared(
    value: ConfigValue,
    path: tuple[str, ...],
    has_non_shared_dist_ancestor: bool,
    shared_paths: list[tuple[str, ...]],
) -> None:
    if isinstance(value, dict):
        if "_distribution_" in value:
            spec = cast(DistributionSpec, value)
            is_shared = spec.get("_shared_", False)
            is_runtime = spec.get("_runtime_", False)

            # _runtime_ can be a distribution, check if it could resolve to True
            runtime_could_be_true = is_runtime is True or (
                isinstance(is_runtime, dict) and "_distribution_" in is_runtime
            )

            if is_shared and runtime_could_be_true:
                raise ValueError(
                    f"Parameter at path {path} has both _shared_=True and "
                    f"_runtime_=True (or _runtime_ distribution). "
                    f"These are mutually exclusive."
                )

            if is_shared and has_non_shared_dist_ancestor:
                raise ValueError(
                    f"Parameter at path {path} has _shared_=True but is "
                    f"inside a non-shared distribution"
                )

            if is_shared:
                shared_paths.append(path)

            new_has_non_shared_ancestor = has_non_shared_dist_ancestor or not is_shared

            if spec["_distribution_"] == "choice":
                for choice_value in spec["values"]:
                    if isinstance(choice_value, dict):
                        choice_dict = cast(dict[str, ConfigValue], choice_value)
                        for key, val in choice_dict.items():
                            _collect_and_validate_shared(
                                val,
                                (*path, key),
                                new_has_non_shared_ancestor,
                                shared_paths,
                            )
        else:
            for key, val in value.items():
                _collect_and_validate_shared(
                    val,
                    (*path, key),
                    has_non_shared_dist_ancestor,
                    shared_paths,
                )


def _sample_shared_recursive(
    value: ConfigValue,
    path: tuple[str, ...],
    shared_values: dict[tuple[str, ...], ConfigValue],
) -> None:
    if isinstance(value, dict):
        if "_distribution_" in value:
            spec = cast(DistributionSpec, value)
            is_shared = spec.get("_shared_", False)

            if is_shared:
                sampled = sample_value(spec)
                shared_values[path] = sampled

                if isinstance(sampled, dict):
                    sampled_dict = cast(dict[str, ConfigValue], sampled)
                    for key, val in sampled_dict.items():
                        _sample_shared_recursive(val, (*path, key), shared_values)
        else:
            for key, val in value.items():
                _sample_shared_recursive(val, (*path, key), shared_values)


def sample_configs(
    base_config: DistributionConfig,
    batch_size: int,
) -> list[SampledConfig]:
    """Sample batch_size configs, ensuring _shared_ params are identical.

    1. Validate: all _shared_ params have _shared_ or concrete ancestors
    2. First pass: sample all _shared_ params once
    3. Second pass: sample non-shared params independently per dataset

    Raises:
        ValueError: if _shared_ param has non-shared distribution ancestor
    """
    # >>> Step 1: Validate _shared_ ancestry
    shared_paths: list[tuple[str, ...]] = []
    _collect_and_validate_shared(base_config, (), False, shared_paths)

    # >>> Step 2: Sample _shared_ params once
    shared_values: dict[tuple[str, ...], ConfigValue] = {}
    _sample_shared_recursive(base_config, (), shared_values)

    # >>> Step 3: Sample non-shared params per config
    configs: list[SampledConfig] = []
    for _ in range(batch_size):
        config = sample_config(base_config, shared_values=shared_values)
        configs.append(config)

    return configs
