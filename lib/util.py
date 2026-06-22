import argparse
import datetime
import enum
import functools
import hashlib
import importlib
import inspect
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tomllib
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import (
    Any,
    NotRequired,
    Required,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import tomli_w
import yaml
from loguru import logger
from optuna import Study
from optuna.trial import TrialState

try:
    from dev.infra import save_snapshot, tracker

    INTERNAL_INFRA = True
except ImportError:
    from types import SimpleNamespace

    tracker = SimpleNamespace(
        init=lambda *a, **kw: None,
        log=lambda *a, **kw: None,
        get_uid=lambda: None,
        finish=lambda: None,
    )
    save_snapshot = lambda *a, **kw: None  # noqa: E731
    INTERNAL_INFRA = False

# NOTE
# This file must NOT import anything from lib except for `env`,
# because all other submodules are allowed to import `util`.
from . import env

# The purpose of the following snippet is to optimize import times
# when slow-to-import modules are not needed.
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


# ==================================================================================
# Const
# ==================================================================================
WORST_SCORE = -999999.0


# ==================================================================================
# Types
# ==================================================================================
KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # Must be JSON-serializable.

DataKey = str  # 'x_num', 'x_bin', 'x_cat', 'y', ...
PartKey = str  # 'train', 'val', 'test', ...

PROJECT_DIR = Path(__file__).parent.parent
CACHE_DIR = PROJECT_DIR / "cache"
DATA_DIR = PROJECT_DIR / "data"
EXP_DIR = PROJECT_DIR / "exp"

assert PROJECT_DIR.exists()
CACHE_DIR.mkdir(exist_ok=True)


class TaskType(enum.Enum):
    REGRESSION = "regression"
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"


class PredictionType(enum.Enum):
    LABELS = "labels"
    PROBS = "probs"
    LOGITS = "logits"


class Score(enum.Enum):
    ACCURACY = "accuracy"
    CROSS_ENTROPY = "cross-entropy"
    MAE = "mae"
    R2 = "r2"
    RMSE = "rmse"
    ROC_AUC = "roc-auc"
    AP = "ap"


# ==================================================================================
# `main` function
# ==================================================================================
# The following utilities expect that the `main` function
# has one of the following signatures:
#
# 1. main(config, output = None, *, force: bool = False) -> None | JSONDict
# 2. main(config, output = None, *, force: bool = False, continue_: bool = False) -> None | JSONDict  # noqa
#
# Notes:
# * `config` is a Python dictionary or a path to a config in the TOML format.
# * `output` is the output directory with all results of the run.
#   If not provided, it it automatically inferred from the config path.
# * Setting `force=True` means removing the already existing output.
# * Setting `continue_=True` means continuing the execution from an existing output
#   (if exists; otherwise, from scratch).
# * The return value is `report` -- a JSON-serializable Python dictionary
#   with arbitrary information about the run.
T = TypeVar("T")


def _is_typeddict(tp: object) -> bool:
    return (
        isinstance(tp, type)
        and issubclass(tp, dict)
        and hasattr(tp, "__required_keys__")
        and hasattr(tp, "__optional_keys__")
    )


def _check_typeddict_keys(config: dict, config_type: type, path: str = "") -> None:
    presented_keys = frozenset(config)
    required_keys = config_type.__required_keys__
    optional_keys = config_type.__optional_keys__

    prefix = f'At "{path}": ' if path else ""
    assert presented_keys >= required_keys, (
        f"{prefix}The config is missing the following required keys:"
        f" {', '.join(required_keys - presented_keys)}"
    )
    assert set(config) <= (required_keys | optional_keys), (
        f"{prefix}The config has unknown keys:"
        f" {', '.join(presented_keys - required_keys - optional_keys)}"
    )

    hints = get_type_hints(config_type, include_extras=True)
    for key, value in config.items():
        if key not in hints or not isinstance(value, dict):
            continue
        inner_type = hints[key]
        origin = get_origin(inner_type)
        if origin is Required or origin is NotRequired:
            inner_type = get_args(inner_type)[0]
        if _is_typeddict(inner_type):
            _check_typeddict_keys(
                value, inner_type, path=f"{path}.{key}" if path else key
            )


def check[T](
    config, output: None | str | Path, *, config_type: type[T] = dict
) -> tuple[T, Path]:
    """Load the config and infer the path to the output directory."""
    # >>> This is a snippet for the internal infrastructure, ignore it.
    snapshot_dir = os.environ.get("SNAPSHOT_PATH")
    if snapshot_dir and Path(snapshot_dir).joinpath("SNAPSHOT_UNPACKED").exists():
        caller_info = inspect.stack()[1]
        caller_fn = caller_info.frame.f_globals[caller_info.function]
        if "continue_" in inspect.signature(caller_fn).parameters:
            assert caller_info.frame.f_locals["continue_"]
    del snapshot_dir
    # <<<

    # >>> Check paths.
    if isinstance(config, str | Path):
        # config is a path.
        config = Path(config)
        assert config.suffix == ".toml"
        assert config.exists(), f"The config {config} does not exist"
        if output is None:
            # In this case, output is a directory located next to the config.
            output = config.with_suffix("")
        config = load_config(config)
    else:
        # config is already a dictionary.
        assert output is not None, (
            "If config is a dictionary, then the `output` directory must be provided."
        )
    output = Path(output).resolve()

    # >>> Check the config.
    if config_type is not dict and _is_typeddict(config_type):
        _check_typeddict_keys(config, config_type)

    return cast(T, config), output


def start(
    main_fn: Callable,
    output: str | Path,
    *,
    continue_: bool = False,
    force: bool = False,
) -> bool:
    """Create the output directory (if missing).

    Returns:
        True if the caller should continue the execution.
        False if the caller should immediately return.
    """
    output = Path(output).resolve()

    print_sep()
    print(
        f"{get_function_full_name(main_fn)}"
        f" | {try_get_relative_path(output)}"
        f" | {datetime.datetime.now()}"
    )
    print_sep()

    if output.exists():
        if force:
            logger.warning("Removing the existing output")
            shutil.rmtree(output)
            _create_output(output)
            return True
        elif not continue_:
            backup(output)
            logger.warning("The output already exists!")
            return False
        elif is_done(output):
            backup(output)
            logger.info("Already done!\n")
            return False
        else:
            logger.info("Continuing with the existing output")
            _create_output(output, exist_ok=True)
            return True
    else:
        logger.info("Creating the output")
        _create_output(output)
        return True


def create_report(
    function,
    config: dict[str, Any],
    output: None | Path = None,
    *,
    continue_: bool = False,
) -> JSONDict:
    if output is not None and get_report_path(output).exists():
        if not continue_:
            raise RuntimeError("The report already exists")
        report = load_report(output)
        if report["config"] != config:
            raise RuntimeError(
                "An existing report was loaded,"
                " however, it contains a different config than the new one."
            )
    else:
        report = {
            "function": get_function_full_name(function),
            "commit": get_git_revision_hash(),
            "gpus": get_gpu_names(),
            "config": jsonify(config),
        }
    return report


def _summarize_to_dict(report: JSONDict) -> JSONDict:
    summary = {}
    if "function" in report:
        function = report["function"]
        summary["function"] = function
    else:
        function = None

    def try_add(key: str) -> None:
        if key in report:
            summary[key] = deepcopy(report[key])

    if "time" in report:
        summary["time"] = str(datetime.timedelta(seconds=int(report["time"])))
    gpus = report.get("gpus")
    if gpus is not None and gpus:
        assert len(gpus) == 1 or all(x == gpus[0] for x in gpus)
        summary["gpus"] = gpus[0].removeprefix("NVIDIA ") + (
            f" x{len(gpus)}" if len(gpus) > 1 else ""
        )

    if function == "bin.tune.main":
        try_add("n_completed_trials")
        try_add("tuning_time")
        summary["best"] = _summarize_to_dict(report["best"])
        summary["best"].pop("gpus", None)

    elif function in ["bin.evaluate.main", "bin.ensemble.main"]:
        reports = report["reports"]
        if function == "bin.evaluate.main":
            summary["average_time"] = str(
                datetime.timedelta(seconds=statistics.mean(x["time"] for x in reports))
            )
        summary["n_reports"] = len(reports)
        summary["scores"] = {
            part: float(statistics.mean(x["metrics"][part]["score"] for x in reports))
            for part in reports[0]["metrics"]
        }
        del reports

    else:
        try_add("trial_id")
        try_add("n_parameters")
        try_add("best_stage")
        try_add("best_step")
        if "best_step" in report and "epoch_size" in report:
            summary["best_epoch"] = report["best_step"] // report["epoch_size"]
        metrics = report.get("metrics")
        if metrics is not None and "score" in next(iter(metrics.values())):
            summary["scores"] = {part: metrics[part]["score"] for part in metrics}

    return summary


def summarize(report: JSONDict) -> str:
    """Make a human-readable summary of the report."""
    # NOTE
    # The fact that summary is a valid YAML document
    # is an implementation detail and can change in future.
    buf = io.StringIO()
    yaml.dump(_summarize_to_dict(report), buf, indent=4, sort_keys=False)
    return buf.getvalue()


def finish(output: Path, report: JSONDict) -> None:
    dump_report(output, report)

    # >>> A code block for the internal infrastructure, ignore it.
    JSON_OUTPUT_FILE = os.environ.get("JSON_OUTPUT_FILE")
    if JSON_OUTPUT_FILE:
        try:
            key = str(output.relative_to(env.get_project_dir()))
        except ValueError:
            pass
        else:
            json_output_path = Path(JSON_OUTPUT_FILE)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_report(output)
            json_output_path.write_text(json.dumps(json_data, indent=4))
            shutil.copyfile(
                json_output_path,
                os.path.join(os.environ["SNAPSHOT_PATH"], "json_output.json"),
            )
    # <<<

    dump_summary(output, summarize(report))
    _mark_as_done(output)
    print_summary(output)
    backup(output)


def run(function: Callable[..., None | JSONDict]) -> None | JSONDict:
    """Run CLI for the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    if "continue_" in inspect.signature(function).parameters:
        parser.add_argument("--continue", action="store_true", dest="continue_")

    return function(**vars(parser.parse_args(sys.argv[1:])))


def backup(output: Path) -> None:
    save_snapshot(output, project_dir=env.get_project_dir(), extra_suffixes=[".toml"])


def _get_running_path(output: str | Path) -> Path:
    return Path(output).joinpath("_RUNNING")


def _create_output(output: Path, *, exist_ok: bool = False) -> None:
    # This function ensures that the _RUNNING file is created
    # immediately after the creation of the output directory.
    if output.exists():
        assert exist_ok
    else:
        output.mkdir()
    _get_running_path(output).touch()


def _mark_as_done(output: Path) -> None:
    assert output.exists()
    assert not is_done(output)
    _get_running_path(output).unlink()


def is_done(output: str | Path) -> bool:
    # The report must be presented. Otherwise, an empty directory would be "done"
    # (for example, just after the creation of the output directory).
    return get_report_path(output).exists() and not _get_running_path(output).exists()


def _get_deprecated_path(path: str | Path) -> Path:
    return Path(path).joinpath("_DEPRECATED")


def is_deprecated(path: str | Path) -> bool:
    return _get_deprecated_path(path).exists()


# ==================================================================================
# IO for the output directory
# ==================================================================================
def get_report_path(output: str | Path) -> Path:
    return Path(output) / "report.json"


def load_config(output_or_config: str | Path) -> JSONDict:
    return tomllib.loads(Path(output_or_config).with_suffix(".toml").read_text())


def dump_config(
    output_or_config: str | Path, config: JSONDict, *, force: bool = False
) -> None:
    config_path = Path(output_or_config).with_suffix(".toml")
    if config_path.exists() and not force:
        raise RuntimeError(
            "The following config already exists (pass force=True to overwrite it)"
            f" {config_path}"
        )
    config_path.write_text(tomli_w.dumps(config))


def load_report(output: str | Path) -> JSONDict:
    return json.loads(get_report_path(output).read_text())


def dump_report(output: str | Path, report: JSONDict) -> None:
    get_report_path(output).write_text(json.dumps(report, indent=4))


def get_summary_path(output: str | Path) -> Path:
    return Path(output).joinpath("summary.txt")


def load_summary(output: str | Path) -> str:
    return get_summary_path(output).read_text()


def dump_summary(output: str | Path, summary: str) -> None:
    get_summary_path(output).write_text(summary)


def get_predictions_path(output: str | Path) -> Path:
    return Path(output) / "predictions.npz"


def load_predictions(output: str | Path) -> dict[PartKey, np.ndarray]:
    path = get_predictions_path(output)
    assert path.exists(), f"The prediction file {path} does not exist"
    x = np.load(path)
    return {key: x[key] for key in x}


def dump_predictions(
    output: str | Path, predictions: dict[PartKey, np.ndarray]
) -> None:
    np.savez(get_predictions_path(output), allow_pickle=True, **predictions)


def get_stats_path(output: str | Path) -> Path:
    return Path(output) / "stats.npz"


def load_stats(output: str | Path) -> dict[str, np.ndarray]:
    path = get_stats_path(output)
    assert path.exists(), f"The stats file {path} does not exist"
    return dict(np.load(path))


def dump_stats(output: str | Path, stats: dict[str, np.ndarray]) -> None:
    np.savez(get_stats_path(output), allow_pickle=True, **stats)


def get_checkpoint_path(output: str | Path) -> Path:
    return Path(output) / "checkpoint.pt"


def load_checkpoint(output: str | Path, **kwargs) -> Any:
    return _torch().load(get_checkpoint_path(output), weights_only=False, **kwargs)


def dump_checkpoint(output: str | Path, checkpoint: Any, **kwargs) -> None:
    _torch().save(checkpoint, get_checkpoint_path(output), **kwargs)


def get_byproducts_path(output: str | Path) -> Path:
    return Path(output) / "byproducts.pt"


def load_byproducts(output: str | Path, **kwargs) -> dict[str, Any]:
    return _torch().load(get_byproducts_path(output), weights_only=False, **kwargs)


def dump_byproducts(output: str | Path, byproducts: dict[str, Any], **kwargs) -> None:
    _torch().save(byproducts, get_byproducts_path(output), **kwargs)


def remove_tracked_files(output: str | Path) -> None:
    """Remove files that are tracked by VCS."""
    get_report_path(output).unlink(missing_ok=True)


# ==================================================================================
# Printing
# ==================================================================================
try:
    _TERMINAL_SIZE = os.get_terminal_size().columns
except OSError:
    # Jupyter
    _TERMINAL_SIZE = 80
_SEPARATOR = "─" * _TERMINAL_SIZE


def print_sep():
    print(_SEPARATOR)


def print_config(config: dict) -> None:
    print("\nConfig")
    pprint(config, sort_dicts=False)


def print_summary(output: str | Path, *, newline: bool = True) -> None:
    lines = load_summary(output).splitlines()
    width = max(map(len, lines))
    hline = "─" * (width + 2)
    if newline:
        print()
    print(
        f"  {try_get_relative_path(output)}"
        f" ({'done' if is_done(output) else 'running'})"
    )
    print(
        "\n".join(
            [
                f"╭{hline}╮",
                *(f"│ {line}{' ' * (width - len(line))} │" for line in lines),
                f"╰{hline}╯",
            ]
        )
    )


# ==================================================================================
# DDP
# ==================================================================================
def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_ddp() -> bool:
    return "RANK" in os.environ


def is_master_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_ddp():
        _torch().distributed.barrier()  # type: ignore


def broadcast_bool(value: bool = False) -> bool:
    if not is_ddp():
        return value
    torch = _torch()
    t = torch.tensor([value], dtype=torch.bool, device=f"cuda:{get_local_rank()}")
    torch.distributed.broadcast(t, src=0)  # type: ignore
    return t.item()  # type: ignore


def allreduce_float(value: float, mode: str = "mean") -> float:
    if not is_ddp():
        return value
    torch = _torch()
    t = torch.tensor([value], dtype=torch.float64, device=f"cuda:{get_local_rank()}")
    match mode:
        case "mean":
            torch.distributed.all_reduce(t)  # type: ignore
            return t.item() / get_world_size()
        case _:
            raise ValueError(f"Unknown {mode=}")


def configure_ddp(backend="nccl", timeout_minutes: int = 10) -> None:
    _torch().distributed.init_process_group(  # type: ignore
        backend,
        timeout=datetime.timedelta(minutes=timeout_minutes),
        rank=get_rank(),
    )
    barrier()


# ==================================================================================
# CUDA
# ==================================================================================
def get_device(rank: int | None = None):  # -> torch.device
    torch = _torch()
    rank = get_local_rank() if rank is None else rank
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def is_dataparallel_available() -> bool:
    torch = _torch()
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and "CUDA_VISIBLE_DEVICES" in os.environ
    )


def get_gpu_names() -> list[str]:
    return [
        _torch().cuda.get_device_name(i) for i in range(_torch().cuda.device_count())
    ]


def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, _torch().cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            "CUDA out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDA error: out of memory",
        ]
    )


class OutOfMemoryException(Exception):
    """
    Exception to wrap Out-Of-Memory errors encountered during experiments.
    Stores the original exception for further inspection.
    """

    def __init__(self, err: RuntimeError):
        super().__init__(str(err))
        self.err = err

    def __str__(self):
        return str(self.err)

    def __repr__(self):
        return f"<OutOfMemoryException(err={self.err})>"


def catch_oom_exception():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as err:
                if is_oom_exception(err):
                    raise OutOfMemoryException(err)
                else:
                    raise

        return wrapper

    return decorator


def is_failed_trial(study: Study, index: int = -1) -> bool:
    trial = study.get_trials(deepcopy=False)[index]
    return trial.state == TrialState.FAIL


# ==================================================================================
# Other
# ==================================================================================
def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def random_float(a: float, b: float) -> float:
    assert a <= b
    return a + (b - a) * np.random.rand()


def make_seed(base: int, *keys: int | str) -> int:
    """Derive a deterministic seed from a base seed and named keys.

    Uses SHA-256 hashing to map (base, *keys) to a non-negative integer.
    Different key tuples produce different seeds with overwhelming probability,
    so callers don't need to reason about overlapping arithmetic ranges.

    Examples:
        >>> make_seed(42, "eval", 0, 3)          # eval dataset, rank=0, idx=3
        >>> make_seed(42, "main", 1, 100)        # main RNG, rank=1, step=100
        >>> make_seed(42, "sampler", 50)         # sampler workers, step=50
        >>> make_seed(42, "sampler", 50) + 3     # sampler worker_id=3
    """
    data = f"{base}:{':'.join(map(str, keys))}"
    return int(hashlib.sha256(data.encode()).hexdigest()[:8], 16)


def configure_logging(enqueue: bool = True):
    logger.remove()
    rank = get_rank()
    logger.add(
        sys.stderr,
        format=f"<level>[{{level}} ({rank=} {{time:HH:mm:ss}})] {{message}}</level>",
        enqueue=enqueue,
    )


def configure_torch(deterministic: bool = True):
    torch = _torch()
    torch.set_num_threads(1)
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def init(torch: bool = True, /):
    if Path.cwd() != env.get_project_dir():
        raise RuntimeError("The code must be run from the project root")
    configure_logging()
    if torch:
        configure_torch()


def try_get_relative_path(path: str | Path) -> Path:
    path = Path(path)
    project_dir = env.get_project_dir()
    return path.relative_to(project_dir) if project_dir in path.parents else path


def jsonify(value):
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    elif isinstance(value, list):
        return [jsonify(x) for x in value]
    elif isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    else:
        return f"<nonserializable (type={type(value)})>"


def are_valid_predictions(predictions: dict) -> bool:
    # predictions: dict[PartKey, np.ndarray]
    assert all(isinstance(x, np.ndarray) for x in predictions.values())
    return all(np.isfinite(x).all() for x in predictions.values())


def _flatten_dict(d: dict, key_prefix: str, result: dict) -> None:
    for k, v in d.items():
        new_k = f"{key_prefix}.{k}" if key_prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, new_k, result)
        else:
            if result.setdefault(new_k, v) is not v:
                RuntimeError(
                    "Different parts of the dictionary resulted"
                    f' in the same flat key "{new_k}"'
                )


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    flat_d: dict[str, Any] = {}
    _flatten_dict(d, "", flat_d)
    return flat_d


def import_(qualname: str) -> Any:
    """
    Examples:

    >>> import_('bin.model.main')
    """
    try:
        module, name = qualname.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)
    except Exception as err:
        raise ValueError(f'Cannot import "{qualname}"') from err


def get_function_full_name(function: Callable) -> str:
    """
    Examples:

    >>> # In the script bin/model.py
    >>> get_function_full_name(main) == 'bin.model.main'

    >>> # In the script a/b/c/foo.py
    >>> assert get_function_full_name(main) == 'a.b.c.foo.main'
    """
    module = inspect.getmodule(function)
    assert module is not None, "Failed to locate the module of the function."

    module_path = getattr(module, "__file__", None)
    assert module_path is not None, (
        "Failed to locate the module of the function."
        " This can happen if the code is running in a Jupyter notebook."
    )

    module_path = Path(module_path).resolve()
    project_dir = env.get_project_dir()
    assert project_dir in module_path.parents, (
        "The module of the function must be located within the project directory: "
        f" {project_dir}"
    )

    module_full_name = str(
        module_path.relative_to(project_dir).with_suffix("")
    ).replace("/", ".")
    return f"{module_full_name}.{function.__name__}"


DATA_PARTS: list[PartKey] = ["train", "val", "test"]


def print_metrics(loss: float, metrics: dict) -> None:
    def scale_metric(x: float) -> float:
        ax = abs(x)
        if ax > 1000:
            x /= 10 ** (len(str(int(ax))) - 3)
        elif 0 < ax < 0.01:
            while ax < 1:
                x *= 10
                ax *= 10
        return x

    print(
        f"(val) {scale_metric(metrics['val']['score']):.3f}"
        + (
            f" (test) {scale_metric(metrics['test']['score']):.3f}"
            if "test" in metrics
            else ""
        )
        + (
            f" (train) {scale_metric(metrics['train']['score']):.3f}"
            if "train" in metrics and "score" in metrics["train"]
            else ""
        )
        + f" (loss) {loss:.5f}"
    )


def get_default_metrics() -> dict[PartKey, dict[str, float]]:
    return {part: {"score": WORST_SCORE} for part in DATA_PARTS}


# ==================================================================================
# Optuna
# ==================================================================================
def dsp_log_prior(kernel_params: Any) -> Any:
    """Dimensionality-Scaled Prior for `GPSampler` (Hvarfner et al., ICML 2024)."""
    import math

    torch = _torch()

    # --- Lengthscale prior: LogNormal(mu, sigma) ---
    # mu scales with D so the median lengthscale grows as sqrt(D),
    # matching the expected inter-point distance in a D-dimensional unit hypercube.
    LENGTHSCALE_MU_BASE = math.sqrt(2)  # base location from the paper
    LENGTHSCALE_SIGMA = math.sqrt(3)  # wide spread to allow optimizer flexibility

    # --- Kernel scale prior ---
    # The paper fixes output scale to 1 (Y is standardized, so no learned scale needed).
    # We approximate this with a tight quadratic penalty
    # instead of patching the optimizer.
    KERNEL_SCALE_TARGET = 1.0
    KERNEL_SCALE_PENALTY = 100.0  # high penalty ≈ effectively fixing the value

    # --- Noise prior: LogNormal(mu, sigma) ---
    # Favors small noise (median ≈ exp(-4) ≈ 0.018) since Y is standardized.
    NOISE_MU = -4.0
    NOISE_SIGMA = 1.0

    D = len(kernel_params.inverse_squared_lengthscales)
    lengthscales = 1.0 / torch.sqrt(kernel_params.inverse_squared_lengthscales)

    mu = LENGTHSCALE_MU_BASE + math.log(D) * 0.5
    log_l = torch.log(lengthscales)
    lengthscale_prior = (-log_l - 0.5 * ((log_l - mu) / LENGTHSCALE_SIGMA) ** 2).sum()

    kernel_scale_prior = (
        -KERNEL_SCALE_PENALTY * (kernel_params.kernel_scale - KERNEL_SCALE_TARGET) ** 2
    )

    log_noise = torch.log(kernel_params.noise_var)
    noise_prior = -log_noise - 0.5 * ((log_noise - NOISE_MU) / NOISE_SIGMA) ** 2

    return lengthscale_prior + kernel_scale_prior + noise_prior
