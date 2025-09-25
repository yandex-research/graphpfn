import argparse
import datetime
import delu
import enum
import functools
import importlib
import inspect
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, TypeVar, cast
from typing_extensions import NotRequired, TypedDict  # noqa: UP035

import numpy as np
import tomli_w
import yaml
from loguru import logger
from optuna import Study
from optuna.trial import TrialState

# NOTE: this is internal infrastructure, ignore it
try:
    from dev.infra.util import save_snapshot, load_snapshot, init_exp_tracker

    INTERNAL_INFRA = True
except ImportError:
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
WORST_SCORE = -999999.0  # TODO: replace it with None?


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
# `Experiment` class
# ==================================================================================
# The following class largely duplicates the "`main` function" and
# "IO for the output directory" sections below. However, the sections below were not
# created to handle long pretrain runs, so we've needed to rewrite them, and it was
# easier to create new class from the scratch.
# TODO: refactor to have no duplication
T = TypeVar("T")


class Checkpoint(TypedDict):
    run_uid: NotRequired[str]  # internal infrastructure, ignore it
    step: int
    report: JSONDict
    model: dict[str, Any]
    optimizer: dict[str, Any]
    lr_scheduler: dict[str, Any]
    random_state: dict[str, Any]
    timer: delu.tools.Timer
    extra: NotRequired[dict[str, Any]]


# TODO: ideally, support all features from original reports
class Experiment:
    def __init__(self) -> None:
        self.finished = False
        self.init_ddp()
        self.init_loguru()

        logger.info(f"Launching on {_torch().cuda.device_count()} devices")
        logger.info(f"{INTERNAL_INFRA=}")

    def init_ddp(self, backend="nccl"):
        self.ddp = "RANK" in os.environ
        self.rank = int(os.environ.get("RANK", 0))
        self.master_process = (not self.ddp) | (self.rank == 0)

        if self.ddp:
            self.device_id = os.environ["LOCAL_RANK"]
            _torch().distributed.init_process_group(backend, rank=self.rank)
            self.sync()
        else:
            self.device_id = 0

    def init_loguru(self, log_master_only: bool = True, level: str = "INFO") -> None:
        logger.remove()
        if self.master_process or not log_master_only:
            logger.add(
                sys.stderr,
                format="<level>{message}</level>",
                level=level,
                enqueue=(not log_master_only),
            )

    def check[T](
        self, config, output: None | str | Path, *, config_type: type[T] = dict
    ) -> tuple[T, Path]:
        """Load the config and infer the path to the output directory."""
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
                "If config is a dictionary, "
                "then the `output` directory must be provided."
            )
        output = Path(output).resolve()

        # >>> Check the config.
        if config_type is dict:
            pass
        elif (
            # If all conditions are True, config_type is assumed to be a TypedDict.
            issubclass(config_type, dict)
            and hasattr(config_type, "__required_keys__")
            and hasattr(config_type, "__optional_keys__")
        ):
            # >>> Check the keys.
            presented_keys = frozenset(config)
            required_keys = config_type.__required_keys__  # type: ignore
            optional_keys = config_type.__optional_keys__  # type: ignore
            assert presented_keys >= required_keys, (
                "The config is missing the following required keys:"
                f" {', '.join(required_keys - presented_keys)}"
            )
            assert set(config) <= (required_keys | optional_keys), (
                "The config has unknown keys:"
                f" {', '.join(presented_keys - required_keys - optional_keys)}"
            )

        self.config = cast(T, config)
        self.output = output
        return self.config, self.output

    def start(
        self,
        main_fn: Callable,
        *,
        continue_: bool = False,
        force: bool = False,
        exp_tracker: bool = True,
    ) -> bool:
        """Checks if caller should continue execution and inits (internal) exp tracker

        Returns:
            True if the caller should continue the execution.
            False if the caller should immediately return.
        """
        self.sync()

        should_start = _torch().tensor([True], device=f"cuda:{self.device_id}")
        if self.master_process:
            should_start[0] = self.check_start(
                main_fn, continue_=continue_, force=force
            )
        if self.ddp:
            _torch().distributed.broadcast(should_start, src=0)
        should_start = should_start.item()

        if should_start:
            checkpoint = self.load_checkpoint() if continue_ else None
            if self.master_process and exp_tracker:
                self.exp_tracker = init_exp_tracker(
                    self.config, self.output, checkpoint
                )
                print_config(self.config)
                print()
            else:
                self.exp_tracker = None
            seed = self.config["seed"] + self.rank
            if checkpoint is not None:
                seed += 8 * checkpoint["step"]
            delu.random.seed(seed)
        return should_start

    def check_start(
        self,
        main_fn: Callable,
        *,
        continue_: bool = False,
        force: bool = False,
    ) -> bool:
        """Create the output directory (if missing).

        Returns:
            True if the caller should continue the execution.
            False if the caller should immediately return.
        """
        output = Path(self.output).resolve()

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
                logger.warning("The output already exists!")
                return False
            elif is_done(output):
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

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.sync()
        if self.master_process:
            if self.exp_tracker:
                info = {
                    k: v["score"] for k, v in checkpoint["report"]["metrics"].items()
                } | {"lr": checkpoint["report"]["lr"][0]}
                logger.info(f"{info=}")
                self.exp_tracker.log(
                    info,
                    step=checkpoint["step"],
                    flush=True,
                )
            _torch().save(checkpoint, self.output / "checkpoint.ckpt")
            if INTERNAL_INFRA:
                save_snapshot(self.output)
        self.sync()

    def load_checkpoint(self) -> Checkpoint | None:
        self.sync()
        if self.master_process:
            if INTERNAL_INFRA:
                load_snapshot(self.output)
        self.sync()
        path = self.output / "checkpoint.ckpt"
        checkpoint = _torch().load(path) if path.exists() else None
        self.sync()
        return checkpoint

    @property
    def run_uid(self) -> str | None:
        if self.exp_tracker:
            return self.exp_tracker.uid
        else:
            return None

    def sync(self):
        if self.ddp:
            _torch().distributed.barrier()

    def finish(self) -> None:
        _mark_as_done(self.output)
        self.finished = True

    def __del__(self):
        logger.complete()
        # TODO: finish exp_tracker
        self.sync()
        if self.ddp:
            _torch().distributed.destroy_process_group()


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
    if config_type is dict:
        pass
    elif (
        # If all conditions are True, config_type is assumed to be a TypedDict.
        issubclass(config_type, dict)
        and hasattr(config_type, "__required_keys__")
        and hasattr(config_type, "__optional_keys__")
    ):
        # >>> Check the keys.
        presented_keys = frozenset(config)
        required_keys = config_type.__required_keys__  # type: ignore
        optional_keys = config_type.__optional_keys__  # type: ignore
        assert presented_keys >= required_keys, (
            "The config is missing the following required keys:"
            f" {', '.join(required_keys - presented_keys)}"
        )
        assert set(config) <= (required_keys | optional_keys), (
            "The config has unknown keys:"
            f" {', '.join(presented_keys - required_keys - optional_keys)}"
        )

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
        summary["time"] = report["time"]
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
        # if function == 'bin.evaluate.main':  # TODO: fix this
        #     summary['time-mean'] = str(
        #         datetime.timedelta(
        #             seconds=statistics.mean(x['time'] for x in reports)
        #         )
        #     )
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


_LAST_SNAPSHOT_TIME = None


def backup(output: Path) -> None:
    """A function for the internal infrastructure, ignore it."""
    if env.is_local():
        return

    project_dir = env.get_project_dir()
    if project_dir not in output.resolve().parents:
        # The output is outside of the project directory
        # (e.g. output may be a temporary directory).
        return

    relative_output = output.relative_to(project_dir)
    for new_project_dir in [os.environ["TMP_OUTPUT_PATH"], os.environ["SNAPSHOT_PATH"]]:
        new_output = new_project_dir / relative_output
        with tempfile.TemporaryDirectory() as tmp:
            # Copy the output before removing the existing backup.
            tmp_output = Path(tmp) / new_output.name
            shutil.copytree(output, tmp_output)
            if new_output.exists():
                # Remove the old backup.
                shutil.rmtree(new_output)
            new_output.parent.mkdir(exist_ok=True, parents=True)
            tmp_output.rename(new_output)
        if output.with_suffix(".toml").exists():
            # Some scripts (e.g. go.py) produce configs during runs,
            # so configs is a part of the result.
            shutil.copyfile(
                output.with_suffix(".toml"), new_output.with_suffix(".toml")
            )

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        logger.info("The snapshot was saved")


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


def load_predictions(output: str | Path) -> dict[PartKey, np.ndarray]:
    path = Path(output) / "predictions.npz"
    assert path.exists(), f"The prediction file {path} does not exist"
    x = np.load(path)
    return {key: x[key] for key in x}


def dump_predictions(
    output: str | Path, predictions: dict[PartKey, np.ndarray]
) -> None:
    np.savez(Path(output) / "predictions.npz", **predictions)  # type: ignore[arg-type]


def get_checkpoint_path(output: str | Path) -> Path:
    return Path(output) / "checkpoint.pt"


def load_checkpoint(output: str | Path, **kwargs) -> Any:
    return _torch().load(get_checkpoint_path(output), weights_only=False, **kwargs)


def dump_checkpoint(output: str | Path, checkpoint: Any, **kwargs) -> None:
    _torch().save(checkpoint, get_checkpoint_path(output), **kwargs)


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
# CUDA
# ==================================================================================
def get_device(rank: int = 0):  # -> torch.device
    torch = _torch()
    return torch.device(
        f"cuda:{rank}"
        if torch.cuda.is_available()
        # else 'mps:0'
        # if torch.mps.is_available()
        else "cpu"
    )


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
        return "OutOfMemoryException"

    def __repr__(self):
        return f"<OutOfMemoryException(err={self.err})>"


def catch_oom_decorator():
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
def configure_logging():
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>")


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


def backup_output(output: Path) -> None:
    """
    A function for the internal infrastructure, ignore it.
    """
    backup_dir = os.environ.get("TMP_OUTPUT_PATH")
    snapshot_dir = os.environ.get("SNAPSHOT_PATH")
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output.relative_to(PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output = dir_ / relative_output_dir
        prev_backup_output = new_output.with_name(new_output.name + "_prev")
        new_output.parent.mkdir(exist_ok=True, parents=True)
        if new_output.exists():
            new_output.rename(prev_backup_output)
        shutil.copytree(output, new_output)
        # the case for evaluate.py which automatically creates configs
        if output.with_suffix(".toml").exists():
            shutil.copyfile(
                output.with_suffix(".toml"), new_output.with_suffix(".toml")
            )
        if prev_backup_output.exists():
            shutil.rmtree(prev_backup_output)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print("The snapshot was saved!")
