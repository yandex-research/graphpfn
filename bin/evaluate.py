import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import delu
from loguru import logger

import lib


class Config(TypedDict):
    function: str
    n_seeds: int
    base_config: dict[str, Any]


def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
    continue_: bool = False,
) -> None | lib.JSONDict:
    if isinstance(config, str | Path):
        assert Path(config).name == "evaluation.toml"

    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    assert "seed" not in config["base_config"]
    if not lib.start(main, output, continue_=continue_, force=force):
        return None

    lib.print_config(config)  # type: ignore
    report = lib.create_report(
        function=main,
        config=config,  # type: ignore
        output=output,
        continue_=continue_,
    )

    function = lib.import_(config["function"])
    report.setdefault("reports", [])
    timer = delu.tools.Timer()
    timer.run()
    for seed in range(config["n_seeds"]):
        if seed < len(report["reports"]):
            assert report["reports"][seed]["config"]["seed"] == seed
            continue

        next_config: dict[str, Any] = {"seed": seed} | config["base_config"]
        if "catboost" in config["function"]:
            if next_config["model"]["task_type"] == "GPU":
                next_config["model"]["task_type"] = (
                    "CPU"  # This is crucial for good results.
                )
                next_config["model"]["thread_count"] = max(
                    next_config["model"].get("thread_count", 1), 4
                )

        next_output = output / str(seed)
        if next_output.exists():
            logger.warning(f"Removing the incomplete output {output}")
            shutil.rmtree(output)

        with tempfile.TemporaryDirectory(suffix=f"_evaluation_{seed}") as tmp:
            tmp_output = Path(tmp) / "output"
            next_report = function(next_config, tmp_output)
            lib.remove_tracked_files(tmp_output)
            tmp_output.rename(next_output)

        report["reports"].append(next_report)
        lib.dump_report(output, report)
        lib.dump_summary(output, lib.summarize(report))
        lib.backup(output)

        if seed + 1 < config["n_seeds"]:
            lib.print_summary(output)

    report["time"] = timer.elapsed()
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.init()
    lib.run(main)
