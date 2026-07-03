import shutil
from pathlib import Path
from typing import Any, TypedDict

import delu
import numpy as np
import scipy.special
from loguru import logger

import lib
import lib.data
import lib.graph.data


class Config(TypedDict):
    ensemble_size: int


def _is_graph_config(config: dict):
    # NOTE: this is a quick hack to handle graph tasks
    # that require more information than standard ones
    return "setting" in config["data"]


def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
    continue_: bool = False,
) -> None | lib.JSONDict:
    # >>> Start
    _config = config
    config, output = lib.check(config, output, config_type=Config)
    if isinstance(_config, str | Path):
        assert Path(_config).name == f"ensemble-{config['ensemble_size']}.toml"
    del _config

    evaluation_output = output.with_name("evaluation")
    assert lib.is_done(evaluation_output), (
        "Cannot evaluate ensembles, because the evaluation of single models"
        " is either missing or has not yet finished"
        f" (the expected evaluation directory: {evaluation_output})"
    )
    if not lib.start(main, output, continue_=continue_, force=force):
        return None

    lib.print_config(config)  # type: ignore
    report = lib.create_report(
        function=main,
        config=config,  # type: ignore
        output=output,
        continue_=continue_,
    )

    single_reports = lib.load_report(evaluation_output)["reports"]
    first_single_report = single_reports[0]

    # NOTE: graphs tasks require more information
    task = (
        lib.graph.data.GraphTask.from_dir(**first_single_report["config"]["data"])
        if _is_graph_config(first_single_report["config"])
        else lib.data.Task.from_dir(first_single_report["config"]["data"]["path"])
    )
    timer = delu.tools.Timer()

    report.setdefault("reports", [])
    timer.run()
    for ensemble_id in range(len(single_reports) // config["ensemble_size"]):
        if ensemble_id < len(report["reports"]):
            continue

        next_output = output / str(ensemble_id)
        if next_output.exists():
            logger.warning(f"Removing the incomplete output {output}")
            shutil.rmtree(output)

        next_report: dict[str, Any] = {
            "single-model-info": {
                "function": first_single_report["function"],
                "config": {
                    "data": {"path": first_single_report["config"]["data"]["path"]}
                },
            },
            "seeds": list(
                range(
                    ensemble_id * config["ensemble_size"],
                    (ensemble_id + 1) * config["ensemble_size"],
                )
            ),
            "prediction_type": "labels" if task.is_regression else "probs",
        }

        single_predictions = [
            lib.load_predictions(evaluation_output / str(x))
            for x in next_report["seeds"]
        ]
        predictions = {}
        for part in ["train", "val", "test"]:
            stacked_predictions = np.stack([x[part] for x in single_predictions])
            if task.is_binclass:
                # Predictions for binary classifications are expected to contain
                # only the probability of the positive label.
                assert stacked_predictions.ndim == 2
                if first_single_report["prediction_type"] == "logits":
                    stacked_predictions = scipy.special.expit(stacked_predictions)
            elif task.is_multiclass:
                assert stacked_predictions.ndim == 3
                if first_single_report["prediction_type"] == "logits":
                    stacked_predictions = scipy.special.softmax(stacked_predictions, -1)
            else:
                assert task.is_regression
                assert stacked_predictions.ndim == 2
            predictions[part] = stacked_predictions.mean(0)

        next_report["metrics"] = task.calculate_metrics(
            predictions, next_report["prediction_type"]
        )
        next_output.mkdir()
        lib.dump_predictions(next_output, predictions)
        lib.dump_summary(next_output, lib.summarize(next_report))

        report["reports"].append(next_report)
        lib.dump_report(output, report)

    report["time"] = timer.elapsed()
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.init(False)
    lib.run(main)
