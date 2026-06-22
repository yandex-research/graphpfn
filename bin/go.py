import argparse
import sys
from pathlib import Path
from typing import cast

import torch

import bin.evaluate
import bin.tune
import lib

_DEFAULT_N_SEEDS = 10


def main(
    config: str | Path,  # A tuning or evaluation config.
    n_seeds: int = _DEFAULT_N_SEEDS,
    *,
    continue_: bool = False,
    force: bool = False,
    tiny: bool = False,
    profiler: torch.profiler.profile | None = None,
):
    config = Path(config).resolve()

    if config.name == "tuning.toml":
        evaluation_config_path = config.with_stem("evaluation")
        tuning_output = config.with_suffix("")

        if evaluation_config_path.exists() and not force:
            assert lib.is_done(tuning_output)

        elif (
            tuning_output.exists()
            and not lib.is_done(tuning_output)
            and not continue_
            and not force
        ):
            raise RuntimeError(
                f"An unfinished tuning experiment is found at {tuning_output}."
                " Consider using the --continue flag"
                " or removing the unfinished experiment first"
            )

        else:
            tuning_config = cast(bin.tune.Config, lib.load_config(config))
            bin.tune.main(
                tuning_config, tuning_output, continue_=continue_, force=force
            )
            tuning_report = lib.load_report(tuning_output)
            assert tuning_report is not None
            evaluation_config: bin.evaluate.Config = {
                "function": tuning_config["function"],
                "n_seeds": n_seeds,
                "base_config": tuning_report["best"]["config"],
            }
            evaluation_config["base_config"].pop("seed", None)
            lib.dump_config(evaluation_config_path, evaluation_config, force=force)  # type: ignore

    elif config.name == "evaluation.toml":
        evaluation_config_path = config

    elif config.name == "pretrain.toml":
        pretrain_config = lib.load_config(config)
        function = lib.import_(pretrain_config["function"])
        function(
            pretrain_config["base_config"],
            force=force,
            continue_=continue_,
            tiny=tiny,
            output=config.with_suffix(""),
            profiler=profiler,
        )
        return

    else:
        raise ValueError(
            'The config name must be either "tuning.toml", "evaluation.toml"'
            f' or "evaluation.toml". However: {config.name=}'
        )

    bin.evaluate.main(evaluation_config_path, continue_=continue_, force=force)


if __name__ == "__main__":
    lib.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--n_seeds", type=int, default=_DEFAULT_N_SEEDS)
    parser.add_argument("--continue", action="store_true", dest="continue_")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tiny", action="store_true")

    main(**vars(parser.parse_args(sys.argv[1:])))
