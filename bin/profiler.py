import argparse
import sys
from functools import partial
from pathlib import Path

import torch

import bin.go
import lib

_DEFAULT_N_SEEDS = 10


def handle_on_trace_ready(
    profiler: torch.profiler.profile,
    trace_path: str | None = None,
) -> None:
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)


def main(
    config: str | Path,
    profile_path: str,
    n_seeds: int = _DEFAULT_N_SEEDS,
    *,
    continue_: bool = False,
    force: bool = False,
):
    config = Path(config)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=partial(handle_on_trace_ready, trace_path=profile_path),
    ) as profiler:
        bin.go.main(
            config,
            continue_=continue_,
            force=force,
            n_seeds=n_seeds,
            profiler=profiler,
        )


if __name__ == "__main__":
    lib.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--profile_path", type=str, default="local/trace.json")
    parser.add_argument("--n_seeds", type=int, default=_DEFAULT_N_SEEDS)
    parser.add_argument("--continue", action="store_true", dest="continue_")
    parser.add_argument("--force", action="store_true")

    main(**vars(parser.parse_args(sys.argv[1:])))
