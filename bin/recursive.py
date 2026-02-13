import argparse
import sys
from pathlib import Path

import delu
from loguru import logger

import bin.go
import lib


def main(
    paths: list[str | Path],
    mode: str = "tuning",
    *,
    dry: bool = False,
    continue_: bool = False,
    force: bool = False,
    n_seeds: int = 10,
):
    configs = list()
    for path in paths:
        path = Path(path).resolve()
        configs.extend(list(path.rglob(f"{mode}.toml")))
    configs = sorted(set(configs))
    print(f"Found {len(configs)} {mode}.toml configs at {paths}\n")

    if dry:
        return

    for config in configs:
        try:
            bin.go.main(
                config,
                continue_=continue_,
                force=force,
                n_seeds=n_seeds,
            )
        except Exception as e:
            logger.warning(e)
        delu.cuda.free_memory()


if __name__ == "__main__":
    lib.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=str, nargs="+")
    parser.add_argument("--mode", type=str, default="tuning")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--continue", action="store_true", dest="continue_")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n_seeds", type=int, default=10)

    main(**vars(parser.parse_args(sys.argv[1:])))
