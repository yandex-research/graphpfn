"""
Script for generating info.yaml for graph collections
"""
import argparse
import numpy as np
import yaml
import typing as tp
from pathlib import Path
from tqdm.auto import tqdm


def main(path: Path | str) -> None:
    path = Path(path)
    items = list(path.rglob("categorical_features.npy"))

    info: list[dict[str, tp.Any]] = []

    for item in tqdm(items):
        graph_dir_path = item.parent
        edge_list = np.load(graph_dir_path / "edgelist.npy")
        max_id = np.max(edge_list)
        num_nodes = np.load(item).shape[0]
        assert num_nodes == max_id+1, (num_nodes, max_id)
        graph_info = {
            "num_nodes": num_nodes,
            "path": str(graph_dir_path),
        }
        info.append(graph_info)

    with open(path / "info.yaml", "w") as file:
        yaml.safe_dump(info, file, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/graphpfn-graphs")
    args = parser.parse_args()

    main(**vars(args))