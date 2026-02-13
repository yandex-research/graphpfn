import argparse
import os
import tomllib
from datetime import datetime
from pathlib import Path
from pprint import pprint

import delu
import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

import lib
from lib.graph.data import GraphDataset, build_dataset
from lib.graphpfn.prior.wrapper import GraphPrior
from lib.util import KWArgs

# >>> Properties computation


# Unbiased homophily
def compute_homophily(dataset: GraphDataset, labels: np.ndarray) -> float | None:
    if dataset.task.is_regression:
        return None

    src, dst = dataset.data["graph"].edges()
    edges = torch.stack([src, dst], dim=-1).cpu().numpy()

    labels_1 = labels[edges[:, 0]]
    labels_2 = labels[edges[:, 1]]
    edge_mask = ~(np.isnan(labels_1) | np.isnan(labels_2))
    edges = edges[edge_mask]

    intraclass_edges_mask = labels[edges[:, 0]] == labels[edges[:, 1]]
    intraclass_edges = edges[intraclass_edges_mask]
    intraclass_edges_labels = labels[intraclass_edges[:, 0]]

    _, counts = np.unique(intraclass_edges_labels, return_counts=True)

    c_diag = counts / len(edges)

    c_diag_sqrt = np.sqrt(c_diag)
    c_diag_sqrt_sum_sq = c_diag_sqrt.sum() ** 2

    h_unb = (c_diag_sqrt_sum_sq - 1) / (c_diag_sqrt_sum_sq + 1 - 2 * c_diag.sum())

    if h_unb <= -1 + 1e-6:
        return None

    return h_unb


# Pearson correlation coefficient
def compute_assortativity(dataset: GraphDataset, labels: np.ndarray) -> float | None:
    if dataset.task.is_classification:
        return None

    src, dst = dataset.data["graph"].edges()
    edges = torch.stack([src, dst], dim=-1).cpu().numpy()

    labels_1 = labels[edges[:, 0]]
    labels_2 = labels[edges[:, 1]]
    edge_mask = ~(np.isnan(labels_1) | np.isnan(labels_2))
    edges = edges[edge_mask]

    labels_1 = labels[edges[:, 0]]
    labels_2 = labels[edges[:, 1]]
    h = np.corrcoef(labels_1, labels_2)
    assert tuple(h.shape) == (2, 2)
    return h[0, 1].item()


def compute_node_label_informativeness(
    dataset: GraphDataset,
    labels: np.ndarray,
) -> float | None:
    if dataset.task.is_regression:
        return None

    if np.isnan(labels).any():
        return None

    graph = dataset.data["graph"]
    return dgl.node_label_informativeness(
        graph, y=torch.tensor(labels, dtype=torch.int32)
    )


def compute_degree_powerlaw_r2(
    dataset: GraphDataset,
) -> float:
    degrees = dataset.data["graph"].in_degrees()
    values, counts = torch.unique(degrees, return_counts=True, sorted=True)
    values, counts = values.flip([0]), counts.flip([0])

    probs = counts / counts.sum()
    cum_probs = torch.cumsum(probs, dim=0)

    x = torch.log(values).cpu().numpy()
    y = torch.log(cum_probs).cpu().numpy()

    linreg = LinearRegression()
    linreg.fit(x.reshape(-1, 1), y)
    r_squared = float(linreg.score(x.reshape(-1, 1), y))

    return r_squared


def compute_clustering_coefficient(
    dataset: GraphDataset,
) -> float:
    degree = dataset.data["graph"].in_degrees()
    triples = (degree * (degree - 1) / 2).sum()

    adj = dataset.data["graph"].adj_external(scipy_fmt="csr")
    adj = sp.triu(adj, k=1)
    adj.eliminate_zeros()
    triangles = ((adj @ adj).multiply(adj)).sum()  # type: ignore

    return float(3 * triangles / triples)


def estimate_avg_pairwise_distance(
    dataset: GraphDataset,
    n_anchors: int = 100,
) -> float:
    n_anchors = min(dataset.size(), n_anchors)
    anchors = torch.tensor(np.random.choice(dataset.size(), n_anchors, replace=False))
    dist = dgl.shortest_dist(dataset.data["graph"], root=anchors, return_paths=False)
    avg_dist = dist.sum() / (dist.numel() - n_anchors)  # type: ignore
    return avg_dist.item()


# <<<


# >>> Tests


def test_homophily() -> None:
    ds = build_dataset(
        path="data/tolokers-2",
        setting="transductive",
        num_policy="noisy-quantile-normal",
        ratio_policy="noisy-quantile-uniform",
        cat_policy="ordinal",
        internal_split_name="RL",
    )
    h = compute_homophily(ds, ds.data["labels"])
    assert h is not None
    assert np.allclose(h, 0.10155749834851732), h


def test_assortativity() -> None:
    ds = build_dataset(
        path="data/artnet-views",
        setting="transductive",
        num_policy="noisy-quantile-normal",
        ratio_policy="noisy-quantile-uniform",
        cat_policy="ordinal",
        internal_split_name="RL",
    )
    h = compute_assortativity(ds, ds.data["labels"])
    assert h is not None
    assert np.allclose(h, 0.19281946541531367), h


def run_tests() -> None:
    test_homophily()
    test_assortativity()


# <<<


def analyze_dataset(dataset: GraphDataset) -> dict:
    return {
        "n_nodes": dataset.size(),
        "n_edges": dataset.n_edges // 2,
        "n_features": dataset.n_features,
        "avg_degree": dataset.n_edges / dataset.size(),
        "target_homophily": compute_homophily(dataset, dataset.data["labels"]),
        "target_assortativity": compute_assortativity(dataset, dataset.data["labels"]),
        # "node_label_informativeness": compute_node_label_informativeness(
        #     dataset, dataset.data["labels"]
        # ),
        "degree_powerlaw_r2": compute_degree_powerlaw_r2(dataset),
        "clustering_coefficient": compute_clustering_coefficient(dataset),
        "avg_pairwise_distance": estimate_avg_pairwise_distance(dataset),
    }


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "mean": table.apply(lambda x: np.mean(x[~np.isnan(x)])),
        "std": table.apply(lambda x: np.std(x[~np.isnan(x)])),  # type: ignore
        "min": table.apply(lambda x: np.min(x[~np.isnan(x)])),
        "Q25": table.apply(lambda x: np.quantile(x[~np.isnan(x)], 0.25)),
        "median": table.apply(lambda x: np.median(x[~np.isnan(x)])),
        "Q75": table.apply(lambda x: np.quantile(x[~np.isnan(x)], 0.75)),
        "max": table.apply(lambda x: np.max(x[~np.isnan(x)])),
    }
    return pd.DataFrame(summary).map(lambda x: f"{x:.2f}")


def plot_hist(
    array: np.ndarray,
    savepath: Path,
    n_bins: int = 50,
    title: str | None = None,
) -> None:
    plt.hist(array, bins=n_bins, density=True)
    if title is not None:
        plt.title(title)
    plt.savefig(savepath)
    plt.close()


def plot_hists(data: pd.DataFrame, basepath: Path, extension: str = "png") -> None:
    for col in data.columns:
        plot_hist(data.loc[:, col].values, basepath / f"{col}.{extension}", title=col)  # type: ignore


def analyze_prior(
    prior: GraphPrior,
    n_datasets: int,
    path: str | Path | None,
) -> pd.DataFrame:
    analysis_results = [analyze_dataset(next(prior)) for _ in tqdm(range(n_datasets))]
    analysis_results_df = pd.DataFrame(analysis_results)
    summary = summarize(analysis_results_df)
    if path is not None:
        path = Path(path) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path.mkdir(exist_ok=True, parents=True)
        analysis_results_df.to_csv(path / "data.csv")
        summary.to_csv(path / "summary.csv")
        plot_hists(analysis_results_df, path)
    return summary


def analyze_real_datasets(
    dataset_names: list[str] = [
        "tolokers-2",
        # "city-reviews",
        # "artnet-exp",
        # "hm-prices",
        # "avazu-ctr",
        # "city-roads-M",
        # "twitch-views",
        "artnet-views",
    ],
) -> None:
    results = dict()
    for dataset_name in dataset_names:
        dataset = build_dataset(
            path=f"data/{dataset_name}",
            setting="transductive",
            num_policy="noisy-quantile-normal",
            ratio_policy="noisy-quantile-uniform",
            cat_policy="ordinal",
            internal_split_name="RL",
        )
        results[dataset_name] = analyze_dataset(dataset)
    results = pd.DataFrame(results)
    results = results.map(lambda x: f"{x:.4f}" if x is not None else None)
    print(results.to_markdown(disable_numparse=True))


def get_prior_config(pretrain_config_path: str | Path) -> KWArgs:
    with open(pretrain_config_path, "rb") as file:
        pretrain_config = tomllib.load(file)
    pprint(pretrain_config["prior"])
    return pretrain_config["prior"]


def get_config() -> dict:
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrain_config_path", type=str)
    parser.add_argument("--n_datasets", type=int, default=1_000)
    parser.add_argument("--n_workers", type=int, required=False)
    args = parser.parse_args()
    config = vars(args)

    # Extra
    config["tests"] = False
    config["analyze_real_datasets"] = True
    config["prior"] = get_prior_config(config["pretrain_config_path"])

    n_workers = config.pop("n_workers", None)
    if n_workers is not None:
        config["prior"] |= {"n_workers": n_workers}

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("Launching on gpu")
        config["device"] = "cuda:0"
    else:
        print("Launching on cpu")
        config["device"] = "cpu"

    return config


if __name__ == "__main__":
    lib.configure_logging()

    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    delu.random.seed(42)

    config = get_config()

    if config["tests"]:
        run_tests()

    if config["analyze_real_datasets"]:
        analyze_real_datasets()

    prior = GraphPrior(
        **config["prior"],
        device=torch.device(config["device"]),
        verbose=False,
    )

    analysis_results = analyze_prior(
        prior,
        config["n_datasets"],
        path=(Path("local/analysis") / config["pretrain_config_path"]).parent,
    )
    print(f"{config["n_datasets"]=}")
    print(analysis_results.to_markdown(disable_numparse=True))
