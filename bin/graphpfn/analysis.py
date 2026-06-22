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
from lib.graph.data import GraphDataset
from lib.graphpfn.prior import GraphPriorSampler
from lib.graphpfn.prior.prior_typings import PriorDataset
from lib.util import TaskType

# >>> Properties computation


def compute_homophily(
    graph: dgl.DGLGraph,
    labels: np.ndarray,
    is_regression: bool,
    eps: float = 1e-6,
) -> float | None:
    if is_regression:
        return None

    src, dst = graph.edges()
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

    if h_unb <= -1 + eps:
        return None

    return h_unb


def compute_assortativity(
    graph: dgl.DGLGraph,
    labels: np.ndarray,
    is_classification: bool,
) -> float | None:
    if is_classification:
        return None

    src, dst = graph.edges()
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


def compute_degree_powerlaw_r2(graph: dgl.DGLGraph) -> float:
    degrees = graph.in_degrees()
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


def compute_clustering_coefficient(graph: dgl.DGLGraph) -> float:
    degree = graph.in_degrees()
    triples = (degree * (degree - 1) / 2).sum()

    adj = graph.adj_external(scipy_fmt="csr")
    adj = sp.triu(adj, k=1)
    adj.eliminate_zeros()
    triangles = ((adj @ adj).multiply(adj)).sum()  # type: ignore

    return float(3 * triangles / triples)


def estimate_avg_pairwise_distance(
    graph: dgl.DGLGraph,
    n_anchors: int = 100,
) -> float:
    n_nodes = graph.num_nodes()
    n_anchors = min(n_nodes, n_anchors)
    anchors = torch.tensor(np.random.choice(n_nodes, n_anchors, replace=False))
    dist = dgl.shortest_dist(graph, root=anchors, return_paths=False)
    avg_dist = dist.sum() / (dist.numel() - n_anchors)  # type: ignore
    return avg_dist.item()


# <<<


# >>> Tests


def test_homophily() -> None:
    ds = GraphDataset.from_dir(
        path="data/tolokers-2",
        setting="transductive",
        internal_split_name="RL",
    )
    h = compute_homophily(
        ds.data["graph"], ds.data["labels"], is_regression=ds.task.is_regression
    )
    assert h is not None
    assert np.allclose(h, 0.10155749834851732), h


def test_assortativity() -> None:
    ds = GraphDataset.from_dir(
        path="data/artnet-views",
        setting="transductive",
        internal_split_name="RL",
    )
    h = compute_assortativity(
        ds.data["graph"], ds.data["labels"], is_classification=ds.task.is_classification
    )
    assert h is not None
    assert np.allclose(h, 0.19281946541531367), h


def run_tests() -> None:
    test_homophily()
    test_assortativity()


# <<<


def analyze_prior_dataset(dataset: PriorDataset) -> dict:
    edges = dataset["edges"]
    n_nodes = dataset["features"].shape[0]
    graph = dgl.graph(
        (edges[0], edges[1]),
        num_nodes=n_nodes,
    )
    labels = dataset["labels"].cpu().numpy()
    n_edges = graph.num_edges()
    n_features = dataset["features"].shape[1]
    is_regression = dataset["task_type"] == TaskType.REGRESSION

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges // 2,
        "n_features": n_features,
        "avg_degree": n_edges / n_nodes,
        "target_homophily": compute_homophily(graph, labels, is_regression),
        "target_assortativity": compute_assortativity(graph, labels, not is_regression),
        "degree_powerlaw_r2": compute_degree_powerlaw_r2(graph),
        "clustering_coefficient": compute_clustering_coefficient(graph),
        "avg_pairwise_distance": estimate_avg_pairwise_distance(graph),
    }


def analyze_real_dataset(dataset: GraphDataset) -> dict:
    graph = dataset.data["graph"]
    labels = dataset.data["labels"]
    n_nodes = dataset.size()
    n_edges = graph.num_edges()
    n_features = dataset.n_features
    is_regression = dataset.task.is_regression

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges // 2,
        "n_features": n_features,
        "avg_degree": n_edges / n_nodes,
        "target_homophily": compute_homophily(graph, labels, is_regression),
        "target_assortativity": compute_assortativity(graph, labels, not is_regression),
        "degree_powerlaw_r2": compute_degree_powerlaw_r2(graph),
        "clustering_coefficient": compute_clustering_coefficient(graph),
        "avg_pairwise_distance": estimate_avg_pairwise_distance(graph),
    }


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    table = table.apply(pd.to_numeric, errors="coerce")
    summary = {
        "mean": table.mean(),
        "std": table.std(),
        "min": table.min(),
        "Q25": table.quantile(0.25),
        "median": table.median(),
        "Q75": table.quantile(0.75),
        "max": table.max(),
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
    sampler: GraphPriorSampler,
    n_datasets: int,
    path: str | Path | None,
) -> pd.DataFrame:
    analysis_results = []
    for _ in tqdm(range(n_datasets)):
        batch = next(sampler)
        n_nodes = int(batch["n_nodes"][0].item())
        n_features = int(batch["n_features"][0].item())
        n_edges = int(batch["n_edges"][0].item())
        dataset: PriorDataset = {
            "features": batch["features"][0, :n_nodes, :n_features],
            "labels": batch["labels"][0, :n_nodes],
            "edges": batch["edges"][0, :, :n_edges],
            "n_train_nodes": batch["n_train_nodes"],
            "task_type": batch["task_type"],
        }
        analysis_results.append(analyze_prior_dataset(dataset))

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
        "artnet-views",
    ],
) -> None:
    results = dict()
    for dataset_name in dataset_names:
        dataset = GraphDataset.from_dir(
            path=f"data/{dataset_name}",
            setting="transductive",
            internal_split_name="RL",
        )
        results[dataset_name] = analyze_real_dataset(dataset)
    results = pd.DataFrame(results)
    results = results.map(lambda x: f"{x:.4f}" if x is not None else None)
    print(results.to_markdown(disable_numparse=True))


def get_prior_config(pretrain_config_path: str | Path) -> dict:
    with open(pretrain_config_path, "rb") as file:
        pretrain_config = tomllib.load(file)
    config = pretrain_config["base_config"]["prior"]
    pprint(config)
    return config


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrain_config_path", type=str)
    parser.add_argument("--n_datasets", type=int, default=1_000)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = vars(args)

    config["tests"] = False
    config["analyze_real_datasets"] = True
    config["prior"] = get_prior_config(config["pretrain_config_path"])

    return config


if __name__ == "__main__":
    lib.configure_logging()

    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    config = get_config()
    delu.random.seed(config["seed"])

    if config["tests"]:
        run_tests()

    if config["analyze_real_datasets"]:
        analyze_real_datasets()

    seed = config["seed"] if config["n_workers"] > 0 else None
    sampler = GraphPriorSampler(
        config=config["prior"],
        batch_size=1,
        seed=seed,
        n_workers=config["n_workers"],
        verbose=False,
    )

    analysis_results = analyze_prior(
        sampler,
        config["n_datasets"],
        path=(Path("local/analysis") / config["pretrain_config_path"]).parent,
    )
    print(f"{config['n_datasets']=}")
    print(analysis_results.to_markdown(disable_numparse=True))
