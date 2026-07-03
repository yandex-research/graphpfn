# GraphPFN

**GraphPFN** is a graph foundation model for node-level tasks, built on the [prior-data fitted networks (PFN)](https://arxiv.org/abs/2112.10510) framework and accepted at ICML 2026 ([arXiv](https://arxiv.org/abs/2509.21489), [poster](https://icml.cc/virtual/2026/poster/66511)). It is pretrained on a novel synthetic prior that combines multi-level stochastic block models with graph-aware structured causal models to generate millions of diverse attributed graphs, and extends the PFN-based tabular foundation model [LimiX](https://github.com/limix-ldm/LimiX) with attention-based graph neighborhood aggregation (message passing) modules. This lets GraphPFN generalize across diverse real-world graph domains, supporting both in-context learning and finetuning regimes. Pretrained weights are available on [HuggingFace](https://huggingface.co/eremeev-d/graphpfn-1.3).

> [!NOTE]
> This repository provides both the `graphpfn` **Python package** for using pretrained GraphPFN on your own graphs, and the **paper code** used to reproduce all experiments from the paper, including model pretraining and synthetic dataset generation. See the [Paper Code](#paper-code) section and `paper` directory for details.

## News

- 2026-07-03: `graphpfn` Python package released on [PyPI](https://pypi.org/project/graphpfn/), offering convenient interfaces for using pretrained GraphPFN on your own graphs!
- 2026-06-22: GraphPFN-1.3 released, featuring a refactored graph prior, native ECOC support, extended evaluation, and various other improvements!
- 2026-04-30: GraphPFN [accepted](https://icml.cc/virtual/2026/poster/66511) to ICML 2026!
- 2026-02-13: GraphPFN-1.2 released, featuring improved ICL performance, end-to-end dataset generation and more!
- 2025-09-25: GraphPFN-1.0 released!

## Installation

**Prerequisites:** Python 3.11+ and a CUDA-capable GPU (CUDA 11.8 or newer).

There are two ways to use the `graphpfn` package. You can install it via `pip` into your own environment, handling PyTorch and DGL dependencies yourself. Alternatively, you can clone the repo and write your scripts inside it — [`uv`](https://github.com/astral-sh/uv) will manage the entire environment for you, including PyTorch and DGL, with no manual installation steps.

### Install as a package

**1. Install [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai)**

The commands below use CUDA 11.8 and PyTorch 2.4. For other combinations, check the [DGL getting started page](https://www.dgl.ai/pages/start.html). Please note that DGL does not officially support more modern PyTorch versions, so `graphpfn` pins `torch<2.5`.

```bash
pip install "torch==2.4" --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
```

**2. Install `graphpfn`**

From PyPI:
```bash
pip install graphpfn
```

Or from a local clone:
```bash
git clone https://github.com/yandex-research/graphpfn
cd graphpfn
pip install -e .
```

### Work within the repo

Clone the repo, place your script anywhere inside it (e.g. `examples/`), and run it with `uv run`. The full environment — including PyTorch and DGL — is defined in `uv.lock` and set up automatically on first run, with no manual `pip install` needed.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/yandex-research/graphpfn
cd graphpfn

# Run a script
uv run examples/basic.py
```

## Usage

### Quick Start

The snippet below runs ICL inference on an [amazon-ratings](https://arxiv.org/abs/2302.11640) dataset using pretrained GraphPFN weights downloaded automatically from HuggingFace:

```python
from torch_geometric.datasets import HeterophilousGraphDataset
from graphpfn import GraphDataset, predict_icl
from graphpfn.inference.util import compute_metrics

pyg_data = HeterophilousGraphDataset("data", "amazon-ratings")[0]
dataset = GraphDataset.from_pyg(pyg_data, task_type="multiclass", name="amazon-ratings")

preds = predict_icl(dataset=dataset, device="cuda:0")

test_mask = dataset.masks["test"]
metrics = compute_metrics(
    y_true=dataset.targets[test_mask],
    y_pred=preds[test_mask],
    task_type=dataset.task_type,
)
print(metrics)  # {"accuracy": ...}
```

See [`examples/basic.py`](examples/basic.py) and [`examples/finetune.py`](examples/finetune.py) for full runnable examples.

---

### GraphDataset

`GraphDataset` is a simple dataclass that holds everything needed for inference: the graph, node features, targets, train/val/test masks, and the task type.

```python
@dataclass
class GraphDataset:
    name: str
    graph: dgl.DGLGraph
    features: FeatureArrays   # TypedDict with keys: "num", "frac", "cat", "other"
    targets: np.ndarray       # shape [n_nodes]; NaN for unlabeled nodes
    masks: MaskArrays         # TypedDict with keys: "train", "val", "test"
    task_type: TaskType       # "binclass" | "multiclass" | "regression"
```

Features are split by type so the preprocessing pipeline can apply the right transform to each:
- **`"num"`** — general numerical features
- **`"frac"`** — fractional features in [0, 1]
- **`"cat"`** — categorical features as integer codes
- **`"other"`** — anything else, e.g. neural or sparse embeddings

Only the keys present in the dict are processed, you don't need to provide all four types.

**From PyTorch Geometric:**

```python
from graphpfn import GraphDataset

dataset = GraphDataset.from_pyg(
    pyg_data,               # torch_geometric.data.Data
    task_type="multiclass",
    name="my-dataset",
    split_index=0,          # which column of *_mask to use (default: 0)
)
```

PyG node features (`pyg_data.x`) are loaded as `"other"` features. If they are high-dimensional neural embeddings, consider setting `d_pca=64` in `preprocessing_kwargs` (see [ICL Inference](#icl-inference)).

You can also replace masks after construction (e.g. to use custom splits):

```python
custom_splits = dict(np.load("split.npz"))
dataset.masks = custom_splits
```

**From GraphLand:**

[GraphLand](https://arxiv.org/abs/2409.14500) is a benchmark collection of real-world industrial graph datasets with pre-defined splits.

```python
dataset = GraphDataset.from_graphland(
    "tolokers-2",
    split="RL",          # "RL" (default) | "RH" | "TH"
    root_dir="data",     # path where datasets are stored
)
```

This loader automatically separates numerical, fractional, and categorical features based on dataset metadata.

**From custom data:**

You can build a `GraphDataset` directly from any DGL graph and NumPy arrays:

```python
import dgl, numpy as np, torch
from graphpfn import GraphDataset

graph = dgl.graph((src_nodes, dst_nodes), num_nodes=N)
features = {"num": numerical_array, "cat": categorical_array}
targets = np.array(...)   # shape [N], float32; NaN for unlabeled nodes
masks = {"train": train_bool_array, "val": val_bool_array, "test": test_bool_array}

dataset = GraphDataset(
    name="my-dataset",
    graph=graph,
    features=features,
    targets=targets,
    masks=masks,
    task_type="binclass",
)
```

---

### ICL Inference

In-context learning (ICL) runs the pretrained model **without any weight updates**. Training nodes are passed as context and the model predicts all other nodes in a single forward pass. This is the fastest mode and works well out of the box.

```python
from graphpfn import predict_icl

preds = predict_icl(
    dataset=dataset,
    preprocessing_kwargs={"d_pca": 64},  # recommended for neural-embedding features
    amp=True,       # bfloat16 autocast (default: True)
    device="cuda:0",
)
```

`predict_icl` returns a NumPy array of predictions for **all nodes**: shape `[n_nodes]` for binary classification and regression, or `[n_nodes, n_classes]` for multiclass. Index with a mask to get per-split results.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_kwargs` | `{}` | Passed to `GraphPFN.from_pretrained`. Use `"checkpoint"` key to specify a local path or HuggingFace URI. |
| `preprocessing_kwargs` | `{}` | Feature preprocessing options; see the table below. |
| `amp` | `True` | Enables bfloat16 autocast for faster inference on supported GPUs. |
| `device` | — | Required. Target CUDA device, e.g. `"cuda:0"`. |

**Preprocessing options** (via `preprocessing_kwargs`):

| Parameter | Default | Description |
|---|---|---|
| `num_transform` | `"quantile-normal"` | Transform for `"num"` features. One of `"standard"`, `"quantile-normal"`, `"quantile-uniform"`, or `None`. |
| `frac_transform` | `"quantile-uniform"` | Transform for `"frac"` features. Same options as `num_transform`. |
| `cat_transform` | `"ordinal"` | Transform for `"cat"` features. One of `"ordinal"`, `"one-hot"`, or `None`. |
| `other_transform` | `"standard"` | Transform for `"other"` features. Same options as `num_transform`. |
| `d_pca` | `None` | If set, all features are concatenated and reduced to `d_pca` dimensions with PCA. Recommended (e.g. `64`) when features are high-dimensional neural embeddings. |

See [`examples/basic.py`](examples/basic.py) for a full runnable example.

---

### Finetuning

Finetuning updates the model weights on training nodes of the target graph. This typically improves performance over ICL at the cost of more compute. **We recommend** `tune_and_predict_ft`, which performs a grid search over learning rates and returns predictions from the best trial.

```python
from functools import partial
from graphpfn import tune_and_predict_ft
from graphpfn.inference.util import compute_score, compute_loss

preds, best_hparams = tune_and_predict_ft(
    dataset=dataset,
    score_fn=partial(compute_score, task_type=dataset.task_type),
    loss_fn=partial(compute_loss, task_type=dataset.task_type),
    device="cuda:0",
)
print(best_hparams)  # {"lr": 0.0005}
```

`tune_and_predict_ft` returns `(predictions, best_hyperparams)`. Predictions have the same shape as in `predict_icl`.

`score_fn` **and** `loss_fn`**:**

The library provides ready-to-use implementations in `graphpfn.inference.util`:

```python
from graphpfn.inference.util import compute_score, compute_loss, compute_metrics

# compute_score: single scalar, higher is always better (used for early stopping)
# Defaults: average_precision (binclass), accuracy (multiclass), R^2 (regression)
score = compute_score(y_true=..., y_pred=..., task_type="multiclass")

# compute_loss: differentiable Tensor loss (cross-entropy or MSE)
loss = compute_loss(y_true=..., y_pred=..., task_type="multiclass")

# compute_metrics: dict of all relevant metrics
metrics = compute_metrics(y_true=..., y_pred=..., task_type="binclass")
# {"average_precision": ..., "roc_auc": ...}
```

You can also pass any custom callables matching the `ScoreFn` / `LossFn` protocols from `graphpfn.inference`.

**Key parameters** (shared between `tune_and_predict_ft` and `predict_ft`):

| Parameter | Default | Description |
|---|---|---|
| `score_fn` | — | Required. Scalar metric used for validation and early stopping (higher = better). |
| `loss_fn` | — | Required. Differentiable loss used for gradient updates. |
| `n_steps` | `-1` | Total training steps. `-1` means run until early stopping triggers. |
| `steps_per_epoch` | `10` | Steps between validation evaluations. |
| `patience` | `4` | Early stopping patience in epochs. |
| `train_batch_size` | `1024` | Number of training nodes predicted per gradient step. |
| `model_kwargs` | `{}` | Same as in `predict_icl`. |
| `preprocessing_kwargs` | `{}` | Same as in `predict_icl`. |
| `amp` | `True` | Same as in `predict_icl`. |
| `device` | — | Required. |
| `verbose` | `True` | Print validation scores during training. |

**Learning rate:** `predict_ft` takes a single `lr: float` (required), while `tune_and_predict_ft` takes `lr_grid: list[float]` — it runs one `predict_ft` trial per value and returns predictions from the best-scoring trial.

| Parameter | Function | Default | Description |
|---|---|---|---|
| `lr` | `predict_ft` | — | Learning rate for the Adam optimizer. |
| `lr_grid` | `tune_and_predict_ft` | 10 log-spaced values 5e-6 to 5e-4 | List of learning rates to try; the trial with the best validation score wins. |

See [`examples/finetune.py`](examples/finetune.py) for a full runnable example.

---

### Model API

`GraphPFN` and `GraphPFNLight` are standard `torch.nn.Module` subclasses, useful when you need more control than the high-level inference functions provide.

`GraphPFN` wraps `GraphPFNLight` with an ensemble of feature/target permutations. It is used internally by all inference functions and is the recommended way to interact with the model directly:

```python
from graphpfn import GraphPFN

model = GraphPFN.from_pretrained(
    n_features=64,
    n_classes=5,           # None for regression
    device="cuda:0",
    checkpoint="hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt",  # default
    ensemble_kwargs={"n_members": 10},  # default
)

preds = model(
    graph=graph,                  # DGL Graph
    features=features,            # Tensor [n_nodes, n_features]
    y_train=y_train,              # Tensor [n_train]
    train_mask=train_mask,        # BoolTensor [n_nodes]
    task_type="multiclass",
)  # → Tensor [n_nodes, n_classes]
```

Each of the `n_members` ensemble runs uses a different random permutation of features and (for multiclass) class labels, and the outputs are aggregated before returning.

For **custom finetuning loops**, use `GraphPFN.train_step_forward` instead of `forward` — it samples a single random ensemble member and returns `(query_preds, query_targets)` in the transformed target space, ready for a loss and `.backward()`. Running one member per step keeps gradient steps as cheap as a single ICL forward pass while still covering the ensemble's permutation space over many steps. See [`src/graphpfn/model/graphpfn.py`](src/graphpfn/model/graphpfn.py) for the method and [`src/graphpfn/inference/finetune.py`](src/graphpfn/inference/finetune.py) for a reference implementation.

`GraphPFNLight` is the bare backbone without ensembling — a single forward pass with no permutations. Its `forward` signature is identical to `GraphPFN`. It is intended for advanced use cases where you manage ensembling yourself. For standard inference, prefer `GraphPFN`.

## Paper Code

The [`paper/`](paper/) directory contains the code used to reproduce all paper experiments, including model pretraining, synthetic dataset generation, and evaluation. See [`paper/README.md`](paper/README.md) for setup instructions and full details.

## Licenses

GraphPFN code and weights are released under the [Apache License 2.0](LICENSE). The vendored LimiX component is also distributed under Apache 2.0 — see [`LICENSES/LimiX.txt`](LICENSES/LimiX.txt). Note that **LimiX pretrained weights** are distributed under a separate license, users are responsible for complying with it — see the [LimiX repository](https://github.com/limix-ldm/LimiX) for details. All vendored third-party components are listed in the [`NOTICE`](NOTICE) file and [`LICENSES/`](LICENSES/) directory.

## Citation

If you use GraphPFN in your research, please cite our paper:

```bibtex
@inproceedings{eremeev2026graphpfn,
  title={{GraphPFN}: A Prior-Data Fitted Graph Foundation Model},
  author={Eremeev, Dmitry and Platonov, Oleg and Bazhenov, Gleb and Babenko, Artem and Prokhorenkova, Liudmila},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
