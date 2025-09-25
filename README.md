# GraphPFN

This is the official repository of "GraphPFN: A Prior-Data Fitted Graph Foundation Model" paper. In this repository, we provide code for reproducing our experiments with GraphPFN, both pretraining and evaluation.

## Licenses

Please note that we use third-party components (specifically, [TabICL](https://github.com/soda-inria/tabicl) and [LimiX](https://github.com/limix-ldm/LimiX)) in our code with some modifications. Please see `NOTICE` file and `LICENSES/` directory for details. Also, LimiX serves as a backbone for GraphPFN, and LimiX weights have their own license, please check out [LimiX repository](https://github.com/limix-ldm/LimiX) for details.

## Reproducing Experiments

**Prerequisites**

1. [Install uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
2. Install dependencies
```
uv sync
```
3. For experiments on [GraphLand](https://github.com/yandex-research/graphland), download datasets and place them in "data" directory

**Running the evaluation**

You can execute a minimal evaluation run with a following command:

```
uv run bin/go.py exp/graphpfn-eval/finetune/raw/tolokers-2/tuning.toml --force
```

**Running the pretraining**

First, you will need to generate graphs and store them in `data/graphpfn-graphs`, check `bin/prior/README.md` for details.

Then, to run GraphPFN pretraining you can use the following command:

```
DGLBACKEND=pytorch uv run -m torch.distributed.run --nproc-per-node 8 bin/graphpfn_pretrain.py exp/graphpfn-pretrain/pretrain.toml
```

## Project Structure

- `bin/` - Training and evaluation scripts
- `exp/` - Experiment configurations and results
- `data/` - Dataset directory (created after download)
- `lib/` - Common utilities and tools

## Configuration

Experiments are configured using TOML files located in the `exp/` directory. Each configuration specifies:
- Dataset path and preprocessing
- Model hyperparameters
- Training settings
- Evaluation metrics

## Results

Evaluation results are saved in the same directory as the configuration file:
- `report.json` - Evaluation metrics
- Model checkpoints
- Training logs
