# GraphPFN

Here we provide the official implementation for reproducing experiments from the "GraphPFN: A Prior-Data Fitted Graph Foundation Model" ICML 2026 paper ([arXiv](https://arxiv.org/abs/2509.21489)), including pretraining and evaluation. See also the HuggingFace [page](https://huggingface.co/eremeev-d/graphpfn-1.3) for model weights and the `graphpfn` [package](../README.md) for a convenient Python interface.

## Licenses

- This project uses third-party components [LimiX](https://github.com/limix-ldm/LimiX), [TabICL](https://github.com/soda-inria/tabicl) and [TabPFN](https://github.com/PriorLabs/TabPFN). See the `NOTICE` file and `LICENSES/` directory for details.
- GraphPFN prior in `lib/graphpfn/prior` is largely based on the [TabICLv1](https://github.com/soda-inria/tabicl) prior.
- LimiX serves as the backbone for GraphPFN, and its weights have a separate license – see the LimiX [repository](https://github.com/limix-ldm/LimiX).

## Reproducing Experiments

**Prerequisites**

1. [Install uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
2. Install dependencies
```
uv sync
```
3. For experiments on [GraphLand](https://github.com/yandex-research/graphland), download datasets and place them in the `data` directory

**Running the evaluation**

You can execute a minimal evaluation run (GraphPFN finetuning with 10 ensemble members) with the following command:

```
uv run bin/go.py exp/graphpfn/eval/main/finetune/10/tolokers-2/tuning.toml --force
```

**Running the pretraining**

To run GraphPFN pretraining you can use the following command:

```
DGLBACKEND=pytorch uv run -m torch.distributed.run --nproc-per-node 8 bin/go.py exp/graphpfn/pretrain/main/pretrain.toml
```

## Project Structure

- `bin/` - Training and evaluation scripts
- `exp/` - Experiment configurations and results
- `data/` - Dataset directory
- `lib/` - Common utilities and tools
- `vendor/` – Vendored third-party code with minor import modifications for compatibility

## Configuration

Experiments are configured using TOML files located in the `exp/` directory. Each configuration specifies:
- Dataset path and preprocessing
- Model hyperparameters
- Training settings
- Evaluation metrics

## Results

Evaluation results are saved in the same directory as the configuration file:
- `report.json` – Evaluation metrics
- Model checkpoints
- Training logs

