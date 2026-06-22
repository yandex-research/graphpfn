# GraphPFN

This is the official repository of the "GraphPFN: A Prior-Data Fitted Graph Foundation Model" ICML 2026 paper ([arXiv](https://arxiv.org/abs/2509.21489)). In this repository, we provide code for reproducing our experiments with GraphPFN, both pretraining and evaluation. See also the HuggingFace [page](https://huggingface.co/eremeev-d/graphpfn-1.3) for model weights.

## News

- 2026-06-22: GraphPFN-1.3 released, featuring a refactored graph prior, native ECOC support, extended evaluation, and various other improvements!
- 2026-04-30: GraphPFN [accepted](https://icml.cc/virtual/2026/poster/66511) to ICML 2026!
- 2026-02-13: GraphPFN-1.2 released, featuring improved ICL performance, end-to-end dataset generation and more!
- 2025-09-25: GraphPFN-1.0 released!

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

