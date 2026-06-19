# Experiments

Each experiment tests one hypothesis by varying a single axis. All experiments share the same infrastructure (`experiments.core`).

## Architecture

```
experiments/
├── core/           # Shared infrastructure (bootstrap, runner, model_params, grid)
├── paper/          # Active experiments
│   ├── README.md   # This file
│   ├── p01_arch_model/
│   ├── p02_font_feature/
│   ├── p03_leaning_coef/
│   ├── p04_count_gnn/
│   ├── p05_model_type/
│   ├── p06_lr_batch/
│   ├── p07_diff_graphs/
│   ├── p08_restart/
│   └── p09_portability/
└── archive/        # Retired experiments
```

## Core API

```python
from experiments.core import init, ExperimentRunner, product
from experiments.core.model_params import default_arch

init()
runner = ExperimentRunner("result")
runner.run({name: default_arch(**overrides) for ...})
```

## Experiments

### 01. arch_model — GNN convolution types

**Hypothesis:** TAG convolution outperforms GCN and GAT for row-region graphs.

**Axes:** `type_model` ∈ {base, plus_gnn}, `conv_type` ∈ {tag, conv, gat}, `lp` ∈ {lp, nolp}

**Run:** `python experiments/paper/p01_arch_model/start.py`

### 02. font_feature — Font features

**Hypothesis:** Adding font features (size, family, color) to node vectors improves region classification.

**Axes:** `input_dim` ∈ {15, 527} (with/without font features)

**Run:** `python experiments/paper/p02_font_feature/start.py`

### 03. leaning_coef — Loss coefficients

**Hypothesis:** The edge component of the loss is more important than the node component.

**Axis:** `edge_coef` ∈ {0.0, 0.2, 0.5, 0.8, 1.0}

**Run:** `python experiments/paper/p03_leaning_coef/start.py`

### 04. count_gnn — Number of GNN layers

**Hypothesis:** More layers ≠ better. Deep GNNs on small graphs overfit.

**Axis:** `num_layers` ∈ {1, 2, 3, 4}

**Run:** `python experiments/paper/p04_count_gnn/start.py`

### 05. model_type — Architecture variants

**Hypothesis:** The main block-based architecture outperforms legacy alternatives.

**Axes:** architecture variant ∈ {base, main, custom, loget, loget_no}

**Run:** `python experiments/paper/p05_model_type/start.py`

### 06. lr_batch — Learning rate and batch size

**Hypothesis:** Smaller batch sizes and moderate learning rates work best for graph-level batches.

**Axes:** `lr` ∈ {0.01, 0.005, 0.001, 0.0005}, `batch_size` ∈ {1, 8, 64, 128}

**Run:** `python experiments/paper/p06_lr_batch/start.py`

### 07. diff_graphs — Oracle upper bound

**Hypothesis:** If the oracle on true nodes/edges produces low quality, the problem is in post-processing, not GNN.

**Axes:** tokenizer type ∈ {glam (sparse), all (full graph)}

**Run:** `python experiments/paper/p07_diff_graphs/start.py`

### 08. restart — Stability

**Hypothesis:** Model metrics are stable across 10 random initializations.

**Run:** `python experiments/paper/p08_restart/start.py`

### 09. portability — Cross-dataset

**Hypothesis:** Models trained on PubLayNet transfer to DocLayNet and vice versa.

**Axes:** train ∈ {pub, doc}, test ∈ {pub, doc}

**Run:** `python experiments/paper/p09_portability/start.py`
