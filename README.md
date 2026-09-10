# Fake News Detection via Cross-Modal Transformer Fusion

Reproducing and extending the **M3DUSA** framework (Martirano et al., 2025) on the
**PolitiFact** dataset, with a novel **Cross-Modal Transformer Fusion** module replacing
late-fusion baselines.

---

## Background

[M3DUSA](https://github.com/lilymart/M3DUSA) *(Multimodal Multi-Domain Unified Stance
Analysis)* is a graph-augmented multi-modal fake news detection framework that encodes
claim text via a pre-trained language model and evidence structure via a Graph Attention
Network (GAT), fusing the two representations through a late-fusion strategy before
classification.

This project:
1. **Reproduces** the M3DUSA baseline on PolitiFact using the same architecture.
2. **Extends** it by replacing the late-fusion step with a novel
   **Cross-Modal Transformer Fusion** module — a stack of bidirectional cross-attention
   blocks that let the text and graph representations attend to each other before
   being pooled and classified.

---

## Dataset

**PolitiFact** — fact-checked political claims with six veracity labels (true, mostly-true,
half-true, barely-true, false, pants-on-fire), binarised to *real* / *fake* for
binary classification experiments.

Raw data should be placed in `data/raw/`. Processing is handled by `src/data/`.

---

## Project Structure

```
.
├── data/
│   ├── raw/              # Original PolitiFact dumps (not tracked in git)
│   └── processed/        # Graphs, tokenised tensors, splits (not tracked)
├── experiments/
│   ├── baseline_m3dusa.yaml       # M3DUSA reproduction config
│   └── cross_modal_fusion.yaml    # Novel Cross-Modal Fusion config
├── results/
│   ├── metrics/          # JSON metric files per experiment run
│   ├── plots/            # Confusion matrices, ROC curves, etc.
│   └── logs/             # Training logs
├── notebooks/            # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── dataset.py         # PolitiFact dataset & DataLoader
│   │   ├── preprocess.py      # Text cleaning & tokenisation
│   │   └── graph_builder.py   # k-NN evidence graph construction
│   ├── models/
│   │   ├── text_encoder.py    # HuggingFace transformer wrapper
│   │   ├── gat_encoder.py     # GATv2Conv graph encoder
│   │   ├── fusion.py          # Late-fusion & Cross-Modal Transformer Fusion
│   │   └── classifier.py      # MLP classifier head
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── evaluator.py       # Evaluation loop
│   └── utils/
│       ├── config.py          # YAML config loader (DotDict)
│       ├── metrics.py         # Accuracy, F1, AUC
│       ├── seed.py            # RNG seeding for reproducibility
│       └── logging.py         # Logger factory
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch (match your CUDA version)

```bash
# Example: CUDA 12.1
pip install torch==2.3.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for other builds.

### 3. Install PyG CUDA extra dependencies

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

Adjust the `--find-links` URL to match your exact `torch+cuda` combo.
Full list: <https://data.pyg.org/whl/>

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Experiments are fully config-driven. The entry point is `scripts/train.py`:

### 1. Data Pipeline Diagnostics
Sanity check raw data loading, graph extraction, and stratified 60/15/25 split:
```bash
python scripts/run_diagnostics.py
```

### 2. Train Baseline M3DUSA (Late Fusion)
```bash
python scripts/train.py --config experiments/baseline_m3dusa.yaml \
    --override model.freeze_text_encoder=True training.epochs=5 training.batch_size=16
```

### 3. Train Novel Cross-Modal Transformer Fusion (CMTF)
```bash
python scripts/train.py --config experiments/cross_modal_fusion.yaml \
    --override model.freeze_text_encoder=True training.epochs=5 training.batch_size=16
```

### 4. Generate Comparisons, Bar Chart, & UMAP Visualizations
Extract test set embeddings, compute UMAP 2D projections, generate bar charts, and export CSV/Markdown reports:
```bash
python scripts/generate_results.py
```

Outputs are automatically saved to:
- Checkpoints: `results/checkpoints/{experiment_name}/best_model.pt`
- Metrics: `results/metrics/{experiment_name}/test_metrics.json` & `history.csv`
- Full Comparison Table: `results/metrics/comparison.csv`
- Plots: `results/plots/baseline_vs_cmtf_metrics.png` & `results/plots/umap_embeddings.png`
- Summary Report: `results_summary.md`

---

## Config System

Configs are YAML files loaded into dot-accessible `DotDict` objects:

```python
from src.utils.config import load_config, merge_configs, save_config

cfg = load_config("experiments/baseline_m3dusa.yaml")
print(cfg.model.gat.num_layers)   # 2
print(cfg.experiment.fusion_method)  # "late_fusion"

# Deep merge CLI overrides:
cfg = merge_configs(cfg, {"training": {"lr": 1e-4}})

# Save active config alongside results for reproducibility:
save_config(cfg, "results/run_001/config.yaml")
```

---

## Citation

If you build on M3DUSA, please cite:

```bibtex
@article{martirano2025m3dusa,
  title   = {M3DUSA: Multimodal Multi-Domain Unified Stance Analysis for Fake News Detection},
  author  = {Martirano, Lily and others},
  year    = {2025},
  url     = {https://github.com/lilymart/M3DUSA}
}
```

---

## License

MIT
