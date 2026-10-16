# Fake News Detection via Cross-Modal Transformer Fusion

Reproducing and extending the **M3DUSA** framework (Martirano et al., 2025) across **PolitiFact** and **FakeNewsCorpus** datasets, featuring a novel **Cross-Modal Transformer Fusion (CMTF)** architecture replacing traditional late-fusion baselines.

---

## Background

[M3DUSA](https://github.com/lilymart/M3DUSA) *(Multimodal Multi-Domain Unified Stance Analysis)* is a graph-augmented multi-modal fake news detection framework that encodes claim text via a pre-trained language model and evidence structure via a Graph Attention Network (GAT), fusing the two representations through a late-fusion strategy before classification.

This project:
1. **Reproduces** the M3DUSA baseline using text + GAT graph representations.
2. **Extends** it with **Cross-Modal Transformer Fusion (CMTF)** — a bidirectional cross-attention module allowing text tokens and graph node representations to attend to each other before pooling and classification.
3. **Scales** training to large-scale corpora (**FakeNewsCorpus**) with multi-part streaming, balanced class sampling, and continual learning pipelines.
4. **Provides** an interactive **Neural Observatory** web application (FastAPI + Modern Web UI) for real-time model inference and cross-attention comparison.

---

## Datasets

The repository includes curated raw benchmark datasets in `data/raw/`:

- **PolitiFact**: Fact-checked political statements with six veracity labels binarised to *real* (`politifact_real.csv`) and *fake* (`politifact_fake.csv`).
- **FakeNewsCorpus**: A balanced benchmark subset of 2,000 articles (1,000 fake, 1,000 reliable) extracted from the FakeNewsCorpus dump (`fakenewscorpus_fake.csv` and `fakenewscorpus_real.csv`).
- **Multi-Part Large Scale Corpora**: Support for streaming and training across arbitrary multi-gigabyte CSVs (e.g. `NEW_DATA/` with 5-part splits) via `scripts/split_csv.py` and `scripts/train_parts.py`.

---

## Project Structure

```
.
├── data/
│   ├── raw/                       # Raw benchmark datasets (PolitiFact & FakeNewsCorpus)
│   └── processed/                 # Cached graph structures and tokenized tensors (.pt)
├── experiments/
│   ├── baseline_m3dusa.yaml       # M3DUSA late-fusion reproduction on PolitiFact
│   ├── cross_modal_fusion.yaml    # CMTF configuration on PolitiFact
│   ├── fakenewscorpus_baseline.yaml # M3DUSA baseline on FakeNewsCorpus
│   └── fakenewscorpus_cmtf.yaml   # CMTF configuration on FakeNewsCorpus
├── notebooks/
│   └── kaggle_train.ipynb         # GPU training pipeline for Kaggle / Colab
├── results/
│   ├── metrics/                   # JSON test metrics, histories & comparison tables
│   ├── plots/                     # Metric comparison charts & UMAP embeddings
│   └── logs/                      # Training logs
├── scripts/
│   ├── run_diagnostics.py         # Data loading, graph building & split validation
│   ├── train.py                   # Main config-driven training entry point
│   ├── train_multiseed.py         # Multi-seed statistical significance runner
│   ├── train_ablation.py          # Ablation study runner (fusion depth, heads, etc.)
│   ├── plot_ablation.py           # Ablation visualization generator
│   ├── prepare_new_data.py        # Balanced sample extractor from large CSV dumps
│   ├── split_csv.py               # RFC 4180 quotation-safe multi-part CSV splitter
│   ├── train_parts.py             # Multi-part continual / pooled training runner
│   └── generate_results.py        # Generates plots, UMAPs, and benchmark reports
├── src/
│   ├── data/
│   │   ├── dataset.py             # Multi-dataset loader & PyG DataLoader
│   │   ├── preprocess.py          # Text normalization, tokenisation & sentence embedding
│   │   └── graph_builder.py       # Evidence graph & hashtag-augmented graph construction
│   ├── models/
│   │   ├── text_encoder.py        # HuggingFace Transformer wrapper (RoBERTa / DeBERTa)
│   │   ├── gat_encoder.py         # Heterogeneous GATv2Conv graph encoder
│   │   ├── fusion.py              # Late Fusion & Cross-Modal Transformer Fusion
│   │   ├── classifier.py          # Multilayer perceptron classification head
│   │   └── model.py               # Unified FakeNewsDetector model module
│   ├── training/
│   │   ├── trainer.py             # Training loop with warmup, cosine decay & early stopping
│   │   └── evaluator.py           # Evaluation loop computing Acc, F1, AUC, Loss
│   └── utils/
│       ├── config.py              # YAML config loader with DotDict access
│       ├── metrics.py             # Classification metric computations
│       ├── seed.py                # Deterministic random seed management
│       └── logging.py             # Structured logger factory
├── web/
│   ├── app.py                     # FastAPI backend for real-time model inference
│   └── static/                    # Neural Observatory web UI (HTML, CSS, JS)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites
- **Python**: Version 3.9, 3.10, or 3.11 recommended.
- **Git**

### 2. Clone the Repository & Create a Virtual Environment

```bash
git clone https://github.com/mouryajonnada/m3dusa-cmtf.git
cd m3dusa-cmtf

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install PyTorch

Choose either GPU (CUDA) or CPU build depending on your hardware:

#### Option A: GPU Build (CUDA 12.1 / 11.8)
```bash
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Option B: CPU-Only Build
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> [!TIP]
> Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to select other OS and CUDA combinations.

### 4. Install PyTorch Geometric (PyG) Dependencies

Install the PyG sparse acceleration libraries matching your PyTorch and CUDA versions:

```bash
# If using CUDA 12.1 with PyTorch 2.3.0:
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# If using CPU-only build (torch 2.x):
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.14.0+cpu.html
```

*(Note: If using newer PyTorch versions where prebuilt wheels for torch-scatter are not required, standard `pip install torch-geometric` in the next step will automatically handle dependencies).*

### 5. Install Project Dependencies

Install all remaining packages (NLP transformers, scikit-learn, web interface, and utilities):

```bash
pip install -r requirements.txt
```

### 6. Verify Installation

Run the environment diagnostics script to verify that datasets, HuggingFace models, graph builder, and PyTorch devices are ready:

```bash
python scripts/run_diagnostics.py
```

---

## Running Experiments

### 1. PolitiFact Benchmark

#### Train Baseline M3DUSA (Late Fusion)
```bash
python scripts/train.py --config experiments/baseline_m3dusa.yaml
```

#### Train Novel Cross-Modal Transformer Fusion (CMTF)
```bash
python scripts/train.py --config experiments/cross_modal_fusion.yaml
```

#### Run Multi-Seed Validation
Run experiments across multiple seeds (e.g. 42, 123, 456, 789, 999) to compute mean, std, and statistical significance:
```bash
python scripts/train_multiseed.py --seeds 42 123 456 789 999
```

#### Run Ablation Studies
Explore the effect of cross-modal transformer depth, number of attention heads, and residual connections:
```bash
python scripts/train_ablation.py
python scripts/plot_ablation.py
```

---

### 2. FakeNewsCorpus Benchmark

Train directly on the balanced FakeNewsCorpus sample dataset:
```bash
# Late Fusion baseline
python scripts/train.py --config experiments/fakenewscorpus_baseline.yaml

# Cross-Modal Transformer Fusion
python scripts/train.py --config experiments/fakenewscorpus_cmtf.yaml
```

---

### 3. Multi-Part Large CSV Training (FakeNewsCorpus 60GB)

For handling massive news archives (such as the 29 GB `news_cleaned_2018_02_13.csv`):

1. **Split the large CSV** into N balanced chunks with RFC 4180 quotation integrity:
   ```bash
   python scripts/split_csv.py --input NEW_DATA/news_cleaned_2018_02_13.csv --parts 5
   ```

2. **Train across partitions** using `scripts/train_parts.py`:
   ```bash
   # Mode 1: Sequential Continual Learning across Part 1 -> Part 2 -> ... -> Part 5
   python scripts/train_parts.py --mode sequential --epochs 3 --batch-size 16

   # Mode 2: Combined Pooling across all 5 parts
   python scripts/train_parts.py --mode combined --epochs 5 --samples-per-class 1000

   # Mode 3: Single Part Training
   python scripts/train_parts.py --mode single --single-part 1 --epochs 3
   ```

Outputs, per-part checkpoints, and metrics are saved to `results/parts_training/`.

---

### 4. Interactive Web Interface (Neural Observatory)

Launch the interactive real-time fake news detector:

```bash
python web/app.py --port 8000
```

Open **`http://127.0.0.1:8000`** in your browser to:
- Enter any custom news headline or claim.
- Compare predictions, confidence scores, and latency between M3DUSA Baseline and CMTF.
- Inspect the extracted evidence graph and hashtag connections.

---

### 5. Generate Visualizations & Benchmark Reports

Generate test set UMAP 2D projections, metric comparison charts, and updated Markdown tables:

```bash
python scripts/generate_results.py
```

Generated outputs:
- **Metrics Table**: `results/metrics/comparison.csv`
- **Bar Chart**: `results/plots/baseline_vs_cmtf_metrics.png`
- **UMAP Projections**: `results/plots/umap_embeddings.png`
- **Summary Document**: `results_summary.md`

---

## 📊 Experimental Results (PolitiFact Benchmark)

Trained on NVIDIA Tesla T4 GPU with 5-epoch early stopping:

| Metric | Baseline M3DUSA (Late Fusion) | CMTF (Cross-Modal Transformer) | Notes |
|:---|:---:|:---:|:---|
| **Accuracy** | **90.57%** | **88.30%** | Both architectures achieve ~90% accuracy |
| **Macro F1** | **90.16%** | **88.13%** | Balanced classification across classes |
| **Fake News F1** | **88.15%** | **86.70%** | Robust detection of disinformation |
| **Real News F1** | **92.16%** | **89.56%** | High precision on factual claims |
| **Macro Precision** | **90.52%** | **87.90%** | Minimal false alarm rate |
| **Macro Recall** | **89.87%** | **89.12%** | Captures >89% of all fake and real news |
| **AUC-ROC** | **0.9631** | **0.9596** | **State-of-the-art class separability** |
| **Cross-Entropy Loss** | **0.5833** | **0.7650** | Smooth convergence |

---

## ⚡ Training on Kaggle (Free GPU)

You can train both models directly on Kaggle with a free T4 or P100 GPU using [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb):

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. Under **Notebook Settings** (right panel):
   - Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
   - Turn **Internet** **ON**.
3. In the first cell, clone the repository and run the automated pipeline:
   ```bash
   !git clone https://github.com/mouryajonnada/m3dusa-cmtf.git
   %cd m3dusa-cmtf
   !pip install -q torch-geometric transformers sentence-transformers
   !python scripts/train.py --config experiments/baseline_m3dusa.yaml --device cuda
   !python scripts/train.py --config experiments/cross_modal_fusion.yaml --device cuda
   !python scripts/generate_results.py
   ```
4. Download the trained checkpoints and metric charts directly from the Kaggle output directory (`m3dusa_cmtf_trained_models.zip`).

---

## Config System

Experiment configurations are declarative YAML files loaded into dot-accessible `DotDict` instances:

```python
from src.utils.config import load_config, merge_configs, save_config

cfg = load_config("experiments/cross_modal_fusion.yaml")
print(cfg.model.gat.num_layers)        # 2
print(cfg.experiment.fusion_method)   # "cross_modal"

# Deep merge CLI overrides:
cfg = merge_configs(cfg, {"training": {"lr": 1e-4}})

# Save active config alongside results for full reproducibility:
save_config(cfg, "results/run_001/config.yaml")
```

---

## Citation

If you build on M3DUSA, please cite the original work:

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

MIT License. See [LICENSE](LICENSE) for details.
