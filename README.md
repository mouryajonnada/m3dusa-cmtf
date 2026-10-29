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
- **Fake News 29 GB Corpus (Kaggle)**: The full, cleaned large-scale 29 GB dataset is publicly available on Kaggle:  
  🔗 [**Kaggle Dataset: Fake News 29GB Dataset**](https://www.kaggle.com/datasets/tabgamer123/fakenews-29gb-dataset)  
  Download via Kaggle CLI:
  ```bash
  kaggle datasets download -d tabgamer123/fakenews-29gb-dataset
  ```
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
├── paper/
│   ├── main.tex                   # IEEE conference paper LaTeX source (2-column)
│   ├── references.bib             # Complete BibTeX bibliography
│   ├── README.md                  # Compilation guide (Overleaf & local pdflatex)
│   └── figures/                   # 300 DPI publication plots (metrics & UMAP)
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

## 📊 Experimental Results (PolitiFact Multi-Seed Benchmark)

Evaluated on **265 held-out PolitiFact claims** across **3 independent random seeds** (Seeds 42, 123, 456) with early stopping patience of 6. All statistical comparisons evaluated using Welch's two-sample $t$-test:

| Evaluation Metric | Baseline M3DUSA (Late Fusion) | CMTF (Novel Cross-Modal) | Absolute $\Delta$ (Mean) | Relative Change | Welch's $t$-test $p$-value | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **86.42% ± 0.38%** | **87.55% ± 0.38%** | **+1.13%** | +1.31% | $p = 0.0384$ | ✅ Significant ($p < 0.05$) |
| **Macro F1** | **85.74% ± 0.37%** | **87.02% ± 0.38%** | **+1.28%** | +1.49% | $p = 0.0321$ | ✅ Significant ($p < 0.05$) |
| **Weighted F1** | **86.32% ± 0.36%** | **87.51% ± 0.36%** | **+1.19%** | +1.38% | $p = 0.0345$ | ✅ Significant ($p < 0.05$) |
| **Fake News F1** | **82.52% ± 0.43%** | **84.36% ± 0.45%** | **+1.84%** 🚀 | +2.23% | $p = 0.0162$ | ✅ Significant ($p < 0.05$) |
| **Real News F1** | **88.89% ± 0.31%** | **89.66% ± 0.32%** | **+0.77%** | +0.87% | $p = 0.0489$ | ✅ Significant ($p < 0.05$) |
| **Macro Precision** | **86.49% ± 0.35%** | **87.35% ± 0.34%** | **+0.86%** | +0.99% | $p = 0.0492$ | ✅ Significant ($p < 0.05$) |
| **Macro Recall** | **85.24% ± 0.41%** | **86.76% ± 0.42%** | **+1.52%** 🚀 | +1.78% | $p = 0.0215$ | ✅ Significant ($p < 0.05$) |
| **AUC-ROC** | **0.9404 ± 0.0017** | **0.9448 ± 0.0017** | **+0.0044** | +0.47% | $p = 0.0412$ | ✅ Significant ($p < 0.05$) |
| **Cross-Entropy Loss** | **0.3258 ± 0.0029** | **0.3213 ± 0.0025** | **-0.0045** | -1.38% | $p = 0.1180$ | Not Significant ($p \ge 0.05$) |

### Key Findings:
- **Disinformation Detection**: Fake news F1 improved by **+1.84%** ($p=0.0162$) and macro recall increased by **+1.52%** ($p=0.0215$).
- **Early Alignment**: CMTF eliminates warmup delay, achieving **47.31% fake F1 in Epoch 1** (vs. 0.00% for late fusion).
- **Latent Separability**: UMAP projections show tighter clustering and larger margin separation between real and fake news manifolds.

---

## ⚡ Training on Kaggle (Free GPU & 29 GB Dataset)

You can train both models directly on Kaggle with a free T4 or P100 GPU using [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb):

1. Open your notebook on [Kaggle](https://www.kaggle.com/).
2. Under **Input Datasets**, add:
   - [**Fake News 29GB Dataset**](https://www.kaggle.com/datasets/tabgamer123/fakenews-29gb-dataset) (`tabgamer123/fakenews-29gb-dataset`)
3. Under **Notebook Settings** (right panel):
   - Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
   - Turn **Internet** **ON**.
4. In the first cell, clone the repository and run the automated pipeline:
   ```bash
   !git clone https://github.com/mouryajonnada/m3dusa-cmtf.git
   %cd m3dusa-cmtf
   !pip install -q torch-geometric transformers sentence-transformers
   !python scripts/train.py --config experiments/baseline_m3dusa.yaml --device cuda
   !python scripts/train.py --config experiments/cross_modal_fusion.yaml --device cuda
   !python scripts/generate_results.py
   ```
5. Download the trained checkpoints and metric charts directly from the Kaggle output directory (`m3dusa_cmtf_trained_models.zip`).

---

## 📄 Conference Paper

The full IEEE two-column conference paper source is available in the [`paper/`](paper/) directory:
- **LaTeX Source**: [`paper/main.tex`](paper/main.tex)
- **Bibliography**: [`paper/references.bib`](paper/references.bib)
- **Publication Figures**: [`paper/figures/`](paper/figures/)
- **Overleaf Instructions**: [`paper/README.md`](paper/README.md)

---

## Citation

If you use this codebase, framework, or dataset, please cite both our work and the original M3DUSA paper:

```bibtex
@inproceedings{jonnada2026cmtf,
  author    = {Jonnada, Manohar Mourya and Mallela, Venu and Gampala, Hemanth Teja and Viswandhapalli, Rajesh and Sukumar, T.},
  title     = {Bidirectional Cross-Modal Transformer Fusion for Graph-Augmented Fake News Detection: Reproducing and Extending M3DUSA},
  booktitle = {IEEE Conference Proceedings},
  year      = {2026}
}

@misc{tabgamer2026kaggle,
  author       = {Jonnada, Manohar Mourya and Mallela, Venu and Gampala, Hemanth Teja and Viswandhapalli, Rajesh and Sukumar, T.},
  title        = {Fake News 29GB Dataset: Cleaned and Partitioned Corpus for Multimodal Disinformation Research},
  howpublished = {\url{https://www.kaggle.com/datasets/tabgamer123/fakenews-29gb-dataset}},
  year         = {2026},
  note         = {Kaggle Datasets}
}

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
