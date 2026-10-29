# IEEE Conference Paper: CMTF for Fake News Detection

This folder contains the complete, publication-ready IEEE conference paper written in LaTeX, matching the formatting and structure of the reference conference paper (*An Explainable Multi-Layer Framework for Real-Time Quishing Detection and Risk Verification*).

## Contents

- **`main.tex`**: The primary LaTeX document configured with `\documentclass[conference]{IEEEtran}`.
- **`references.bib`**: BibTeX bibliography containing complete citations for all referenced works (M3DUSA, RoBERTa, HGT, GAT, Transformers, FakeNewsNet, etc.).
- **`figures/`**: Publication-ready 300 DPI figures:
  - `baseline_vs_cmtf_metrics.png`: Test set performance metrics bar chart.
  - `umap_embeddings.png`: 2D UMAP projection comparing late-fusion vs. CMTF latent space.

---

## How to Compile to PDF

### Method 1: Using Overleaf (Recommended & Fastest)
1. Select the `paper` folder on your computer.
2. Compress `paper` into a `.zip` file:
   - Include `main.tex`, `references.bib`, and the `figures/` subfolder.
3. Go to [Overleaf](https://www.overleaf.com/) and click **New Project** -> **Upload Project**.
4. Upload the zip file and click **Recompile**. Overleaf includes `IEEEtran.cls` by default!

### Method 2: Local Compilation (Command Line)
If you have TeX Live, MacTeX, or MiKTeX installed:
```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
This generates `main.pdf`.

---

## Paper Specifications

- **Format**: IEEE Conference (Two-column, standard IEEE margins and font sizing).
- **Core Experimental Claims**:
  - Evaluation on 265 held-out PolitiFact claims across 3 independent random seeds (42, 123, 456).
  - All 9 classification metrics statistically significant under Welch's two-sample $t$-test ($p < 0.05$).
  - Fake news $F_1$: $+1.84\%$ ($84.36\% \pm 0.45\%$ vs. $82.52\% \pm 0.43\%$, $p=0.0162$).
  - Macro recall: $+1.52\%$ ($86.76\% \pm 0.42\%$ vs. $85.24\% \pm 0.41\%$, $p=0.0215$).
  - Test accuracy: $+1.13\%$ ($87.55\% \pm 0.38\%$ vs. $86.42\% \pm 0.38\%$, $p=0.0384$).
  - Training convergence: Fake $F_1$ of $47.31\%$ in Epoch 1 vs. $0.00\%$ for late fusion.
  - Continual streaming evaluated on 60 GB multi-part FakeNewsCorpus.

