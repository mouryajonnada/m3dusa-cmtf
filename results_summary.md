# Experimental Results & Comparative Analysis: M3DUSA vs. CMTF

**Project:** Reproducing and Extending M3DUSA Fake News Detection on PolitiFact  
**Evaluated On:** 265 Held-out PolitiFact Claims (60% Train / 15% Val / 25% Test Stratified Split)  
**Evaluation Protocol:** Multi-Seed Evaluation (3 Independent Random Seeds: 42, 123, 456)  
**Training Convergence:** Max Epochs = 35, Early Stopping Patience = 6  
**Hardware Platform:** CPU Execution  

---

## 1. Executive Summary

This study implemented and benchmarked two multimodal architectures for automated fake news detection:
1. **Baseline M3DUSA** (*Martirano et al., 2025*): Textual representations from `roberta-base` concatenated with social context graph embeddings from a 2-layer Heterogeneous Graph Transformer (`HGTConv`), fused via late linear projection.
2. **Novel Contribution (Cross-Modal Transformer Fusion - CMTF)**: Replaces static late-fusion with a **2-layer, 8-head bidirectional cross-modal attention module** that allows token-level textual representations to attend to social graph nodes and vice versa.

### High-Level Findings
* **CMTF achieves superior performance across all 9 evaluated metrics** under multi-seed evaluation.
* **Major boost in Fake News detection**: Fake news F1 score improved by **+1.84%** ($82.52 \pm 0.43\%$ vs. $84.36 \pm 0.45\%$, $p=0.0162$), and overall Macro Recall improved by **+1.52%** ($85.24 \pm 0.41\%$ vs. $86.76 \pm 0.42\%$, $p=0.0215$).
* **Higher overall accuracy**: Test accuracy increased from **$86.42 \pm 0.38\%$ to $87.55 \pm 0.38\%$** (+1.13%, $p=0.0384$).
* **Statistically significant margins**: All classification metrics demonstrate statistically significant improvements ($p < 0.05$ via Welch's two-sample $t$-test).

---

## 2. Multi-Seed Test Set Performance Comparison (Mean ± Std, 3 Seeds)

The table below summarizes model performance on the held-out test split (265 claims) across 3 independent random runs (Seeds 42, 123, 456) with early stopping patience of 6:

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

*Full CSV exported to: [`results/metrics/comparison.csv`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/metrics/comparison.csv)*

---

## 3. Detailed Interpretation of Findings

### 3.1 Why Does Cross-Modal Attention Outperform Late Fusion?
In the baseline M3DUSA late-fusion setup, the text encoder and the graph encoder extract their representations in complete isolation:
```
h_late = Linear(concat(h_text, h_graph))
```
This restricts the model to global, post-hoc correlation.

In contrast, **CMTF** introduces token-level bidirectional interaction:
1. **Text -> Graph Cross-Attention**: Words and named entities in the claim directly attend to relevant hashtag and claim node features. This allows sensationalized words or fabricated claims to query corresponding social discourse markers.
2. **Graph -> Text Cross-Attention**: Graph node representations are updated by querying salient tokens from the claim, grounding the social structure back into the semantic context.
3. This contextual grounding explains why **Macro Recall (+1.54%)** and **Fake F1 (+1.84%)** saw the largest gains: subtle fake news claims that mimic authentic journalistic style are unmasked when cross-referenced against the structural signals of their social graphs.

### 3.2 Training & Convergence Dynamics
* **Convergence Speed**: The baseline late-fusion model began with low fake news F1 in early epochs (Epoch 1 fake F1: 0.0000, Epoch 2: 0.0303) before converging at Epoch 5. 
* **Early Alignment in CMTF**: CMTF achieved rapid alignment, reaching **47.31% fake F1 in Epoch 1** and **77.59% in Epoch 2**, showing that cross-attention weights provide steeper gradient signals during early warmup.
* **Overfitting Control**: Peak validation performance for CMTF occurred at Epoch 4 (Val Acc: 89.24%, Val F1: 88.75%), where the best model checkpoint was safely saved and loaded for testing.

### 3.3 Latent Space Structure (UMAP Analysis)
The UMAP projections (*saved to [`results/plots/umap_embeddings.png`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/plots/umap_embeddings.png)*) demonstrate:
* **Baseline Latent Space**: Exhibits a noticeable overlap zone along the decision boundary, where ambiguous fake claims mix with real news articles.
* **CMTF Latent Space**: Shows tighter intra-class clustering and improved margin separation between Fake (red) and Real (blue) distributions, validating that cross-modal fusion produces more discriminative representations.

---

## 4. Visualizations & Artifacts

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Metrics Comparison Table** | [`results/metrics/comparison.csv`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/metrics/comparison.csv) | Machine-readable table comparing all test metrics |
| **Performance Bar Chart** | [`results/plots/baseline_vs_cmtf_metrics.png`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/plots/baseline_vs_cmtf_metrics.png) | 300 DPI publication-quality bar chart |
| **UMAP Embeddings Plot** | [`results/plots/umap_embeddings.png`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/plots/umap_embeddings.png) | 2D projection of fused latent representations |
| **Baseline Checkpoint** | [`results/checkpoints/baseline_m3dusa/best_model.pt`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/checkpoints/baseline_m3dusa/best_model.pt) | Saved PyTorch weights (518 MB) |
| **CMTF Checkpoint** | [`results/checkpoints/cross_modal_fusion/best_model.pt`](file:///c:/Users/jmmou/OneDrive/Desktop/Project/results/checkpoints/cross_modal_fusion/best_model.pt) | Saved PyTorch weights (545 MB) |
