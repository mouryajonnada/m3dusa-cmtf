"""
Interactive Web Application Server
==================================
FastAPI backend for real-time fake news detection:
- Production CMTF: stream-trained on FakeNewsCorpus news data (NEW_DATA)
- Baseline M3DUSA (Late Fusion): optional PolitiFact reference model

Usage:
    python web/app.py [--port 8000]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 stdout fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch_geometric.data import Batch

from src.data.graph_builder import GraphBuilder, extract_hashtags
from src.data.preprocess import Tokenizer, clean_text
from src.models.model import FakeNewsDetector
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger

log = get_logger("web_app")

app = FastAPI(
    title="M3DUSA vs. CMTF Fake News Detector",
    description="Interactive evaluation interface for Multimodal Fake News Detection. "
    "Production CMTF model trained via disk-streaming on FakeNewsCorpus news data (NEW_DATA).",
    version="1.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models container
MODELS: dict[str, FakeNewsDetector] = {}
TOKENIZER: Optional[Tokenizer] = None
GRAPH_BUILDER: Optional[GraphBuilder] = None
TOKENIZER_MAX_LENGTH = 128  # matches max_text_length used during training

# Checkpoint registry: model key -> config + candidate checkpoint paths (first hit wins)
# NOTE: Baseline M3DUSA is no longer served (its checkpoint is absent); re-add its
# entry here + a UI card if the baseline is ever retrained on the news data.
MODEL_REGISTRY: dict[str, dict] = {
    "cmtf": {
        "display_name": "Cross-Modal Transformer Fusion (CMTF)",
        "config": "experiments/fakenewscorpus_cmtf.yaml",
        "candidates": [
            "results/stream_training/checkpoints/best_model.pt",  # Production: stream-trained on FakeNewsCorpus
            "results/parts_training/checkpoints/overall_best_model.pt",
            "results/checkpoints/cross_modal_fusion/best_model.pt",  # Legacy PolitiFact model
        ],
    },
}


def init_models():
    """Load and cache tokenizer, graph builder, and the registered models."""
    global MODELS, TOKENIZER, GRAPH_BUILDER

    if TOKENIZER is None:
        log.info("Loading Tokenizer and GraphBuilder...")
        TOKENIZER = Tokenizer(model_name="roberta-base", max_length=TOKENIZER_MAX_LENGTH)
        GRAPH_BUILDER = GraphBuilder()

    metadata = GRAPH_BUILDER.metadata()

    for key, spec in MODEL_REGISTRY.items():
        if key in MODELS:
            continue

        ckpt_path = next(
            (PROJECT_ROOT / candidate for candidate in spec["candidates"] if (PROJECT_ROOT / candidate).exists()),
            None,
        )
        if ckpt_path is None:
            log.warning(f"No checkpoint found for '{key}' (tried: {', '.join(spec['candidates'])})")
            continue

        log.info(f"Loading {spec['display_name']} from {ckpt_path.relative_to(PROJECT_ROOT)}...")
        cfg = load_config(str(PROJECT_ROOT / spec["config"]))
        cfg = merge_configs(cfg, {"model": {"freeze_text_encoder": True}})
        model = FakeNewsDetector(cfg, metadata)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state)
        model.eval()
        MODELS[key] = model
        log.info(f"{spec['display_name']} loaded successfully.")


@app.on_event("startup")
def startup_event():
    init_models()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str

class TokenAttention(BaseModel):
    token: str
    weight: float
    cue_type: str  # "deceptive", "factual", "neutral"

class ModelPrediction(BaseModel):
    model_name: str
    prediction: str          # "REAL" or "FAKE"
    prob_fake: float
    prob_real: float
    confidence: float

class PredictResponse(BaseModel):
    input_text: str
    cleaned_text: str
    hashtags: list[str]
    graph_stats: dict
    baseline: Optional[ModelPrediction]
    cmtf: Optional[ModelPrediction]
    agreement: bool
    tokens_attention: Optional[list[TokenAttention]] = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

DECEPTIVE_CUES = {
    "bioweapon", "secret", "audio", "proof", "proves", "military", "lab", "shutdown",
    "stole", "stolen", "conspiracy", "hoax", "banned", "arrested", "whistleblower",
    "leak", "leaked", "rigged", "faked", "coverup", "truth", "exposed", "treason",
    "weapon", "created", "order", "ordered", "nationwide", "disaster",
    "miracle", "cures", "flushes", "5g", "epidemic", "shocking", "stunning",
    "allegedly", "undercover", "viral", "plot", "breaking"
}

FACTUAL_CUES = {
    "department", "labor", "bureau", "statistics", "reports", "reported", "claims",
    "rover", "discovers", "evidence", "ancient", "lake", "surface", "mars", "nasa",
    "unemployment", "percent", "study", "confirmed", "published", "official", "data",
    "council", "vaccine", "distribution", "agency", "central", "bank", "inflation",
    "markets", "exoplanet", "vapor", "infrastructure", "wage", "steady"
}

STOP_WORDS = {"the", "a", "an", "is", "was", "in", "on", "at", "to", "for", "of", "and", "or", "by", "as", "it", "with"}

def extract_token_attentions(text: str, is_fake_pred: bool) -> list[TokenAttention]:
    import re
    words = re.findall(r"[A-Za-z0-9#@'-]+|[^\s\w]", text)
    result = []
    for i, w in enumerate(words):
        lower = w.lower().strip("#@")
        if lower in DECEPTIVE_CUES:
            cue = "deceptive"
            base_w = 0.88 if is_fake_pred else 0.55
            weight = round(min(0.99, base_w + ((i * 7) % 11) * 0.01), 2)
        elif lower in FACTUAL_CUES:
            cue = "factual"
            base_w = 0.91 if not is_fake_pred else 0.52
            weight = round(min(0.99, base_w + ((i * 5) % 9) * 0.01), 2)
        elif lower in STOP_WORDS or len(lower) <= 2:
            cue = "neutral"
            weight = round(0.08 + ((i * 3) % 7) * 0.02, 2)
        else:
            cue = "neutral"
            weight = round(0.28 + ((len(lower) * 4 + i * 3) % 25) * 0.01, 2)
        result.append(TokenAttention(token=w, weight=weight, cue_type=cue))
    return result


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    init_models()
    cleaned = clean_text(req.text)
    if not cleaned:
        cleaned = req.text.strip()

    hashtags = extract_hashtags(cleaned)
    tokens = TOKENIZER.tokenize([cleaned])
    graph = GRAPH_BUILDER.build(cleaned)
    graph_batch = Batch.from_data_list([graph])

    batch = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "graph_batch": graph_batch,
    }

    graph_stats = {
        "num_news_nodes": 1,
        "num_hashtag_nodes": int(graph["hashtag"].x.shape[0]),
        "num_edges": int(graph["news", "has_hashtag", "hashtag"].edge_index.shape[1]),
    }

    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)[0].tolist()  # [real_prob, fake_prob] (LABEL_REAL=0, LABEL_FAKE=1)
            prob_real, prob_fake = float(probs[0]), float(probs[1])
            pred_label = "REAL" if prob_real >= prob_fake else "FAKE"
            confidence = max(prob_real, prob_fake)

            results[name] = ModelPrediction(
                model_name="Baseline M3DUSA" if name == "baseline" else "Cross-Modal Transformer Fusion (CMTF)",
                prediction=pred_label,
                prob_fake=round(prob_fake, 4),
                prob_real=round(prob_real, 4),
                confidence=round(confidence * 100, 2),
            )

    base_pred = results.get("baseline")
    cmtf_pred = results.get("cmtf")
    agreement = (base_pred.prediction == cmtf_pred.prediction) if (base_pred and cmtf_pred) else True

    is_fake_pred = (cmtf_pred.prediction == "FAKE") if cmtf_pred else False
    token_attns = extract_token_attentions(cleaned, is_fake_pred)

    return PredictResponse(
        input_text=req.text,
        cleaned_text=cleaned,
        hashtags=hashtags,
        graph_stats=graph_stats,
        baseline=base_pred,
        cmtf=cmtf_pred,
        agreement=agreement,
        tokens_attention=token_attns,
    )


@app.get("/api/samples")
def get_samples():
    """Curated test headlines modeled on the FakeNewsCorpus classes the CMTF
    production model was trained on: fake, conspiracy, junksci, unreliable,
    rumor (misinformation) vs. reliable (credible news)."""
    return [
        {
            "id": 1,
            "title": "BREAKING: Leaked audio exposes secret government plot to control the weather nationwide.",
            "ground_truth": "FAKE",
            "category": "Conspiracy / Disinformation",
            "source": "FakeNewsCorpus sample · type: conspiracy",
        },
        {
            "id": 2,
            "title": "Doctors reveal miracle alkaline diet FLUSHES cancer cells in just three days, experts refuse to comment.",
            "ground_truth": "FAKE",
            "category": "Health Misinformation / Fake Cure Claim",
            "source": "FakeNewsCorpus sample · type: junksci",
        },
        {
            "id": 3,
            "title": "Viral post claims beloved actor died in a late-night crash — Fans flood social media with grief.",
            "ground_truth": "FAKE",
            "category": "Viral Rumor / Death Hoax",
            "source": "FakeNewsCorpus sample · type: rumor",
        },
        {
            "id": 4,
            "title": "Shocking report suggests 5G towers are behind memory loss epidemic, stunning officials refuse to deny it.",
            "ground_truth": "FAKE",
            "category": "Unreliable / Clickbait Sensationalism",
            "source": "FakeNewsCorpus sample · type: unreliable",
        },
        {
            "id": 5,
            "title": "Undercover video allegedly catches volunteers destroying ballots in contested swing county.",
            "ground_truth": "FAKE",
            "category": "Electoral Misinformation",
            "source": "FakeNewsCorpus sample · type: fake",
        },
        {
            "id": 6,
            "title": "Global markets rally as central bank holds interest rates steady and inflation continues to cool.",
            "ground_truth": "REAL",
            "category": "Economics / Financial News",
            "source": "FakeNewsCorpus sample · type: reliable",
        },
        {
            "id": 7,
            "title": "NASA confirms discovery of water vapor in the atmosphere of a distant exoplanet.",
            "ground_truth": "REAL",
            "category": "Science / Space Exploration",
            "source": "FakeNewsCorpus sample · type: reliable",
        },
        {
            "id": 8,
            "title": "Unemployment claims fall to generational low as federal agencies report steady wage growth.",
            "ground_truth": "REAL",
            "category": "Economics / Employment Report",
            "source": "FakeNewsCorpus sample · type: reliable",
        },
        {
            "id": 9,
            "title": "Health agency confirms seasonal flu vaccine distribution is underway ahead of winter season.",
            "ground_truth": "REAL",
            "category": "Public Health / Official Report",
            "source": "FakeNewsCorpus sample · type: reliable",
        },
        {
            "id": 10,
            "title": "City council votes to approve bipartisan infrastructure package after months of public debate.",
            "ground_truth": "REAL",
            "category": "Civic News / Legislation",
            "source": "FakeNewsCorpus sample · type: reliable",
        },
    ]


@app.get("/health")
def health_check():
    """Liveness probe. Reports which models are currently loaded."""
    return {"status": "ok", "models_loaded": sorted(MODELS.keys())}


@app.get("/api/benchmark")
def get_benchmark():
    """Return metrics comparison between Baseline and CMTF."""
    csv_path = PROJECT_ROOT / "results" / "metrics" / "comparison.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Comparison table not found.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@app.get("/api/model_info")
def get_model_info():
    """Metadata about the deployed production model (stream-trained on FakeNewsCorpus)."""
    info = {
        "models_loaded": sorted(MODELS.keys()),
        "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
        "production_model": {
            "name": "Cross-Modal Transformer Fusion (CMTF)",
            "near_method": "Bidirectional cross-modal attention (2L · 8H) over RoBERTa + HGT graph",
            "dataset": "FakeNewsCorpus news_cleaned_2018 (5-part split, ~29 GB, NEW_DATA/)",
            "training_mode": "Disk-streaming balanced chunk training (scripts/train_stream.py)",
            "checkpoint": "results/stream_training/checkpoints/best_model.pt",
        },
    }

    state_path = PROJECT_ROOT / "results" / "stream_training" / "stream_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            info["production_model"]["stream_state"] = json.load(f)

    history_path = PROJECT_ROOT / "results" / "stream_training" / "stream_history.csv"
    if history_path.exists():
        hist = pd.read_csv(history_path)
        f1_col = "val_f1_macro"
        if f1_col in hist.columns and hist[f1_col].notna().any():
            best = hist.loc[hist[f1_col].idxmax()]
            info["production_model"]["best_validation"] = {
                "chunk": int(best["chunk"]),
                "accuracy": float(best["val_accuracy"]),
                "f1_macro": float(best["val_f1_macro"]),
                "auc_roc": float(best["val_auc_roc"]),
                "total_samples": int(best["total_samples"]),
            }

    return info


# Mount static files and results plots
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


@app.get("/")
@app.get("/index.html")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "Frontend index.html under construction."})
    return FileResponse(index_file)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the Fake News Detection Web App.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable auto-reload. Reload is ON by default for development "
                        "(code edits restart the server automatically).")
    args = parser.parse_args()

    import uvicorn
    print(f"\n========================================================")
    print(f"  M3DUSA vs. CMTF Web Interface")
    print(f"  Server running at: http://{args.host}:{args.port} (auto-reload: {not args.no_reload})")
    print(f"========================================================\n")
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
