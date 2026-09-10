"""
Full-Stack Wiring Smoke Test
Verifies forward pass, backward pass, optimizer build, and evaluator.
Run from project root:
    python scripts/smoke_test.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from src.utils.config       import load_config
from src.utils.seed         import set_seed
from src.data.dataset       import get_loaders
from src.data.graph_builder import GraphBuilder
from src.models.model       import FakeNewsDetector
from src.training.trainer   import build_optimizer_scheduler
from src.training.evaluator import Evaluator


def check(label, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  — {extra}" if extra else ""))
    if not cond:
        raise AssertionError(f"Check failed: {label}")


def main():
    print("\n" + "="*55)
    print("  Full-Stack Wiring Smoke Test")
    print("="*55)

    # --- Config & seed ---
    cfg = load_config("experiments/baseline_m3dusa.yaml")
    set_seed(42)

    # --- DataLoaders ---
    print("\n[1] DataLoaders")
    train_loader, val_loader, test_loader = get_loaders(cfg)
    batch = next(iter(train_loader))
    B = batch["labels"].shape[0]
    check("input_ids shape",      batch["input_ids"].ndim == 2,      str(batch["input_ids"].shape))
    check("attention_mask shape",  batch["attention_mask"].ndim == 2, str(batch["attention_mask"].shape))
    check("labels shape",          batch["labels"].ndim == 1,          str(batch["labels"].shape))
    check("graph_batch present",   hasattr(batch["graph_batch"], "num_graphs"))
    check("graph_batch.num_graphs", batch["graph_batch"].num_graphs == B, str(B))

    # --- Model (late_fusion) ---
    print("\n[2] Model — late_fusion")
    metadata = GraphBuilder().metadata()
    model    = FakeNewsDetector(cfg, metadata)
    total, trainable = model.num_parameters()
    check("total params > 0",     total > 0,     f"{total:,}")
    check("trainable params > 0", trainable > 0, f"{trainable:,}")

    # --- Forward pass ---
    print("\n[3] Forward pass")
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    check("logits shape (B, 2)", logits.shape == (B, 2), str(logits.shape))
    check("no NaN in logits",    not torch.isnan(logits).any().item())

    # --- Backward pass ---
    print("\n[4] Backward pass")
    model.train()
    logits = model(batch)
    loss   = F.cross_entropy(logits, batch["labels"])
    loss.backward()
    check("loss is finite", loss.item() == loss.item(), f"loss={loss.item():.4f}")  # NaN != NaN
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    check("gradients flow", len(grad_norms) > 0, f"{len(grad_norms)} param groups with grads")

    # --- Optimizer & scheduler ---
    print("\n[5] Optimizer & scheduler")
    model.zero_grad()
    opt, sched = build_optimizer_scheduler(model, cfg, total_steps=100)
    check("optimizer param groups >= 2", len(opt.param_groups) >= 2, str(len(opt.param_groups)))
    opt.step()
    sched.step()
    check("scheduler step OK", True)

    # --- Evaluator ---
    print("\n[6] Evaluator (val split)")
    evaluator = Evaluator(device="cpu")
    metrics   = evaluator.evaluate(model, val_loader)
    required  = {"accuracy", "f1_macro", "f1_fake", "auc_roc", "loss"}
    check("all metrics present", required.issubset(metrics.keys()), str(set(metrics.keys())))
    check("accuracy in [0,1]",   0.0 <= metrics["accuracy"] <= 1.0, f"{metrics['accuracy']:.4f}")
    check("f1_macro in [0,1]",   0.0 <= metrics["f1_macro"] <= 1.0, f"{metrics['f1_macro']:.4f}")

    # --- Cross-modal fusion ---
    print("\n[7] CrossModalFusion forward pass")
    cfg2   = load_config("experiments/cross_modal_fusion.yaml")
    model2 = FakeNewsDetector(cfg2, metadata)
    model2.eval()
    with torch.no_grad():
        logits2 = model2(batch)
    check("cross_modal logits shape (B, 2)", logits2.shape == (B, 2), str(logits2.shape))
    check("no NaN (cross_modal)",            not torch.isnan(logits2).any().item())

    print("\n" + "="*55)
    print("  ALL CHECKS PASSED — ready to train!")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
