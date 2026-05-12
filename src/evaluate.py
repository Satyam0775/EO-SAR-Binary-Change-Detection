# ============================================================
# evaluate.py
# GalaxEye Space — Evaluation Script
# ============================================================
# Usage:
#   # Evaluate on val split using config paths
#   python src/evaluate.py --config configs/config.yaml --weights outputs/checkpoints/best_model.pth --split val
#
#   # Evaluate on custom test directory
#   python src/evaluate.py --config configs/config.yaml \
#       --weights outputs/checkpoints/best_model.pth \
#       --data_path /path/to/test  --split test
# ============================================================

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset  import EOSARChangeDataset
from src.model    import build_model, model_forward
from src.metrics  import MetricTracker, print_metrics, print_confusion_matrix
from src.utils    import load_config, load_checkpoint, get_device


@torch.no_grad()
def evaluate(model, loader, tracker, device, cfg, save_preds: bool = False, pred_dir: str = None):
    """
    Run evaluation loop.

    Args:
        model      : trained model
        loader     : DataLoader for val or test split
        tracker    : MetricTracker instance (will be reset inside)
        device     : torch device
        cfg        : config dict
        save_preds : whether to save prediction arrays
        pred_dir   : directory to save predictions (if save_preds=True)
    """
    model.eval()
    tracker.reset()

    all_preds  = []
    all_masks  = []
    all_fnames = []

    for batch in loader:
        pre_images  = batch["pre_image"].to(device)    # (B, 3, H, W)
        post_images = batch["post_image"].to(device)   # (B, 3, H, W)
        masks       = batch["mask"].to(device)         # (B, 1, H, W)
        fnames      = batch["fname"]

        logits = model_forward(
            model,
            pre_images,
            post_images,
            cfg,
        )   # (B, 1, H, W)

        tracker.update(logits, masks)

        if save_preds:
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()   # (B, H, W)
            all_preds.extend(probs)
            all_masks.extend(masks.squeeze(1).cpu().numpy())         # (B, H, W)
            all_fnames.extend(fnames)

    metrics = tracker.compute()

    if save_preds and pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
        for fname, pred, mask in zip(all_fnames, all_preds, all_masks):
            stem = os.path.splitext(fname)[0]
            np.save(os.path.join(pred_dir, f"{stem}_pred.npy"), pred)
            np.save(os.path.join(pred_dir, f"{stem}_mask.npy"), mask)
        print(f"[Eval] Predictions saved to {pred_dir}")

    return metrics, tracker


def plot_confusion_matrix(metrics: dict, split: str, out_path: str):
    """Save a nicely formatted confusion matrix heatmap."""
    tp, fp = metrics["tp"], metrics["fp"]
    fn, tn = metrics["fn"], metrics["tn"]

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["No-Change (0)", "Change (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        annot_kws={"size": 13},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {split.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix saved to {out_path}")


def print_results_table(val_metrics: dict, test_metrics: dict):
    """Print a comparison table for the report."""
    header = f"\n{'─'*52}"
    print(header)
    print(f"{'Metric':<14} {'Validation':>14} {'Test':>14}")
    print(f"{'─'*52}")
    for key in ["iou", "precision", "recall", "f1"]:
        print(f"{key.capitalize():<14} {val_metrics[key]:>14.4f} {test_metrics[key]:>14.4f}")
    print(f"{'─'*52}")


def main(args):
    cfg = load_config(args.config)
    device = get_device()

    # Override data paths if --data_path is given
    if args.data_path:
        cfg["dataset"][f"{args.split}_dir"] = args.data_path

    img_size = cfg["dataset"]["image_size"]
    split    = args.split

    # ── Dataset ──────────────────────────────────────────────
    data_dir = cfg["dataset"][f"{split}_dir"]
    ds = EOSARChangeDataset(data_dir, img_size, cfg, mode=split)

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    load_checkpoint(args.weights, model, device=device)

    # ── Evaluate ──────────────────────────────────────────────
    tracker = MetricTracker()
    pred_dir = os.path.join(cfg["outputs"]["predictions_dir"], split)

    metrics, tracker = evaluate(
        model, loader, tracker, device, cfg,
        save_preds=args.save_preds,
        pred_dir=pred_dir if args.save_preds else None,
    )

    # ── Print results ─────────────────────────────────────────
    print_metrics(metrics, split=split.upper())
    print_confusion_matrix(metrics)

    # ── Save confusion matrix plot ────────────────────────────
    cm_path = os.path.join(cfg["outputs"]["viz_dir"], f"confusion_matrix_{split}.png")
    os.makedirs(cfg["outputs"]["viz_dir"], exist_ok=True)
    plot_confusion_matrix(metrics, split, cm_path)

    # ── Save summary text ─────────────────────────────────────
    summary_path = os.path.join(cfg["outputs"]["logs_dir"], f"eval_{split}.txt")
    os.makedirs(cfg["outputs"]["logs_dir"], exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"Split: {split}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"[Eval] Summary saved to {summary_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate change detection model")
    parser.add_argument("--config",    default="configs/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--weights",   required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split",     default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--data_path", default=None,
                        help="Override data directory for the chosen split")
    parser.add_argument("--save_preds", action="store_true",
                        help="Save prediction .npy arrays to outputs/predictions/")
    args = parser.parse_args()
    main(args)