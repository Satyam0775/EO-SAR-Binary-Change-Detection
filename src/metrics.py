# ============================================================
# metrics.py
# GalaxEye Space — Evaluation Metrics for Binary Change Detection
# ============================================================
# All metrics are computed for the CHANGE class (label = 1)
# as required by the assignment.
#
# Metrics implemented:
#   • IoU (Intersection over Union / Jaccard Index)
#   • Precision
#   • Recall  (Sensitivity / True Positive Rate)
#   • F1 Score (Dice coefficient at threshold 0.5)
#   • Confusion Matrix (TP, FP, TN, FN)
# ============================================================

import numpy as np
import torch
from typing import Dict, Tuple


class MetricTracker:
    """
    Accumulates TP, FP, TN, FN over a full epoch or evaluation loop,
    then computes final metrics in one call.

    Usage:
        tracker = MetricTracker()
        for batch in dataloader:
            preds = model(batch)                  # (B,1,H,W) logits
            tracker.update(preds, batch["mask"])  # (B,H,W) targets
        results = tracker.compute()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Zero all accumulators."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits  : (B, 1, H, W) raw model outputs
            targets : (B, H, W) integer labels {0, 1}
        """
        with torch.no_grad():
            probs  = torch.sigmoid(logits).squeeze(1)      # (B, H, W)
            preds  = (probs >= self.threshold).long()      # (B, H, W)
            tgts   = targets.long()

            self.tp += ((preds == 1) & (tgts == 1)).sum().item()
            self.fp += ((preds == 1) & (tgts == 0)).sum().item()
            self.tn += ((preds == 0) & (tgts == 0)).sum().item()
            self.fn += ((preds == 0) & (tgts == 1)).sum().item()

    def compute(self) -> Dict[str, float]:
        """Return dict of all metrics computed for the change class."""
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        eps = 1e-7   # avoid division by zero

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = tp / (tp + fp + fn + eps)
        accuracy  = (tp + tn) / (tp + fp + tn + fn + eps)

        return {
            "iou":       round(iou,       4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "accuracy":  round(accuracy,  4),
            # raw counts for confusion matrix
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    def confusion_matrix(self) -> np.ndarray:
        """
        Return a 2×2 numpy confusion matrix:
            [[TN, FP],
             [FN, TP]]
        Rows = actual, Columns = predicted.
        """
        return np.array(
            [[self.tn, self.fp],
             [self.fn, self.tp]],
            dtype=np.int64,
        )


# ── Standalone numpy functions ───────────────────────────────
# Useful for post-hoc analysis on saved numpy arrays.

def compute_metrics_from_arrays(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics from numpy arrays.

    Args:
        pred_mask  : (H, W) float probabilities or binary {0,1}
        true_mask  : (H, W) integer labels {0, 1}
        threshold  : binarisation threshold for probabilities
    Returns:
        dict of metric_name → value
    """
    pred_binary = (pred_mask >= threshold).astype(np.int32)
    true_binary = true_mask.astype(np.int32)

    tp = int(((pred_binary == 1) & (true_binary == 1)).sum())
    fp = int(((pred_binary == 1) & (true_binary == 0)).sum())
    tn = int(((pred_binary == 0) & (true_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (true_binary == 1)).sum())

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    accuracy  = (tp + tn) / (tp + fp + tn + fn + eps)

    return {
        "iou":       round(iou,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "accuracy":  round(accuracy,  4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def print_metrics(metrics: Dict[str, float], split: str = ""):
    """Pretty-print a metrics dict."""
    header = f"── Metrics [{split}] " if split else "── Metrics "
    print(f"\n{header}{'─' * (50 - len(header))}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")
    print()


def print_confusion_matrix(metrics: Dict[str, float]):
    """Print a simple confusion matrix from a metrics dict."""
    tp, fp = metrics["tp"], metrics["fp"]
    fn, tn = metrics["fn"], metrics["tn"]

    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print(f"             Pred No-Change  Pred Change")
    print(f"  Actual 0 :   TN={tn:<10}  FP={fp}")
    print(f"  Actual 1 :   FN={fn:<10}  TP={tp}")
    print()


# ============================================================
# Logger helper for training.py
# ============================================================
def log_metrics(logger, metrics, split, epoch):
    """
    Log metrics during training/validation.
    Compatible with train.py expectations.
    """
    logger.info(
        f"[{split}] Epoch {epoch} | "
        f"IoU={metrics['iou']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}"
    )