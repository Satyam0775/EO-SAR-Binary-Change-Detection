# ============================================================
# losses.py
# GalaxEye Space — Loss Functions for Binary Change Detection
# ============================================================
# Implements:
#   1. WeightedBCELoss   — standard BCE with pos_weight for
#                          class-imbalance handling
#   2. DiceLoss          — overlap-based loss; robust to imbalance
#   3. BCEDiceLoss       — weighted combination of both
#
# WHY COMBINED BCE + DICE?
#   • BCE optimises pixel-wise probabilities (calibration).
#   • Dice optimises the overlap ratio (IoU-proxy), which is
#     directly related to the evaluation metric.
#   • Combining them gives faster convergence and better F1/IoU
#     than either alone on imbalanced data.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with optional positive-class weight.

    pos_weight > 1 penalises false negatives more than false positives,
    which is desirable when the change class is the rare class.
    A good starting value is:
        pos_weight ≈ (# no-change pixels) / (# change pixels)
    """

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model outputs (before sigmoid)
            targets : (B, H, W) integer labels {0, 1}
        Returns:
            scalar loss
        """
        # targets arrive as (B, 1, H, W) float32 — no unsqueeze needed
        targets_f = targets.float()
        if targets_f.dim() == 3:
            targets_f = targets_f.unsqueeze(1)   # safety fallback for (B,H,W)

        if self.pos_weight is not None:
            pw = torch.tensor(
                [self.pos_weight],
                dtype=torch.float32,
                device=logits.device,
            )
        else:
            pw = None

        return F.binary_cross_entropy_with_logits(logits, targets_f, pos_weight=pw)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    smooth=1 prevents division by zero and stabilises gradients
    when either the prediction or target is entirely zero (common
    in change detection where changed area is sparse).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model outputs
            targets : (B, H, W) integer labels {0, 1}
        Returns:
            scalar loss (1 - Dice)
        """
        # squeeze channel dim: (B,1,H,W) -> (B,H,W) for both probs and targets
        probs   = torch.sigmoid(logits).squeeze(1)    # (B, H, W)
        targets = targets.float()
        if targets.dim() == 4:
            targets = targets.squeeze(1)               # (B, 1, H, W) -> (B, H, W)

        # Flatten spatial dimensions for dot-product
        probs_f   = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)

        intersection = (probs_f * targets_f).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_f.sum(dim=1) + targets_f.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted combination: α·BCE + β·Dice

    Default α=β=0.5 gives equal weight.  Increase β to push
    harder on IoU/F1-style optimisation.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        assert abs(bce_weight + dice_weight - 1.0) < 1e-4, (
            "bce_weight + dice_weight should sum to 1.0"
        )
        self.bce  = WeightedBCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_w * bce_loss + self.dice_w * dice_loss


# ── Factory function ─────────────────────────────────────────

def build_loss(cfg: dict) -> nn.Module:
    """
    Instantiate loss from config.

    Reads:
        cfg["loss"]["type"]        : "bce" | "dice" | "bce_dice"
        cfg["loss"]["bce_weight"]  : float
        cfg["loss"]["dice_weight"] : float
        cfg["loss"]["pos_weight"]  : float or null
    """
    loss_type  = cfg["loss"]["type"]
    pos_weight = cfg["loss"].get("pos_weight", None)

    if loss_type == "bce":
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_type == "dice":
        return DiceLoss()

    elif loss_type == "bce_dice":
        bce_w  = cfg["loss"].get("bce_weight",  0.5)
        dice_w = cfg["loss"].get("dice_weight", 0.5)
        return BCEDiceLoss(
            bce_weight=bce_w,
            dice_weight=dice_w,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use bce | dice | bce_dice")


def compute_pos_weight(dataset) -> float:
    """
    Estimate pos_weight from a dataset by scanning all masks.
    pos_weight = n_neg / n_pos

    Call this before training if cfg["loss"]["pos_weight"] is null.
    Can be slow for large datasets — consider caching the result.
    """
    n_pos = 0
    n_neg = 0
    print("[Loss] Computing pos_weight from training data …")
    for sample in dataset:
        mask = sample["mask"]
        n_pos += (mask == 1).sum().item()
        n_neg += (mask == 0).sum().item()

    if n_pos == 0:
        print("[Loss] WARNING: No positive (change) pixels found in dataset!")
        return 1.0

    pw = n_neg / n_pos
    print(f"[Loss] pos_weight = {n_neg}/{n_pos} = {pw:.2f}")
    return pw