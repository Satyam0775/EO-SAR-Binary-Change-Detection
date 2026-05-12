# ============================================================
# train.py
# GalaxEye Space — Training Pipeline
# ============================================================
# Usage:
#   python src/train.py --config config.yaml
#   python src/train.py --config config.yaml --resume outputs/checkpoints/last_model.pth
# ============================================================

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.dataset  import EOSARChangeDataset
from src.model    import build_model, model_forward
from src.losses   import build_loss, compute_pos_weight
from src.metrics  import MetricTracker, print_metrics, log_metrics
from src.utils    import (
    set_seed, load_config, make_output_dirs,
    save_checkpoint, load_checkpoint,
    get_logger, get_device, EarlyStopping,
    save_metrics_json,
)


def build_optimizer(model, cfg):
    opt_name = cfg["training"]["optimizer"].lower()
    lr       = cfg["training"]["learning_rate"]
    wd       = cfg["training"].get("weight_decay", 1e-4)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg):
    sched_name = cfg["training"].get("scheduler", "none").lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"].get("scheduler_T_max", cfg["training"]["epochs"]),
            eta_min=1e-6,
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        return None


# ── Training epoch ───────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, tracker, device, cfg, grad_clip):
    """One full training pass. Returns average loss."""
    model.train()
    tracker.reset()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        # Dataset returns pre_image and post_image separately
        pre_images  = batch["pre_image"].to(device)   # (B, 3, H, W)
        post_images = batch["post_image"].to(device)  # (B, 3, H, W)
        masks       = batch["mask"].to(device)        # (B, H, W)

        optimizer.zero_grad()

        logits = model_forward(model, pre_images, post_images, cfg)  # (B, 1, H, W)
        loss   = loss_fn(logits, masks)

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        tracker.update(logits.detach(), masks)

        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}", end="\r")

    print()
    return total_loss / len(loader)


# ── Validation epoch ─────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, loss_fn, tracker, device, cfg):
    """Validation pass. Returns average loss."""
    model.eval()
    tracker.reset()
    total_loss = 0.0

    for batch in loader:
        pre_images  = batch["pre_image"].to(device)
        post_images = batch["post_image"].to(device)
        masks       = batch["mask"].to(device)

        logits = model_forward(model, pre_images, post_images, cfg)
        loss   = loss_fn(logits, masks)

        total_loss += loss.item()
        tracker.update(logits, masks)

    return total_loss / len(loader)


# ── Main training loop ────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    make_output_dirs(cfg)
    device       = get_device()
    logger       = get_logger("train", cfg["outputs"]["logs_dir"])
    metrics_path = os.path.join(cfg["outputs"]["logs_dir"], "metrics.jsonl")

    ds_cfg   = cfg["dataset"]
    img_size = ds_cfg["image_size"]

    train_ds = EOSARChangeDataset(ds_cfg["train_dir"], img_size, cfg, mode="train")
    val_ds   = EOSARChangeDataset(ds_cfg["val_dir"],   img_size, cfg, mode="val")

    nw = ds_cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )

    if cfg["loss"].get("pos_weight") is None:
        cfg["loss"]["pos_weight"] = compute_pos_weight(train_ds)

    model     = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn   = build_loss(cfg)

    start_epoch = 0
    best_val_f1 = 0.0

    resume = args.resume or cfg["training"].get("resume_checkpoint")
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, device)

    train_tracker = MetricTracker()
    val_tracker   = MetricTracker()
    early_stop    = EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"],
        mode="max",
    )

    grad_clip = cfg["training"].get("gradient_clip", 1.0)
    n_epochs  = cfg["training"]["epochs"]

    logger.info(f"Starting training — {n_epochs} epochs | device={device}")

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        print(f"\n{'─'*55}")
        print(f"Epoch {epoch+1}/{n_epochs}  |  LR={optimizer.param_groups[0]['lr']:.6f}")

        train_loss    = train_one_epoch(
            model, train_loader, optimizer, loss_fn, train_tracker, device, cfg, grad_clip
        )
        train_metrics = train_tracker.compute()

        val_loss    = validate(model, val_loader, loss_fn, val_tracker, device, cfg)
        val_metrics = val_tracker.compute()

        elapsed = time.time() - epoch_start

        log_metrics(logger, train_metrics, "Train", epoch + 1)
        log_metrics(logger, val_metrics,   "Val",   epoch + 1)
        logger.info(
            f"Train loss={train_loss:.4f} | Val loss={val_loss:.4f} | Time={elapsed:.1f}s"
        )

        save_metrics_json({"epoch": epoch+1, "split": "train", **train_metrics, "loss": train_loss}, metrics_path)
        save_metrics_json({"epoch": epoch+1, "split": "val",   **val_metrics,   "loss": val_loss},   metrics_path)

        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]

        save_checkpoint(model, optimizer, epoch, val_metrics, cfg, is_best=is_best)

        if scheduler is not None:
            scheduler.step()

        if early_stop(val_metrics["f1"]):
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")
    print(f"\n✓ Best model: {cfg['outputs']['checkpoint_dir']}/{cfg['outputs']['best_model_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary change detection model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    main(args)