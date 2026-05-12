# ============================================================
# utils.py
# GalaxEye Space — Utility Functions
# ============================================================

import os
import random
import logging
import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import yaml


# ── Reproducibility ──────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    Fix all random seeds for full reproducibility.
    Call this before any data loading or model initialisation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Utils] Random seed fixed to {seed}")


# ── Config loading ───────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[Utils] Config loaded from {config_path}")
    return cfg


# ── Directory helpers ────────────────────────────────────────

def make_output_dirs(cfg: dict):
    """Create all output directories specified in config."""
    out = cfg["outputs"]
    for key in ["checkpoint_dir", "predictions_dir", "logs_dir", "viz_dir"]:
        os.makedirs(out[key], exist_ok=True)
    print("[Utils] Output directories ready.")


# ── Checkpoint utilities ─────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    cfg: dict,
    is_best: bool = False,
):
    """
    Save model + optimiser state to disk.

    Saves two files:
        last_model.pth  — always updated (for resuming)
        best_model.pth  — updated only when is_best=True
    """
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "metrics":      metrics,
        "cfg":          cfg,
    }

    last_path = os.path.join(ckpt_dir, cfg["outputs"]["last_model_name"])
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, cfg["outputs"]["best_model_name"])
        torch.save(state, best_path)
        print(f"[Checkpoint] ✓ Best model saved (epoch {epoch+1}): {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """
    Load model (and optionally optimiser) state from a checkpoint.

    Returns:
        start_epoch : epoch to resume from (next epoch index)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    print(f"[Checkpoint] Loaded from {path} — epoch {epoch+1}, metrics: {metrics}")
    return epoch + 1   # return next epoch to run


# ── Logger ───────────────────────────────────────────────────

def get_logger(name: str, log_dir: str) -> logging.Logger:
    """
    Set up a logger that writes to both console and a log file.
    File is named: log_YYYY-MM-DD_HH-MM-SS.txt
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(log_dir, f"log_{timestamp}.txt")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []   # reset handlers to avoid duplicate outputs

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                             datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    print(f"[Utils] Logging to {log_path}")
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict, split: str, epoch: int):
    """Log a metrics dict with epoch prefix."""
    logger.info(
        f"[{split}] Epoch {epoch:03d} | "
        f"IoU={metrics['iou']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Prec={metrics['precision']:.4f} | "
        f"Rec={metrics['recall']:.4f}"
    )


def save_metrics_json(metrics: Dict, path: str):
    """Append a metrics record to a JSON-lines file."""
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


# ── Device helper ────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU. Prints which is used."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[Utils] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Utils] Using CPU (no GPU found)")
    return device


# ── Early stopping ───────────────────────────────────────────

class EarlyStopping:
    """
    Monitors a validation metric and signals when to stop training.
    Tracks best value; counts epochs without improvement.

    Usage:
        es = EarlyStopping(patience=10, mode="max")   # for IoU/F1
        for epoch in range(epochs):
            val_iou = ...
            if es(val_iou):
                break   # stop training
    """

    def __init__(self, patience: int = 10, mode: str = "max", delta: float = 1e-4):
        """
        Args:
            patience : epochs to wait after last improvement
            mode     : "max" to track improving metric, "min" for loss
            delta    : minimum change to qualify as improvement
        """
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = -float("inf") if mode == "max" else float("inf")
        self.stop      = False

    def __call__(self, value: float) -> bool:
        """
        Returns True when training should stop.
        """
        improved = (
            value > self.best + self.delta
            if self.mode == "max"
            else value < self.best - self.delta
        )

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                print("[EarlyStopping] Stopping training.")
                self.stop = True

        return self.stop
