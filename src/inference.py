# ============================================================
# inference.py
# GalaxEye Space — Single-Image Inference Script
# ============================================================
# Run inference on a single pre/post image pair without needing
# ground truth.  Useful for demo, deployment, or blind test sets.
#
# Usage:
#   python src/inference.py \
#       --config  configs/config.yaml \
#       --weights outputs/checkpoints/best_model.pth \
#       --pre     /path/to/pre_event.tif \
#       --post    /path/to/post_event.tif \
#       --out_dir outputs/predictions/demo
# ============================================================

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling

from src.model  import build_model, model_forward
from src.utils  import load_config, load_checkpoint, get_device


def read_and_normalise_tif(path: str, target_size: int) -> np.ndarray:
    """
    Read a .tif file and return float32 (H, W, C) normalised to [0,1].
    """
    with rasterio.open(path) as src:
        data = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)   # (C, H, W)

    data = np.transpose(data, (1, 2, 0))   # → (H, W, C)

    for c in range(data.shape[2]):
        band = data[:, :, c]
        mn, mx = band.min(), band.max()
        data[:, :, c] = (band - mn) / (mx - mn) if mx > mn else 0.0

    return data


def preprocess(pre_path: str, post_path: str, img_size: int, cfg: dict) -> torch.Tensor:
    """
    Load and preprocess a pre/post pair into a model-ready tensor.

    Returns (1, 2*C, H, W) float tensor.
    """
    pre  = read_and_normalise_tif(pre_path,  img_size)   # (H,W,C)
    post = read_and_normalise_tif(post_path, img_size)   # (H,W,C)

    # (H,W,C) → (C,H,W)
    pre_t  = torch.from_numpy(pre.transpose(2, 0, 1)).float()
    post_t = torch.from_numpy(post.transpose(2, 0, 1)).float()

    # Concatenate along channel dimension
    combined = torch.cat([pre_t, post_t], dim=0)          # (2C, H, W)
    return combined.unsqueeze(0)                          # (1, 2C, H, W)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    pre_path: str,
    post_path: str,
    cfg: dict,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Run model inference on a single image pair.

    Returns:
        dict with keys:
            "prob_map"  : (H, W) float numpy array of change probabilities
            "pred_mask" : (H, W) binary uint8 mask
    """
    model.eval()
    img_size = cfg["dataset"]["image_size"]

    tensor = preprocess(pre_path, post_path, img_size, cfg).to(device)  # (1,2C,H,W)

    logits    = model_forward(model, {"image": tensor}, cfg)             # (1,1,H,W)
    prob_map  = torch.sigmoid(logits).squeeze().cpu().numpy()            # (H,W)
    pred_mask = (prob_map >= threshold).astype(np.uint8)

    return {"prob_map": prob_map, "pred_mask": pred_mask}


def save_outputs(
    pre_path:  str,
    post_path: str,
    result:    dict,
    out_dir:   str,
    stem:      str,
):
    """
    Save prediction artefacts:
      • <stem>_prob.npy   — raw probability map
      • <stem>_mask.npy   — binary prediction mask
      • <stem>_result.png — visual comparison figure
    """
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{stem}_prob.npy"),  result["prob_map"])
    np.save(os.path.join(out_dir, f"{stem}_mask.npy"),  result["pred_mask"])
    print(f"[Inference] Arrays saved to {out_dir}")

    # ── Build visual ─────────────────────────────────────────
    def load_rgb(path: str, size: int = 256) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read(
                out_shape=(src.count, size, size),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        data = np.transpose(data, (1, 2, 0))
        if data.shape[2] >= 3:
            rgb = data[:, :, :3]
        else:
            rgb = np.repeat(data[:, :, :1], 3, axis=2)
        rgb -= rgb.min()
        rgb /= (rgb.max() + 1e-7)
        return (rgb * 255).astype(np.uint8)

    pre_rgb  = load_rgb(pre_path)
    post_rgb = load_rgb(post_path)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(pre_rgb);                              axes[0].set_title("Pre-event")
    axes[1].imshow(post_rgb);                             axes[1].set_title("Post-event")
    axes[2].imshow(result["prob_map"], cmap="hot", vmin=0, vmax=1);
    axes[2].set_title("Change Probability")
    axes[3].imshow(result["pred_mask"], cmap="gray", vmin=0, vmax=1);
    axes[3].set_title("Predicted Mask")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Inference: {stem}", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{stem}_result.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Inference] Figure saved to {fig_path}")


def main(args):
    cfg    = load_config(args.config)
    device = get_device()

    model = build_model(cfg).to(device)
    load_checkpoint(args.weights, model, device=device)

    result = run_inference(model, args.pre, args.post, cfg, device, threshold=args.threshold)

    stem = os.path.splitext(os.path.basename(args.pre))[0]
    save_outputs(args.pre, args.post, result, args.out_dir, stem)

    n_change = int(result["pred_mask"].sum())
    n_total  = result["pred_mask"].size
    print(f"\n[Inference] Change pixels: {n_change:,} / {n_total:,} ({100*n_change/n_total:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single EO-SAR image pair")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--weights",   required=True,  help="Path to checkpoint .pth")
    parser.add_argument("--pre",       required=True,  help="Pre-event .tif path")
    parser.add_argument("--post",      required=True,  help="Post-event .tif path")
    parser.add_argument("--out_dir",   default="outputs/predictions/demo")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
