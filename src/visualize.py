# ============================================================
# visualize.py
# GalaxEye Space — Prediction Visualisation Utilities
# ============================================================
# Generates side-by-side figures:
#     pre-event | post-event | ground truth | prediction | overlay
#
# Usage (standalone):
#   python src/visualize.py \
#       --config configs/config.yaml \
#       --weights outputs/checkpoints/best_model.pth \
#       --split test \
#       --n_samples 10
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
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

from src.dataset import EOSARChangeDataset
from src.model   import build_model, model_forward
from src.utils   import load_config, load_checkpoint, get_device


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) float tensor to an (H, W, 3) uint8 RGB image
    suitable for matplotlib.  Uses first 3 channels if C >= 3,
    or replicates a single channel to RGB for grayscale images.
    """
    arr = tensor.cpu().numpy()     # (C, H, W)
    C   = arr.shape[0]

    if C >= 3:
        rgb = arr[:3]              # take first 3 bands (e.g. R,G,B of EO)
    else:
        rgb = np.repeat(arr[:1], 3, axis=0)   # grayscale → repeat to RGB

    # Transpose to (H, W, 3) and clip to [0,1]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def visualise_sample(
    pre_img:      np.ndarray,    # (H, W, 3) uint8
    post_img:     np.ndarray,    # (H, W, 3) uint8
    true_mask:    np.ndarray,    # (H, W)    int
    pred_prob:    np.ndarray,    # (H, W)    float [0,1]
    fname:        str,
    save_path:    str,
    threshold:    float = 0.5,
):
    """
    Create and save a 5-panel figure for one sample.

    Panels:
      1. Pre-event image
      2. Post-event image
      3. Ground truth mask
      4. Predicted probability map
      5. Error overlay (TP=green, FP=red, FN=yellow)
    """
    pred_mask = (pred_prob >= threshold).astype(np.uint8)

    # Error map
    h, w = true_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    tp_mask = (true_mask == 1) & (pred_mask == 1)
    fp_mask = (true_mask == 0) & (pred_mask == 1)
    fn_mask = (true_mask == 1) & (pred_mask == 0)
    tn_mask = (true_mask == 0) & (pred_mask == 0)
    overlay[tp_mask] = [0,   200, 0  ]  # TP → green
    overlay[fp_mask] = [220, 0,   0  ]  # FP → red
    overlay[fn_mask] = [220, 220, 0  ]  # FN → yellow
    overlay[tn_mask] = [50,  50,  50 ]  # TN → dark grey

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Pre-event", "Post-event", "Ground Truth", "Pred Probability", "Error Map"]
    images = [pre_img, post_img, true_mask, pred_prob, overlay]
    cmaps  = [None,    None,     "gray",    "hot",      None]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap != "gray" else None)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Legend for error map
    legend_patches = [
        mpatches.Patch(color=(0, 200/255, 0),     label="TP"),
        mpatches.Patch(color=(220/255, 0, 0),     label="FP"),
        mpatches.Patch(color=(220/255, 220/255, 0), label="FN"),
        mpatches.Patch(color=(50/255, 50/255, 50/255), label="TN"),
    ]
    axes[4].legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.7)

    plt.suptitle(f"Sample: {fname}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_visualisation(
    model,
    dataset: EOSARChangeDataset,
    device: torch.device,
    cfg: dict,
    n_samples: int = 10,
    out_dir: str = "outputs/visualizations",
    threshold: float = 0.5,
):
    """
    Generate visualisation figures for n_samples from a dataset.
    Saves one PNG per sample.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    in_channels = cfg["model"]["in_channels"]
    indices = list(range(min(n_samples, len(dataset))))

    print(f"[Viz] Generating {len(indices)} visualisation figures → {out_dir}")

    for idx in indices:
        sample = dataset[idx]
        pre_t  = sample["pre_image"]                          # (C, H, W)
        post_t = sample["post_image"]                         # (C, H, W)
        pre_image  = pre_t.unsqueeze(0).to(device)            # (1, C, H, W)
        post_image = post_t.unsqueeze(0).to(device)           # (1, C, H, W)
        mask  = sample["mask"].squeeze().cpu().numpy()        # (H, W)
        fname = sample["fname"]

        # Predict
        logits = model_forward(
            model,
            pre_image,
            post_image,
            cfg,
        )   # (1, 1, H, W)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()   # (H, W)

        # Prepare RGB visualisation
        pre_rgb  = tensor_to_display(pre_t)
        post_rgb = tensor_to_display(post_t)

        save_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_viz.png")
        visualise_sample(pre_rgb, post_rgb, mask, pred_prob, fname, save_path, threshold)

        print(f"  [{idx+1}/{len(indices)}] Saved → {save_path}")

    print("[Viz] Done.")


def main(args):
    cfg    = load_config(args.config)
    device = get_device()

    img_size = cfg["dataset"]["image_size"]
    split    = args.split
    data_dir = cfg["dataset"][f"{split}_dir"]

    ds    = EOSARChangeDataset(data_dir, img_size, cfg, mode=split)
    model = build_model(cfg).to(device)
    load_checkpoint(args.weights, model, device=device)

    out_dir = os.path.join(cfg["outputs"]["viz_dir"], split)
    run_visualisation(model, ds, device, cfg, n_samples=args.n_samples, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise change detection predictions")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--weights",   required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split",     default="test", choices=["val", "test"])
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to visualise")
    args = parser.parse_args()
    main(args)