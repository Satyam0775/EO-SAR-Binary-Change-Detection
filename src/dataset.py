# ============================================================
# dataset.py
# GalaxEye Space — EO-SAR Change Detection Dataset Loader
# ============================================================
# Loads co-registered pre/post-event .tif image pairs and
# binary change masks.  Applies mandatory label remapping:
#   0 (Background) -> 0  |  1 (Intact)   -> 0
#   2 (Damaged)    -> 1  |  3 (Destroyed) -> 1
#
# Returns pre_image and post_image as SEPARATE tensors
# for SiameseUNet which needs two independent inputs.
# ============================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.enums import Resampling
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Label remapping table ────────────────────────────────────
REMAP = {0: 0, 1: 0, 2: 1, 3: 1}

# ── Default target channels ──────────────────────────────────
TARGET_CHANNELS = 3


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Apply mandatory 4-class -> binary label remapping."""
    out = np.zeros_like(mask, dtype=np.uint8)
    for orig, new in REMAP.items():
        out[mask == orig] = new
    return out


def read_tif(path: str, target_size: int, n_channels: int = TARGET_CHANNELS) -> np.ndarray:
    """
    Read a GeoTIFF and return float32 ndarray of shape (H, W, n_channels).

    - Reads ALL bands with src.read() — never collapses to 1 band
    - Resizes to target_size x target_size via rasterio resampling
    - Replicates bands if TIFF has fewer channels than n_channels
    - Truncates if TIFF has more channels than n_channels
    - Normalises each band independently to [0, 1]

    Returns:
        float32 ndarray (target_size, target_size, n_channels)
    """
    with rasterio.open(path) as src:
        # Read ALL bands at once — shape: (C_raw, H, W)
        data = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    # (C_raw, H, W) -> (H, W, C_raw)  required by albumentations
    data = np.transpose(data, (1, 2, 0))

    C_raw = data.shape[2]

    # Replicate bands if fewer than needed (e.g. 1-band -> 3-band)
    if C_raw < n_channels:
        repeats = (n_channels + C_raw - 1) // C_raw
        data    = np.repeat(data, repeats, axis=2)

    # Truncate to exactly n_channels
    data = data[:, :, :n_channels]

    # Per-band min-max normalisation to [0, 1]
    for c in range(data.shape[2]):
        band = data[:, :, c]
        mn, mx = float(band.min()), float(band.max())
        if mx > mn:
            data[:, :, c] = (band - mn) / (mx - mn)
        else:
            data[:, :, c] = 0.0

    return data   # (H, W, n_channels) float32


def read_mask_tif(path: str, target_size: int) -> np.ndarray:
    """
    Read single-band mask GeoTIFF.

    Returns:
        uint8 ndarray (H, W) with remapped binary labels.
    """
    with rasterio.open(path) as src:
        mask = src.read(
            1,
            out_shape=(target_size, target_size),
            resampling=Resampling.nearest,
        ).astype(np.int32)

    return remap_mask(mask)   # (H, W) uint8


# ── Albumentations pipelines ─────────────────────────────────

def get_train_transforms(cfg: dict) -> A.Compose:
    """Augmentation pipeline for training — applied identically to both images."""
    aug = cfg.get("augmentation", {})
    ops = []

    if aug.get("horizontal_flip", True):
        ops.append(A.HorizontalFlip(p=0.5))
    if aug.get("vertical_flip", True):
        ops.append(A.VerticalFlip(p=0.5))
    if aug.get("random_rotate_90", True):
        ops.append(A.RandomRotate90(p=0.5))
    if aug.get("random_brightness_contrast", True):
        ops.append(A.RandomBrightnessContrast(p=0.3))
    if aug.get("gaussian_noise", True):
        ops.append(A.GaussNoise(p=0.2))

    ops.append(ToTensorV2())

    # additional_targets ensures post image gets identical spatial transforms
    return A.Compose(ops, additional_targets={"post_image": "image"})


def get_val_transforms() -> A.Compose:
    """Minimal pipeline for val/test — no random transforms."""
    return A.Compose(
        [ToTensorV2()],
        additional_targets={"post_image": "image"},
    )


# ── Dataset class ────────────────────────────────────────────

class EOSARChangeDataset(Dataset):
    """
    PyTorch Dataset for EO-SAR binary change detection.

    Expected directory layout:
        split_dir/
            pre-event/   <- pre-event  .tif files
            post-event/  <- post-event .tif files (same filenames)
            target/      <- annotation .tif files (same filenames)

    Each __getitem__ returns:
        {
            "pre_image"  : FloatTensor (3, H, W)  -- pre-event image
            "post_image" : FloatTensor (3, H, W)  -- post-event image
            "mask"       : LongTensor  (H, W)     -- 0=no-change, 1=change
            "fname"      : str                    -- filename for logging
        }

    pre_image and post_image are returned SEPARATELY so that
    SiameseUNet can pass each through its own encoder branch.
    """

    def __init__(
        self,
        split_dir:  str,
        image_size: int,
        cfg:        dict,
        mode:       str = "train",
    ):
        self.split_dir  = split_dir
        self.image_size = image_size
        self.mode       = mode
        self.n_channels = cfg.get("model", {}).get("in_channels", TARGET_CHANNELS)

        self.pre_dir  = os.path.join(split_dir, "pre-event")
        self.post_dir = os.path.join(split_dir, "post-event")
        self.mask_dir = os.path.join(split_dir, "target")

        # Validate directories
        for d in [self.pre_dir, self.post_dir, self.mask_dir]:
            if not os.path.isdir(d):
                raise RuntimeError(
                    f"[Dataset] Directory not found: {d}\n"
                    f"Expected structure inside: {split_dir}\n"
                    "  pre-event/  post-event/  target/"
                )

        # Match filenames across all three folders
        pre_files  = set(f for f in os.listdir(self.pre_dir)  if f.endswith(".tif"))
        post_files = set(f for f in os.listdir(self.post_dir) if f.endswith(".tif"))
        mask_files = set(f for f in os.listdir(self.mask_dir) if f.endswith(".tif"))

        matched = sorted(pre_files & post_files & mask_files)

        if len(matched) == 0:
            raise RuntimeError(
                f"[Dataset] No matching .tif triplets found in {split_dir}.\n"
                "Ensure pre-event/, post-event/, target/ share identical filenames."
            )

        self.filenames = matched

        print(
            f"[Dataset] {mode}: {len(self.filenames)} samples | "
            f"channels: {self.n_channels} | "
            f"size: {image_size}x{image_size}"
        )

        if mode == "train":
            self.transforms = get_train_transforms(cfg)
        else:
            self.transforms = get_val_transforms()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]

        # ── Load images: (H, W, C) float32 ───────────────────
        pre_img  = read_tif(
            os.path.join(self.pre_dir,  fname),
            self.image_size,
            self.n_channels,
        )
        post_img = read_tif(
            os.path.join(self.post_dir, fname),
            self.image_size,
            self.n_channels,
        )

        # ── Load mask: (H, W) uint8 ───────────────────────────
        mask = read_mask_tif(
            os.path.join(self.mask_dir, fname),
            self.image_size,
        )

        # ── Apply augmentation ────────────────────────────────
        # Identical spatial transforms applied to both images and mask
        augmented = self.transforms(
            image=pre_img,           # (H, W, C) float32
            post_image=post_img,     # (H, W, C) float32
            mask=mask,               # (H, W)    uint8
        )

        # ToTensorV2: (H, W, C) -> (C, H, W) automatically
        pre_t  = augmented["image"].float()                    # (C, H, W)  float32
        post_t = augmented["post_image"].float()               # (C, H, W)  float32
        mask_t = augmented["mask"].float().unsqueeze(0)        # (1, H, W)  float32
        # Shape (1, H, W) matches logits (B, 1, H, W) after batching,
        # making it directly compatible with BCEWithLogitsLoss and Dice loss.

        return {
            "pre_image":  pre_t,    # FloatTensor (3, H, W)
            "post_image": post_t,   # FloatTensor (3, H, W)
            "mask":       mask_t,   # FloatTensor (1, H, W)
            "fname":      fname,
        }


# ── Quick sanity check ───────────────────────────────────────
if __name__ == "__main__":
    import yaml

    cfg_path = "config.yaml"
    if not os.path.isfile(cfg_path):
        cfg_path = "configs/config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds = EOSARChangeDataset(
        split_dir  = cfg["dataset"]["train_dir"],
        image_size = cfg["dataset"]["image_size"],
        cfg        = cfg,
        mode       = "train",
    )

    sample = ds[0]
    print(f"pre_image  shape : {sample['pre_image'].shape}")    # (3, 256, 256)
    print(f"post_image shape : {sample['post_image'].shape}")   # (3, 256, 256)
    print(f"mask       shape : {sample['mask'].shape}")         # (256, 256)
    print(f"mask       unique: {sample['mask'].unique()}")
    print(f"pre_image  dtype : {sample['pre_image'].dtype}")    # float32
    print(f"mask       dtype : {sample['mask'].dtype}")         # int64
    print(f"fname            : {sample['fname']}")