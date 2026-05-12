# ============================================================
# model.py
# GalaxEye Space — Model Definitions
# ============================================================
# Implements two architectures:
#   1. SiameseUNet      — dual-encoder, processes pre and post
#                         images separately, fuses at bottleneck
#   2. LightweightUNet  — single-encoder, receives concatenated
#                         pre+post internally
#
# model_forward() accepts pre and post as SEPARATE tensors,
# consistent with dataset.py returning "pre_image"/"post_image".
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building Blocks ──────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two (Conv -> BN -> ReLU) layers. Standard UNet block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool -> DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample -> concat skip -> DoubleConv (decoder step)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


# ── Architecture 1: Siamese UNet ────────────────────────────

class SiameseUNet(nn.Module):
    """
    Siamese UNet with shared-weight encoders.

    Accepts TWO separate tensors: pre (B, C, H, W) and post (B, C, H, W).
    Fuses at bottleneck via absolute difference + sum concatenation.
    Returns (B, 1, H, W) raw logits.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 16):
        super().__init__()
        f = base_filters

        # Shared encoder — identical weights applied to pre and post
        self.enc1           = DoubleConv(in_channels, f)
        self.enc2           = Down(f,      f * 2)
        self.enc3           = Down(f * 2,  f * 4)
        self.enc4           = Down(f * 4,  f * 8)
        self.bottleneck_enc = Down(f * 8,  f * 16)

        # Bottleneck fusion: |pre - post| + (pre + post) => 2 * f*16
        self.fusion_conv = DoubleConv(f * 16 * 2, f * 16)

        # Decoder — uses averaged skip connections from both branches
        self.dec4 = Up(f * 16, f * 8,  f * 8)
        self.dec3 = Up(f * 8,  f * 4,  f * 4)
        self.dec2 = Up(f * 4,  f * 2,  f * 2)
        self.dec1 = Up(f * 2,  f,      f)

        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def _encode(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bn = self.bottleneck_enc(s4)
        return bn, s1, s2, s3, s4

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre  : (B, C, H, W) pre-event image tensor
            post : (B, C, H, W) post-event image tensor
        Returns:
            (B, 1, H, W) raw logits
        """
        bn_pre,  s1_pre,  s2_pre,  s3_pre,  s4_pre  = self._encode(pre)
        bn_post, s1_post, s2_post, s3_post, s4_post = self._encode(post)

        # Absolute difference explicitly encodes change signal
        diff  = torch.abs(bn_pre - bn_post)
        fused = self.fusion_conv(torch.cat([bn_pre + bn_post, diff], dim=1))

        # Average skip connections across both encoder branches
        s1 = (s1_pre + s1_post) / 2
        s2 = (s2_pre + s2_post) / 2
        s3 = (s3_pre + s3_post) / 2
        s4 = (s4_pre + s4_post) / 2

        d4 = self.dec4(fused, s4)
        d3 = self.dec3(d4,    s3)
        d2 = self.dec2(d3,    s2)
        d1 = self.dec1(d2,    s1)

        return self.out_conv(d1)   # (B, 1, H, W)


# ── Architecture 2: Lightweight UNet ────────────────────────

class LightweightUNet(nn.Module):
    """
    Single-encoder UNet. Concatenates pre and post internally.
    Input: two separate (B, C, H, W) tensors.
    Returns (B, 1, H, W) raw logits.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 16):
        """
        Args:
            in_channels  : channels per image (NOT the concatenated total)
            base_filters : width multiplier
        """
        super().__init__()
        f          = base_filters
        combined_c = in_channels * 2   # pre + post concatenated

        self.enc1       = DoubleConv(combined_c, f)
        self.enc2       = Down(f,      f * 2)
        self.enc3       = Down(f * 2,  f * 4)
        self.enc4       = Down(f * 4,  f * 8)
        self.bottleneck = Down(f * 8,  f * 16)

        self.dec4 = Up(f * 16, f * 8,  f * 8)
        self.dec3 = Up(f * 8,  f * 4,  f * 4)
        self.dec2 = Up(f * 4,  f * 2,  f * 2)
        self.dec1 = Up(f * 2,  f,      f)

        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre  : (B, C, H, W)
            post : (B, C, H, W)
        Returns:
            (B, 1, H, W) raw logits
        """
        x  = torch.cat([pre, post], dim=1)   # (B, 2*C, H, W)

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bn = self.bottleneck(s4)

        d4 = self.dec4(bn, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)   # (B, 1, H, W)


# ── Factory function ─────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate model from config.

    cfg["model"]["architecture"] : "siamese_unet" | "lightweight_unet"
    cfg["model"]["in_channels"]  : bands per image (e.g. 3 for RGB)
    cfg["model"]["base_filters"] : width multiplier
    """
    arch         = cfg["model"]["architecture"]
    in_channels  = cfg["model"]["in_channels"]
    base_filters = cfg["model"].get("base_filters", 16)

    if arch == "siamese_unet":
        model = SiameseUNet(in_channels=in_channels, base_filters=base_filters)
    elif arch == "lightweight_unet":
        model = LightweightUNet(in_channels=in_channels, base_filters=base_filters)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use: siamese_unet | lightweight_unet")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {arch} | {n_params:,} trainable parameters")
    return model


# ── Forward-pass wrapper ─────────────────────────────────────

def model_forward(
    model: nn.Module,
    pre:   torch.Tensor,
    post:  torch.Tensor,
    cfg:   dict,
) -> torch.Tensor:
    """
    Unified forward pass for both architectures.

    Both SiameseUNet and LightweightUNet now accept separate
    pre and post tensors directly — no splitting needed.

    Args:
        model : built model instance
        pre   : (B, C, H, W) pre-event image tensor  (already on device)
        post  : (B, C, H, W) post-event image tensor (already on device)
        cfg   : config dict (kept for API consistency)

    Returns:
        (B, 1, H, W) raw logits
    """
    return model(pre, post)


# ── Quick sanity check ───────────────────────────────────────
if __name__ == "__main__":
    import yaml, os

    cfg_path = "config.yaml" if os.path.isfile("config.yaml") else "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg).to(device)

    C  = cfg["model"]["in_channels"]
    sz = cfg["dataset"]["image_size"]
    B  = 2

    pre  = torch.randn(B, C, sz, sz).to(device)
    post = torch.randn(B, C, sz, sz).to(device)

    logits = model_forward(model, pre, post, cfg)
    print(f"Input  pre  : {pre.shape}")
    print(f"Input  post : {post.shape}")
    print(f"Output logits: {logits.shape}")   # expect (2, 1, 256, 256)