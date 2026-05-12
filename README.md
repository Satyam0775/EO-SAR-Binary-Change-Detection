# EO-SAR-Binary-Change-Detection

**GalaxEye Space — Satellite AI Research Intern Technical Assessment**

---

## Project Overview

Binary pixel-level change detection on co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) image pairs. The model classifies each pixel as **Changed (1)** or **Unchanged (0)** between pre-event and post-event satellite imagery.

- **Architecture:** Siamese UNet — shared-weight dual encoder with absolute-difference bottleneck fusion
- **Loss:** BCE + Dice (combined) with `pos_weight=62.71` to address class imbalance
- **Training:** Google Colab Free Tier — Tesla T4 GPU — 5 epochs
- **Dataset:** Provided dataset only — no external data used

---

## Repository Structure

```
EO-SAR-Binary-Change-Detection/
├── config.yaml                  ← all hyperparameters
├── run.py                       ← unified CLI entry point
├── requirements.txt
├── src/
│   ├── dataset.py               ← EO-SAR dataloader + label remapping
│   ├── model.py                 ← SiameseUNet + LightweightUNet
│   ├── losses.py                ← BCE, Dice, BCE+Dice
│   ├── metrics.py               ← IoU, Precision, Recall, F1, Confusion Matrix
│   ├── train.py                 ← training pipeline
│   ├── evaluate.py              ← evaluation + confusion matrix export
│   ├── visualize.py             ← prediction figure generation
│   ├── inference.py             ← single-pair inference
│   ├── utils.py                 ← seed, checkpoint, logger, early stopping
│   └── remap_labels.py          ← standalone label remapping utility
├── notebooks/
│   └── colab_training.ipynb     ← end-to-end Google Colab notebook
└── outputs/                     ← checkpoints, logs, predictions, visualizations
    ├── checkpoints/
    ├── logs/
    ├── predictions/
    └── visualizations/
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (or Google Colab T4)

```
torch==2.2.2
torchvision==0.17.2
rasterio==1.3.10
albumentations==1.4.10
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
PyYAML==6.0.1
tqdm==4.66.4
Pillow==10.3.0
```

---

## Environment Setup

**Conda (recommended):**
```bash
conda create -n galaxeye python=3.10 -y
conda activate galaxeye
conda install -c conda-forge gdal rasterio -y
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install albumentations==1.4.10 seaborn==0.13.2 PyYAML==6.0.1 tqdm==4.66.4 matplotlib==3.9.0
```

**pip only (Linux / Colab):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Structure

Place the provided dataset under the project root exactly as follows:

```
data/
├── train/
│   ├── pre-event/      ← pre-event .tif files
│   ├── post-event/     ← post-event .tif files (same filenames)
│   └── target/         ← annotation mask .tif files (same filenames)
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Files across `pre-event/`, `post-event/`, and `target/` must share **identical filenames**.

**Mandatory label remapping** (applied automatically during loading):

| Original | Value | → | Remapped | Value |
|----------|-------|---|----------|-------|
| Background | 0 | → | No-Change | 0 |
| Intact | 1 | → | No-Change | 0 |
| Damaged | 2 | → | Change | 1 |
| Destroyed | 3 | → | Change | 1 |

Verify labels before training:
```bash
python run.py remap --verify
```

---

## Training

```bash
python run.py train --config config.yaml
```

Resume from checkpoint:
```bash
python run.py train --config config.yaml --resume outputs/checkpoints/last_model.pth
```

Key hyperparameters (`config.yaml`):

| Parameter | Value |
|-----------|-------|
| image_size | 256 |
| batch_size | 4 |
| epochs | 10 |
| optimizer | AdamW |
| learning_rate | 0.0003 |
| scheduler | Cosine Annealing |
| loss | BCE + Dice |
| pos_weight | 62.71 |
| base_filters | 16 |
| seed | 42 |

---

## Evaluation

**Validation split:**
```bash
python run.py eval --split val --weights outputs/checkpoints/best_model.pth
```

**Test split:**
```bash
python run.py eval --split test --weights outputs/checkpoints/best_model.pth
```

**Custom data path:**
```bash
python run.py eval --split test \
    --weights outputs/checkpoints/best_model.pth \
    --data_path /path/to/test
```

Outputs saved to:
- `outputs/logs/eval_test.txt` — metrics summary
- `outputs/visualizations/confusion_matrix_test.png` — confusion matrix

---

## Visualization

```bash
python run.py viz --split test --weights outputs/checkpoints/best_model.pth --n_samples 10
```

Generates 5-panel figures per sample:

`Pre-event | Post-event | Ground Truth | Change Probability | TP/FP/FN Error Map`

Saved to `outputs/visualizations/test/`.

---

## Model Weights

Download the trained checkpoint and place at `outputs/checkpoints/best_model.pth`:

**[Download best_model.pth — MODEL_WEIGHTS_LINK]**

---

## Results

Metrics computed on the **change class (label = 1)** as required.

| Metric | Validation | Test |
|--------|-----------|------|
| IoU | 0.0227 | 0.0122 |
| Precision | 0.0254 | 0.0129 |
| Recall | 0.1768 | 0.1709 |
| F1 Score | 0.0444 | 0.0240 |

**Notes on results:**

- Training was limited to **5 epochs** due to Google Colab free-tier runtime constraints
- The model demonstrates reasonable **recall (~17%)** — it is detecting changed regions — but low precision indicates significant false positives
- Metrics are expected to improve substantially with longer training (20–50 epochs) and available compute
- `pos_weight=62.71` was computed from the training set to address the severe class imbalance inherent in disaster change detection datasets
- No external datasets or pretrained weights were used

---

*GalaxEye Space — Satellite AI Research Intern Technical Assessment*
