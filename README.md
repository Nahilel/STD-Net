# STD-Net: Structure-Texture Demixing Network for Retinal Image Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

## 📌 Overview
This repository contains a full implementation of **STD-Net**, a bimodal deep learning architecture designed for high-precision retinal image segmentation (vessels and optic disc/cup). Retinal images present significant segmentation challenges due to the complex mix of anatomical structures and fine textures or imaging noise.

This approach solves this by explicitly **demixing** the input image into its structural and textural components before performing segmentation. By isolating the "signal" (structure) from the "noise" (texture), the model achieves robust performance even on low-contrast datasets.

<p align="center">
  <img src="result.png" alt="Visual results of STD-Net showing demixing and segmentation" width="800">
</p>

---

## 📁 Repository Structure

```text
Retinal_Image_Segmentation/
├── basicsr/
│   ├── archs/                          # Model architectures & neural components
│   │   ├── __init__.py                 # Network builder
│   │   └── stdnet_arch.py              # STDNetFullModel, STDModule, TextureBlock, AdaptiveNorm, MNet
│   ├── data/                           # Datasets, dataloaders, and transforms
│   │   ├── __init__.py                 # Dataset builder & dataloaders
│   │   ├── transforms.py               # CLAHE, Polar transform, RetinalSegmentationTransform
│   │   ├── drive_dataset.py            # DRIVEDataset (Vessel segmentation)
│   │   ├── origa_dataset.py            # ORIGADataset (Optic Disc & Cup segmentation)
│   │   └── refuge_dataset.py           # REFUGEDataset (Optic Disc & Cup segmentation)
│   ├── losses/                         # Mathematical loss formulations
│   │   ├── __init__.py                 # Loss builder
│   │   └── stdnet_loss.py              # TVLoss, TextureL1Loss, MaskedBCELoss, MultiScaleSegLoss, STDNetLoss
│   ├── metrics/                        # Quantitative validation & test metrics
│   │   ├── __init__.py                 # Metric calculation
│   │   └── seg_metrics.py              # DRIVE metrics (Acc, AUC, Sen, Spe, IOU), Overlapping Error (OE)
│   ├── models/                         # Model lifecycle wrappers & managers
│   │   ├── __init__.py                 # Model builder
│   │   ├── base_model.py               # BaseModel (optimizers, schedulers, checkpointing)
│   │   └── stdnet_model.py             # STDNetModel (feed_data, optimize, test, validation, visuals)
│   └── utils/                          # Common utilities and registries
│       ├── __init__.py
│       ├── registry.py                 # Registry pattern
│       ├── options.py                  # YAML parser and configuration loader
│       ├── logger.py                   # Root logger, MessageLogger, and TensorBoard logger
│       ├── misc.py                     # Seed setting, directory management, resume helpers
│       └── img_util.py                 # tensor2img & multi-panel sample visualizer
├── options/                            # Experiment configurations
│   ├── train/
│   │   ├── train_stdnet_drive.yml      # Training configuration for DRIVE
│   │   ├── train_stdnet_origa.yml      # Training configuration for ORIGA
│   │   └── train_stdnet_refuge.yml     # Training configuration for REFUGE
│   └── test/
│       ├── test_stdnet_drive.yml       # Evaluation configuration for DRIVE
│       ├── test_stdnet_origa.yml       # Evaluation configuration for ORIGA
│       └── test_stdnet_refuge.yml      # Evaluation configuration for REFUGE
├── train.py                            # Unified training entry point
├── test.py                             # Unified evaluation entry point
├── infer.py                            # Lightweight inference script (single image or folder)
├── requirements.txt                    # Project dependencies
├── setup.py                            # Package setup
├── retinal_segmentation_basicsr.ipynb  # Interactive demonstration notebook
└── retinal_segmentation.ipynb          # Original notebook
```

---

## 🏗️ Architecture Design

The network is composed of two primary modules that work in a unified pipeline:

### 1. Structure-Texture Demixing (STD) Module
Based on the principle that an image $I$ is the sum of a structure $S$ and a texture $T$ ($I = S + T$):
* **Texture Extraction**: 10 convolutional layers (64 filters each) with LeakyReLU activations ($\alpha=0.01$) to isolate the textural component.
* **Texture Block**: A refinement stage that extracts residual structural information from the texture using **Adaptive Normalization** and $1 \times 1$ convolutions:
  $$\Psi(x) = \lambda x + \mu \text{BN}(x)$$

### 2. M-Net Segmentation Backbone
The extracted structure is fed into an **M-Net** architecture:
* **Multi-scale Input**: The structure is down-sampled and fed into multiple levels of the encoder.
* **Texture Fusion**: The level-8 decoder representations are fused directly with extracted texture features.
* **Deep Supervision**: The decoder produces grayscale outputs at every level ($L_5, L_6, L_7, L_{final}$), which are up-sampled and fused to create the final segmentation mask.

---

## 🧪 Mathematical Framework

The model is optimized using a composite loss function that enforces both demixing quality and segmentation accuracy:

$$\mathcal{L}_{total}(S,T,R) = \mathcal{L}_{seg}(R) + \mu(\mathcal{L}_{t}(T) + \lambda\mathcal{L}_{s}(S))$$

* **Segmentation Loss ($\mathcal{L}_{seg}$)**: A weighted Cross-Entropy loss across multiple output scales:
  $$\mathcal{L}_{seg} = \frac{1.0\mathcal{L}_{final} + 0.3\mathcal{L}_{5} + 0.1\mathcal{L}_{6} + 0.05\mathcal{L}_{7}}{1.0 + 0.3 + 0.1 + 0.05}$$
* **Texture Loss ($\mathcal{L}_{t}$)**: $L_1$ norm to encourage sparsity:
  $$\mathcal{L}_{t}(T) = \frac{1}{CHW} \sum |T|$$
* **Structure Loss ($\mathcal{L}_{s}$)**: Total Variation (TV) loss to ensure spatial smoothness:
  $$\mathcal{L}_{s}(S) = \sqrt{\frac{1}{C(H-1)W}\sum_{i,j}(S_{i+1,j}-S_{i,j})^2 + \frac{1}{CH(W-1)}\sum_{i,j}(S_{i,j+1}-S_{i,j})^2}$$

---

## 🛠️ Implementation & Preprocessing Details

* **Vessel Segmentation (DRIVE)**: Images are resized to $512 \times 512$ and processed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance fine vessels. Metrics are strictly computed inside the Field of View (FOV).
* **Disc/Cup Segmentation (ORIGA/REFUGE)**: Images and masks can be converted to **Polar Coordinates** around the optic disc center, linearizing the radial boundary and significantly easing cup/disc boundary delineation.
* **Optimizer & Scheduling**: Adam optimizer with initial $lr = 10^{-3}$, modulated by a `ReduceLROnPlateau` scheduler (factor 0.5, patience 10).
* **Data Augmentations**: Random horizontal/vertical flips and rotations ($\pm 15^\circ$ for DRIVE/ORIGA, $\pm 5^\circ$ for REFUGE).

---

## 📊 Performance & Results

### Quantitative Metrics (DRIVE Dataset)
| Accuracy | AUC (ROC) | Sensitivity | Specificity | IoU |
| :---: | :---: | :---: | :---: | :---: |
| **0.9448** | **0.9663** | **0.7163** | **0.9781** | **0.6228** |

### Disc / Cup Segmentation (ORIGA & REFUGE)
Evaluated via **Overlapping Error** (OE = $1 - \text{IoU}$):
* $\text{OE}_{disc} = 1 - \text{IoU}_{disc}$
* $\text{OE}_{cup} = 1 - \text{IoU}_{cup}$
* $\text{OE}_{total} = \text{OE}_{disc} + \text{OE}_{cup}$

---

## 🚀 Quick Start Guide

### 1. Installation
Install the project dependencies and the package:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Training
Launch training using any of the provided YAML configuration files:

```bash
# Train on DRIVE (vessel segmentation)
python train.py -opt options/train/train_stdnet_drive.yml

# Train on ORIGA (optic disc & cup segmentation)
python train.py -opt options/train/train_stdnet_origa.yml

# Train on REFUGE (optic disc & cup segmentation)
python train.py -opt options/train/train_stdnet_refuge.yml
```

Checkpoints, training logs, and periodic multi-panel sample visualizations are automatically saved in `experiments/<name>/`.

### 3. Testing & Evaluation
Evaluate a trained model checkpoint against the test set:
```bash
python test.py -opt options/test/test_stdnet_drive.yml
```

### 4. Direct Inference
Run inference on any image or directory of images without ground truth:
```bash
# Vessel segmentation
python infer.py -i path/to/retinal_image.tif -m experiments/STDNet_DRIVE/models/net_g_best.pth --task vessel --apply_clahe -o results/demo

# Disc and Cup segmentation
python infer.py -i path/to/fundus_image.jpg -m experiments/STDNet_ORIGA/models/net_g_best.pth --task disc_cup --use_polar -o results/demo
```

---

## 📜 Authors
Developed by **Nahil El Bezzari, Yassine Lazizi, Pascal Le, Anis Melaimi, and Nadir Nehili**.
