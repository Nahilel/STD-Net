import os
from pathlib import Path
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor2img(tensor: torch.Tensor, min_max: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Convert torch tensor to numpy uint8 array [H, W, C] or [H, W].

    Args:
        tensor: [C, H, W] or [H, W] or [1, H, W]
        min_max: (min, max) range for normalization
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    tensor = tensor.detach().cpu().float()

    if min_max is not None:
        t_min, t_max = min_max
        tensor = (tensor - t_min) / (t_max - t_min + 1e-8)
    else:
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)

    img_np = tensor.numpy()
    if img_np.ndim == 3:  # [C, H, W] -> [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))

    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def save_drive_visual_sample(
    structure: torch.Tensor,
    texture: torch.Tensor,
    extracted_structure: torch.Tensor,
    prob: torch.Tensor,
    fov: Optional[torch.Tensor],
    save_path: str,
    title_prefix: str = "DRIVE",
    threshold: Optional[float] = 0.5,
):
    """Save 4-panel DRIVE visualization figure:
    1) Structure S
    2) Texture |T|
    3) Extracted structure (Green channel)
    4) Segmentation mask
    """
    S = structure.detach().cpu()
    T = texture.detach().cpu()
    E = extracted_structure.detach().cpu()
    P = prob.detach().cpu()
    if P.ndim == 3:
        P = P[0]

    H, W = P.shape
    if fov is not None:
        F = fov.detach().cpu()
        if F.ndim == 3:
            F = F[0]
    else:
        F = torch.ones_like(P)

    # 1) Structure S
    S_vis = S.clone()
    s_min, s_max = S_vis.min(), S_vis.max()
    S_vis = (S_vis - s_min) / (s_max - s_min + 1e-8)
    S_vis = S_vis * F.unsqueeze(0)
    S_vis = S_vis.permute(1, 2, 0).numpy()

    # 2) Texture |T|
    T_abs = T.abs().mean(0)
    t_min, t_max = T_abs.min(), T_abs.max()
    T_vis = (T_abs - t_min) / (t_max - t_min + 1e-8)
    T_vis = (T_vis * F).numpy()

    # 3) Extracted structure - Green channel (index 1)
    E_green = E[1] if E.shape[0] > 1 else E[0]
    e_min, e_max = E_green.min(), E_green.max()
    E_vis = (E_green - e_min) / (e_max - e_min + 1e-8)
    E_vis = (E_vis * F).numpy()

    # 4) Segmentation
    if threshold is None:
        mask = (P * F).numpy()
    else:
        mask = ((P >= threshold).float() * F).numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(np.clip(S_vis, 0, 1))
    axes[0].set_title(f"{title_prefix} - Structure")
    axes[0].axis("off")

    axes[1].imshow(T_vis, cmap="gray")
    axes[1].set_title("Texture")
    axes[1].axis("off")

    axes[2].imshow(E_vis, cmap="gray")
    axes[2].set_title("Extracted structure (Green)")
    axes[2].axis("off")

    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Segmentation")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_origa_visual_sample(
    structure: torch.Tensor,
    texture: torch.Tensor,
    extracted_structure: torch.Tensor,
    prob: torch.Tensor,
    fov: Optional[torch.Tensor],
    save_path: str,
    title_prefix: str = "ORIGA",
    threshold: Optional[float] = 0.5,
):
    """Save 5-panel ORIGA/REFUGE visualization figure:
    1) Structure S
    2) Texture |T|
    3) Extracted structure (Green channel)
    4) Disc mask
    5) Cup mask
    """
    S = structure.detach().cpu()
    T = texture.detach().cpu()
    E = extracted_structure.detach().cpu()
    P_disc = prob[0].detach().cpu()
    P_cup = prob[1].detach().cpu() if prob.shape[0] > 1 else torch.zeros_like(P_disc)

    H, W = P_disc.shape
    if fov is not None:
        F = fov.detach().cpu()
        if F.ndim == 3:
            F = F[0]
    else:
        F = torch.ones_like(P_disc)

    # 1) Structure S
    S_vis = S.clone()
    s_min, s_max = S_vis.min(), S_vis.max()
    S_vis = (S_vis - s_min) / (s_max - s_min + 1e-8)
    S_vis = S_vis * F.unsqueeze(0)
    S_vis = S_vis.permute(1, 2, 0).numpy()

    # 2) Texture |T|
    T_abs = T.abs().mean(0)
    t_min, t_max = T_abs.min(), T_abs.max()
    T_vis = (T_abs - t_min) / (t_max - t_min + 1e-8)
    T_vis = (T_vis * F).numpy()

    # 3) Extracted structure
    E_green = E[1] if E.shape[0] > 1 else E[0]
    e_min, e_max = E_green.min(), E_green.max()
    E_vis = (E_green - e_min) / (e_max - e_min + 1e-8)
    E_vis = (E_vis * F).numpy()

    # 4) Disc & Cup masks
    if threshold is None:
        disc_mask = (P_disc * F).numpy()
        cup_mask = (P_cup * F).numpy()
    else:
        disc_mask = ((P_disc >= threshold).float() * F).numpy()
        cup_mask = ((P_cup >= threshold).float() * F).numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(np.clip(S_vis, 0, 1))
    axes[0].set_title(f"{title_prefix} - Structure")
    axes[0].axis("off")

    axes[1].imshow(T_vis, cmap="gray")
    axes[1].set_title("Texture")
    axes[1].axis("off")

    axes[2].imshow(E_vis, cmap="gray")
    axes[2].set_title("Extracted structure (Green)")
    axes[2].axis("off")

    axes[3].imshow(disc_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Disc mask")
    axes[3].axis("off")

    axes[4].imshow(cup_mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Cup mask")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
