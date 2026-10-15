from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_drive_metrics(
    all_probs: List[np.ndarray],
    all_targets: List[np.ndarray],
    all_fovs: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes segmentation metrics for the DRIVE dataset strictly within the FOV:
    - Acc: Overall accuracy
    - AUC: Area Under ROC Curve
    - Sen: Sensitivity (Recall)
    - Spe: Specificity
    - IOU: Intersection over Union
    """
    probs_flat = np.concatenate(all_probs, axis=0)
    targets_flat = np.concatenate(all_targets, axis=0)
    fovs_flat = np.concatenate(all_fovs, axis=0)

    # Valid mask within FOV
    valid = fovs_flat > 0.5
    probs_valid = probs_flat[valid]
    targets_valid = targets_flat[valid]

    preds_valid = (probs_valid >= threshold).astype(np.float32)

    TP = int(np.sum((preds_valid == 1) & (targets_valid == 1)))
    FP = int(np.sum((preds_valid == 1) & (targets_valid == 0)))
    FN = int(np.sum((preds_valid == 0) & (targets_valid == 1)))
    TN = int(np.sum((preds_valid == 0) & (targets_valid == 0)))

    denominator = TP + TN + FP + FN + 1e-8
    Acc = (TP + TN) / denominator
    Sen = TP / (TP + FN + 1e-8)
    Spe = TN / (TN + FP + 1e-8)
    IOU = TP / (TP + FP + FN + 1e-8)

    unique_classes = np.unique(targets_valid)
    if unique_classes.size < 2:
        AUC = float("nan")
    else:
        AUC = float(roc_auc_score(targets_valid, probs_valid))

    return {
        "Acc": float(Acc),
        "AUC": float(AUC),
        "Sen": float(Sen),
        "Spe": float(Spe),
        "IOU": float(IOU),
    }


def overlapping_error(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-8) -> float:
    """Overlapping Error (OE) = 1 - IoU for a single mask.

    Args:
        pred_mask: Tensor [H, W] or [1, H, W] (probabilities or binary)
        gt_mask: Tensor [H, W] or [1, H, W] (binary ground truth)
    """
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]

    pred_bin = (pred_mask >= 0.5).float()
    gt_bin = (gt_mask >= 0.5).float()

    inter = (pred_bin * gt_bin).sum().item()
    union = ((pred_bin + gt_bin) > 0).float().sum().item()

    if union == 0:
        return 0.0

    iou = inter / (union + eps)
    return float(1.0 - iou)


@METRIC_REGISTRY.register()
def calculate_overlapping_error(
    all_disc_preds: List[torch.Tensor],
    all_disc_gts: List[torch.Tensor],
    all_cup_preds: List[torch.Tensor],
    all_cup_gts: List[torch.Tensor],
) -> Dict[str, float]:
    """Computes mean Overlapping Error (OE) across disc, cup, and combined."""
    disc_oes = [overlapping_error(p, g) for p, g in zip(all_disc_preds, all_disc_gts)]
    cup_oes = [overlapping_error(p, g) for p, g in zip(all_cup_preds, all_cup_gts)]

    OE_disc = float(np.mean(disc_oes)) if disc_oes else 0.0
    OE_cup = float(np.mean(cup_oes)) if cup_oes else 0.0
    OE_total = OE_disc + OE_cup

    return {
        "OE_disc": OE_disc,
        "OE_cup": OE_cup,
        "OE_total": OE_total,
    }

