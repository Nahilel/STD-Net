from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    """Total Variation (TV) loss on the structure component S.
    Encourages the structural component to be piecewise smooth.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(structure[:, :, 1:, :] - structure[:, :, :-1, :])
        dw = torch.abs(structure[:, :, :, 1:] - structure[:, :, :, :-1])
        tv = (dh.pow(2).mean() + dw.pow(2).mean()).sqrt()
        return self.loss_weight * tv


@LOSS_REGISTRY.register()
class TextureL1Loss(nn.Module):
    """L1 loss on the texture component T.
    Encourages sparsity in high-frequency texture representations.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        l1 = torch.mean(torch.abs(texture))
        return self.loss_weight * l1


@LOSS_REGISTRY.register()
class MaskedBCELoss(nn.Module):
    """Binary Cross Entropy with Logits restricted to the Field of View (FOV)."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        loss_map = self.bce(pred, target)
        mask = (fov > 0.5).float()
        if mask.shape[1] != pred.shape[1]:
            mask = mask.expand(-1, pred.shape[1], -1, -1)

        loss = (loss_map * mask).sum() / (mask.sum() + 1e-8)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class MultiScaleSegLoss(nn.Module):
    """Deep supervision segmentation loss combining multiple output scales:
        - seg_final (full resolution)
        - seg5, seg6, seg7 (intermediate decoder resolutions)
    Default weights: [1.0, 0.3, 0.1, 0.05]
    """

    def __init__(self, weights: Tuple[float, ...] = (1.0, 0.3, 0.1, 0.05)):
        super().__init__()
        self.weights = list(weights)
        self.sum_w = sum(self.weights)
        self.masked_bce = MaskedBCELoss(loss_weight=1.0)

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gt: torch.Tensor,
        fov: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seg_final, seg5, seg6, seg7 = preds

        # Interpolate ground truth and FOV to intermediate scales
        gt5 = F.interpolate(gt, size=seg5.shape[-2:], mode="nearest")
        gt6 = F.interpolate(gt, size=seg6.shape[-2:], mode="nearest")
        gt7 = F.interpolate(gt, size=seg7.shape[-2:], mode="nearest")

        fov5 = F.interpolate(fov, size=seg5.shape[-2:], mode="nearest")
        fov6 = F.interpolate(fov, size=seg6.shape[-2:], mode="nearest")
        fov7 = F.interpolate(fov, size=seg7.shape[-2:], mode="nearest")

        L_final = self.masked_bce(seg_final, gt, fov)
        L5 = self.masked_bce(seg5, gt5, fov5)
        L6 = self.masked_bce(seg6, gt6, fov6)
        L7 = self.masked_bce(seg7, gt7, fov7)

        Lseg = (
            self.weights[0] * L_final
            + self.weights[1] * L5
            + self.weights[2] * L6
            + self.weights[3] * L7
        ) / self.sum_w

        return Lseg, L_final, L5, L6, L7


@LOSS_REGISTRY.register()
class STDNetLoss(nn.Module):
    """Composite loss function for STD-Net:
        L_total = L_seg + mu * (L_t + lam * L_s)
    """

    def __init__(
        self,
        mu: float = 0.001,
        lam: float = 1.0,
        seg_weights: Tuple[float, ...] = (1.0, 0.3, 0.1, 0.05),
    ):
        super().__init__()
        self.mu = mu
        self.lam = lam
        self.seg_loss = MultiScaleSegLoss(weights=seg_weights)
        self.texture_loss = TextureL1Loss(loss_weight=1.0)
        self.structure_loss = TVLoss(loss_weight=1.0)

    def forward(
        self,
        seg_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gt: torch.Tensor,
        fov: torch.Tensor,
        structure: torch.Tensor,
        texture: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        Lseg, L_final, L5, L6, L7 = self.seg_loss(seg_outputs, gt, fov)
        Lt = self.texture_loss(texture)
        Ls = self.structure_loss(structure)

        Ltotal = Lseg + self.mu * (Lt + self.lam * Ls)

        return {
            "total": Ltotal,
            "segmentation": Lseg,
            "texture": Lt,
            "structure": Ls,
            "L_final": L_final,
            "L5": L5,
            "L6": L6,
            "L7": L7,
        }
