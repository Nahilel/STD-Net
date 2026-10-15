import copy
from basicsr.losses.stdnet_loss import MaskedBCELoss, MultiScaleSegLoss, STDNetLoss, TextureL1Loss, TVLoss
from basicsr.utils.registry import LOSS_REGISTRY

__all__ = [
    "TVLoss",
    "TextureL1Loss",
    "MaskedBCELoss",
    "MultiScaleSegLoss",
    "STDNetLoss",
    "build_loss",
]


def build_loss(opt: dict):
    opt = copy.deepcopy(opt)
    loss_type = opt.pop("type")
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    return loss

