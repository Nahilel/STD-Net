import copy
from basicsr.models.base_model import BaseModel
from basicsr.models.stdnet_model import STDNetModel
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = [
    "BaseModel",
    "STDNetModel",
    "build_model",
]


def build_model(opt: dict):
    opt = copy.deepcopy(opt)
    model_type = opt.get("model_type", "STDNetModel")
    model = MODEL_REGISTRY.get(model_type)(opt)
    return model

