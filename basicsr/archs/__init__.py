import copy
from basicsr.archs.stdnet_arch import AdaptiveNorm, MNet, STDModule, STDNetFullModel, TextureBlock, double_conv
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = [
    "STDModule",
    "AdaptiveNorm",
    "TextureBlock",
    "double_conv",
    "MNet",
    "STDNetFullModel",
    "build_network",
]


def build_network(opt: dict):
    opt = copy.deepcopy(opt)
    network_type = opt.pop("type")
    net = ARCH_REGISTRY.get(network_type)(**opt)
    return net
