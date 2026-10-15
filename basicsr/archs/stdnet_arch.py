from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class STDModule(nn.Module):
    """Module Structure-Texture Demixing (STD).
    10 convolutions 3x3 + LeakyReLU to extract a texture component T.
    The structural component S is computed as: S = I - T.
    """

    def __init__(self, img_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        layers = []
        # Layer 1: image -> features
        layers.append(nn.Conv2d(img_channels, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(inplace=True))

        # Layers 2 to 9: intermediate representations
        for _ in range(8):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))

        # Layer 10: features -> texture T
        layers.append(nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        texture = self.net(input_image)
        structure = input_image - texture
        return structure, texture


class AdaptiveNorm(nn.Module):
    """Adaptive Normalization:
        Psi(x) = lambda * x + mu * BN(x)
    where lambda and mu are learnable scalars, initialized to 1.0 and 0.0 respectively.
    Reference: https://arxiv.org/pdf/1709.00643
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels, affine=True)
        self.lambda_s = nn.Parameter(torch.tensor(1.0))
        self.mu_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambda_s * x + self.mu_s * self.bn(x)


class TextureBlock(nn.Module):
    """1x1 Texture refinement block:
        Conv(1x1) -> AdaptiveNorm -> LeakyReLU -> Conv(1x1)
    Extracts additional structural representations from the texture component.
    """

    def __init__(self, img_channels: int = 3, hidden_channels: int = 15):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, hidden_channels, kernel_size=1)
        self.adapt = AdaptiveNorm(hidden_channels)
        self.act = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(hidden_channels, img_channels, kernel_size=1)

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        x = self.conv1(texture)
        x = self.adapt(x)
        x = self.act(x)
        extracted_structure = self.conv2(x)
        return extracted_structure


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Basic convolutional building block:
        Conv3x3 -> ReLU -> Conv3x3 -> ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class MNet(nn.Module):
    """Multi-level Encoder-Decoder (M-Net) with deep supervision
    and texture fusion.
    """

    def __init__(
        self,
        img_size: int = 512,
        img_channels: int = 3,
        texture_channels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # -------------------- ENCODER --------------------
        self.enc1 = double_conv(img_channels, 32)
        self.down1 = nn.MaxPool2d(2)

        self.resize2 = double_conv(img_channels, 64)
        self.enc2 = double_conv(96, 64)
        self.down2 = nn.MaxPool2d(2)

        self.resize3 = double_conv(img_channels, 128)
        self.enc3 = double_conv(192, 128)
        self.down3 = nn.MaxPool2d(2)

        self.resize4 = double_conv(img_channels, 256)
        self.enc4 = double_conv(384, 256)
        self.down4 = nn.MaxPool2d(2)

        # -------------------- BOTTLENECK --------------------
        self.bottleneck = double_conv(256, 512)

        # -------------------- DECODER --------------------
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = double_conv(512, 256)
        self.out5 = nn.ConvTranspose2d(256, num_classes, kernel_size=2, stride=2)

        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec6 = double_conv(256, 128)
        self.out6 = nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec7 = double_conv(128, 64)
        self.out7 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec8 = double_conv(64, 32)

        # Texture fusion at level 8
        self.fuse_texture = nn.Conv2d(32 + texture_channels, num_classes, kernel_size=3, padding=1)

        # Final multi-scale fusion
        self.final_fusion = nn.Conv2d(4 * num_classes, num_classes, kernel_size=1)

    def _resize_structure(self, structure: torch.Tensor, scale: float) -> torch.Tensor:
        return F.interpolate(
            structure,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )

    def _upsample_and_concat(
        self,
        skip: torch.Tensor,
        x: torch.Tensor,
        up_module: nn.Module,
    ) -> torch.Tensor:
        up = up_module(x)
        return torch.cat([skip, up], dim=1)

    def forward(
        self,
        structure: torch.Tensor,
        extracted_texture: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # -------------------- ENCODER --------------------
        enc1 = self.enc1(structure)
        down1 = self.down1(enc1)

        # Level 2 multi-scale input
        structure_lvl2 = self._resize_structure(structure, 0.5)
        prep2 = self.resize2(structure_lvl2)
        enc2 = self.enc2(torch.cat([prep2, down1], dim=1))
        down2 = self.down2(enc2)

        # Level 3 multi-scale input
        structure_lvl3 = self._resize_structure(structure_lvl2, 0.5)
        prep3 = self.resize3(structure_lvl3)
        enc3 = self.enc3(torch.cat([prep3, down2], dim=1))
        down3 = self.down3(enc3)

        # Level 4 multi-scale input
        structure_lvl4 = self._resize_structure(structure_lvl3, 0.5)
        prep4 = self.resize4(structure_lvl4)
        enc4 = self.enc4(torch.cat([prep4, down3], dim=1))
        down4 = self.down4(enc4)

        # -------------------- BOTTLENECK --------------------
        bottleneck = self.bottleneck(down4)

        # -------------------- DECODER --------------------
        dec5_input = self._upsample_and_concat(enc4, bottleneck, self.up5)
        dec5 = self.dec5(dec5_input)
        out5 = self.out5(dec5)

        dec6_input = self._upsample_and_concat(enc3, dec5, self.up6)
        dec6 = self.dec6(dec6_input)
        out6 = self.out6(dec6)

        dec7_input = self._upsample_and_concat(enc2, dec6, self.up7)
        dec7 = self.dec7(dec7_input)
        out7 = self.out7(dec7)

        dec8_input = self._upsample_and_concat(enc1, dec7, self.up8)
        dec8 = self.dec8(dec8_input)

        # -------------------- FUSION TEXTURE --------------------
        if extracted_texture.shape[-2:] != structure.shape[-2:]:
            extracted_texture = F.interpolate(
                extracted_texture,
                size=structure.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        texture_fused = self.fuse_texture(
            torch.cat([extracted_texture, dec8], dim=1)
        )

        # -------------------- FINAL FUSION --------------------
        target_size = structure.shape[-2:]
        out5 = F.interpolate(out5, size=target_size, mode="bilinear", align_corners=False)
        out6 = F.interpolate(out6, size=target_size, mode="bilinear", align_corners=False)
        out7 = F.interpolate(out7, size=target_size, mode="bilinear", align_corners=False)
        texture_fused = F.interpolate(texture_fused, size=target_size, mode="bilinear", align_corners=False)

        final = self.final_fusion(torch.cat([out5, out6, out7, texture_fused], dim=1))
        return final, out5, out6, out7


@ARCH_REGISTRY.register()
class STDNetFullModel(nn.Module):
    """Complete STD-Net architecture:
    - STDModule: Decomposition of input image I -> (S, T)
    - TextureBlock: Additional structural features extracted from T
    - MNet: Guided segmentation using structure S and refined texture features

    Forward call returns:
        seg_final, seg5, seg6, seg7, texture, extracted_structure, structure
    """

    def __init__(
        self,
        img_channels: int = 3,
        texture_channels: int = 3,
        hidden_dim: int = 64,
        img_size: int = 512,
        num_classes: int = 1,
    ):
        super().__init__()
        self.std_module = STDModule(img_channels=img_channels, hidden_dim=hidden_dim)
        self.texture_block = TextureBlock(img_channels=texture_channels)
        self.mnet = MNet(
            img_size=img_size,
            img_channels=img_channels,
            texture_channels=texture_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        structure, texture = self.std_module(image)
        extracted_structure = self.texture_block(texture)
        seg_final, seg5, seg6, seg7 = self.mnet(structure, extracted_structure)
        return seg_final, seg5, seg6, seg7, texture, extracted_structure, structure
