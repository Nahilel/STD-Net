from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the Luminance (L) channel in LAB color space.

    Args:
        pil_img: PIL Image in RGB format.

    Returns:
        PIL Image after CLAHE enhancement.
    """
    img_np = np.array(pil_img)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final)


def polar_transform_tensor(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    is_mask: bool = False,
) -> torch.Tensor:
    """Transforms a tensor [C, H, W] into polar coordinates around the image center.

    Args:
        tensor: torch.Tensor of shape [C, H, W], [1, H, W], or [H, W].
        size: Target spatial size (H_out, W_out).
        is_mask: If True, uses NEAREST interpolation; otherwise BILINEAR.

    Returns:
        Polar transformed tensor of shape [C, H_out, W_out].
    """
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    c, h, w = tensor.shape
    img_np = tensor.permute(1, 2, 0).cpu().numpy()

    center = (w / 2.0, h / 2.0)
    max_radius = min(center[0], center[1])
    interp_flag = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + interp_flag

    polar = cv2.warpPolar(img_np, (w, h), center, max_radius, flags)
    polar = cv2.rotate(polar, cv2.ROTATE_90_CLOCKWISE)
    polar = cv2.resize(polar, (size[1], size[0]), interpolation=interp_flag)

    if polar.ndim == 2:
        polar = np.expand_dims(polar, axis=-1)

    polar_tensor = torch.from_numpy(polar).permute(2, 0, 1)
    return polar_tensor


def compute_mean_std(
    dataset: Dataset,
    image_key: str = "image",
    image_size: Optional[Tuple[int, int]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes channel-wise mean and std across a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_channels = None
    channel_sum = None
    channel_sq_sum = None
    num_images = 0

    for batch in tqdm(loader, desc="Computing Dataset Stats"):
        images = batch[image_key]
        if image_size is not None:
            images = TF.resize(images, image_size, InterpolationMode.BILINEAR)

        b, c, h, w = images.shape
        if n_channels is None:
            n_channels = c
            channel_sum = torch.zeros(c)
            channel_sq_sum = torch.zeros(c)

        images = images.view(b, c, -1)
        channel_sum += images.mean(dim=2).sum(dim=0)
        channel_sq_sum += (images ** 2).mean(dim=2).sum(dim=0)
        num_images += b

    mean = channel_sum / num_images
    std = (channel_sq_sum / num_images - mean ** 2).sqrt()
    return mean, std


class RetinalSegmentationTransform:
    """General transform pipeline for retinal segmentation datasets:
    - Resize image and masks to image_size
    - Random geometric augmentations (horizontal/vertical flips, rotations) if train=True
    - Polar coordinate transformation if use_polar=True
    - Image normalization with channel-wise mean and std
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        train: bool = True,
        use_polar: bool = False,
        rotation_deg: float = 15.0,
    ):
        self.image_size = tuple(image_size)
        self.mean = mean if mean is not None else torch.tensor([0.5, 0.5, 0.5])
        self.std = std if std is not None else torch.tensor([0.5, 0.5, 0.5])
        self.train = train
        self.use_polar = use_polar
        self.rotation_deg = rotation_deg

    def _resize(self, tensor: torch.Tensor, is_mask: bool) -> torch.Tensor:
        interpolation = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        return TF.resize(tensor, self.image_size, interpolation=interpolation)

    def _maybe_augment(
        self, image: torch.Tensor, masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.train:
            return image, masks

        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            masks = {k: TF.hflip(v) for k, v in masks.items()}

        # Random vertical flip
        if torch.rand(1).item() < 0.5:
            image = TF.vflip(image)
            masks = {k: TF.vflip(v) for k, v in masks.items()}

        # Random rotation
        if self.rotation_deg > 0:
            angle = float(torch.empty(1).uniform_(-self.rotation_deg, self.rotation_deg))
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            masks = {k: TF.rotate(v, angle, interpolation=InterpolationMode.NEAREST) for k, v in masks.items()}

        return image, masks

    def _maybe_polar(
        self, image: torch.Tensor, masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_polar:
            return image, masks

        image = polar_transform_tensor(image, size=self.image_size, is_mask=False)
        masks = {k: polar_transform_tensor(v, size=self.image_size, is_mask=True) for k, v in masks.items()}
        return image, masks

    def __call__(
        self, image: torch.Tensor, masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = self._resize(image, is_mask=False)
        masks = {k: self._resize(v, is_mask=True) for k, v in masks.items()}

        image, masks = self._maybe_augment(image, masks)
        image, masks = self._maybe_polar(image, masks)

        mean = self.mean.to(image.device) if isinstance(self.mean, torch.Tensor) else self.mean
        std = self.std.to(image.device) if isinstance(self.std, torch.Tensor) else self.std
        image = TF.normalize(image, mean, std)

        return image, masks

