import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from basicsr.data.transforms import RetinalSegmentationTransform
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class REFUGEDataset(Dataset):
    """REFUGE Retinal Dataset for Optic Disc and Cup Segmentation."""

    def __init__(
        self,
        opt: Optional[dict] = None,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        transform: Optional[RetinalSegmentationTransform] = None,
        disc_label_val: int = 1,
        cup_label_val: int = 2,
        image_extensions: Tuple[str, ...] = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"),
        mask_extensions: Tuple[str, ...] = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"),
        output_image_size: Optional[Tuple[int, int]] = (256, 256),
    ):
        super().__init__()
        if opt is not None:
            images_dir = opt.get("images_dir")
            masks_dir = opt.get("masks_dir")
            disc_label_val = opt.get("disc_label_val", 1)
            cup_label_val = opt.get("cup_label_val", 2)
            output_image_size = tuple(opt.get("image_size", (256, 256)))
            is_train = opt.get("phase", "train") == "train"
            mean_vals = opt.get("mean", [0.5, 0.5, 0.5])
            std_vals = opt.get("std", [0.5, 0.5, 0.5])
            use_polar = opt.get("use_polar", False)
            rotation_deg = opt.get("rotation_deg", 5.0)

            mean_t = torch.tensor(mean_vals, dtype=torch.float32)
            std_t = torch.tensor(std_vals, dtype=torch.float32)
            transform = RetinalSegmentationTransform(
                image_size=output_image_size,
                mean=mean_t,
                std=std_t,
                train=is_train,
                use_polar=use_polar,
                rotation_deg=rotation_deg,
            )

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.disc_label_val = disc_label_val
        self.cup_label_val = cup_label_val
        self.image_extensions = image_extensions
        self.mask_extensions = mask_extensions
        self.output_image_size = output_image_size

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")

        self.image_files = sorted(
            [f for f in os.listdir(self.images_dir) if f.lower().endswith(self.image_extensions)]
        )
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _find_mask_path(self, image_name: str) -> Path:
        stem = Path(image_name).stem
        for ext in self.mask_extensions:
            cand = self.masks_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No mask found for {image_name} in {self.masks_dir}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        mask_path = self._find_mask_path(img_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.output_image_size is not None and self.transform is None:
            img = img.resize(self.output_image_size, Image.BILINEAR)
            mask = mask.resize(self.output_image_size, Image.NEAREST)

        img_tensor = TF.to_tensor(img)
        mask_np = np.array(mask, dtype=np.float32)

        if mask_np.max() > 2:
            disc_region = (mask_np > 0) & (mask_np < 200)
            cup_region = mask_np >= 200
            labels = np.zeros_like(mask_np, dtype=np.float32)
            labels[disc_region] = self.disc_label_val
            labels[cup_region] = self.cup_label_val
        else:
            labels = mask_np

        mask_tensor = torch.from_numpy(labels).unsqueeze(0)
        masks = {"mask": mask_tensor}

        if self.transform is not None:
            img_tensor, masks = self.transform(img_tensor, masks)

        mask_tensor = masks["mask"].squeeze(0)
        disc = (mask_tensor == float(self.disc_label_val)).float().unsqueeze(0)
        cup = (mask_tensor == float(self.cup_label_val)).float().unsqueeze(0)
        gt = torch.cat([disc, cup], dim=0)
        fov = torch.ones_like(disc)

        return {
            "image": img_tensor,
            "gt": gt,
            "disc": disc,
            "cup": cup,
            "fov": fov,
            "name": img_name,
            "filename": img_name,
        }

