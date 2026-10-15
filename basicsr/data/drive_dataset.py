import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from basicsr.data.transforms import RetinalSegmentationTransform, apply_clahe
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DRIVEDataset(Dataset):
    """DRIVE Retinal Vessel Segmentation Dataset.

    Supports both standalone usage and BasicSR YAML configuration dicts.
    """

    def __init__(
        self,
        opt: Optional[dict] = None,
        images_dir: Optional[str] = None,
        fov_dir: Optional[str] = None,
        gt_dir: Optional[str] = None,
        second_gt_dir: Optional[str] = None,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        apply_clahe_flag: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[RetinalSegmentationTransform] = None,
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if opt is not None:
            self.images_dir = Path(opt.get("images_dir", ""))
            self.fov_dir = Path(opt.get("fov_dir", ""))
            self.gt_dir = Path(opt.get("gt_dir", "")) if opt.get("gt_dir") else None
            self.second_gt_dir = Path(opt.get("second_gt_dir", "")) if opt.get("second_gt_dir") else None
            self.split = opt.get("split", "train")
            self.image_size = tuple(opt.get("image_size", (512, 512)))
            self.apply_clahe_flag = opt.get("apply_clahe", True)
            is_train_phase = opt.get("phase", "train") == "train"
            mean_vals = opt.get("mean", [0.485, 0.456, 0.406])
            std_vals = opt.get("std", [0.229, 0.224, 0.225])
            rotation_deg = opt.get("rotation_deg", 15.0)

            mean_t = torch.tensor(mean_vals, dtype=torch.float32)
            std_t = torch.tensor(std_vals, dtype=torch.float32)
            self.transform = RetinalSegmentationTransform(
                image_size=self.image_size,
                mean=mean_t,
                std=std_t,
                train=is_train_phase,
                use_polar=False,
                rotation_deg=rotation_deg,
            )
        else:
            self.images_dir = Path(images_dir)
            self.fov_dir = Path(fov_dir)
            self.gt_dir = Path(gt_dir) if gt_dir is not None else None
            self.second_gt_dir = Path(second_gt_dir) if second_gt_dir is not None else None
            self.split = split
            self.image_size = tuple(image_size)
            self.apply_clahe_flag = apply_clahe_flag
            self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"DRIVE images directory not found: {self.images_dir}")

        all_filenames = sorted(
            [f for f in os.listdir(self.images_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
        )

        if indices is not None:
            self.image_filenames = [all_filenames[i] for i in indices]
        else:
            self.image_filenames = all_filenames

    def __len__(self) -> int:
        return len(self.image_filenames)

    def _get_mask_names(self, img_name: str) -> Tuple[str, str, Optional[str]]:
        base = img_name.split("_")[0]
        if self.split == "train":
            fov_name = f"{base}_training_mask.gif"
        else:
            fov_name = f"{base}_test_mask.gif"

        gt1_name = f"{base}_manual1.gif"
        gt2_name = f"{base}_manual2.gif" if self.second_gt_dir is not None else None
        return fov_name, gt1_name, gt2_name

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.image_filenames[idx]
        fov_name, gt1_name, gt2_name = self._get_mask_names(img_name)

        # Image
        img_path = self.images_dir / img_name
        img = Image.open(img_path).convert("RGB")
        if self.apply_clahe_flag:
            img = apply_clahe(img)
        img_tensor = TF.to_tensor(img)

        # FOV
        fov_path = self.fov_dir / fov_name
        if not fov_path.exists():
            # Check with .png / .tif / other extensions if gif not found
            stem = fov_path.stem
            for ext in [".gif", ".png", ".tif", ".jpg"]:
                cand = self.fov_dir / f"{stem}{ext}"
                if cand.exists():
                    fov_path = cand
                    break
        if fov_path.exists():
            fov = Image.open(fov_path)
            fov_tensor = TF.to_tensor(fov)
        else:
            fov_tensor = torch.ones((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)

        # Ground truth 1
        gt1_tensor = None
        if self.gt_dir is not None:
            gt1_path = self.gt_dir / gt1_name
            if not gt1_path.exists():
                stem = gt1_path.stem
                for ext in [".gif", ".png", ".tif", ".jpg"]:
                    cand = self.gt_dir / f"{stem}{ext}"
                    if cand.exists():
                        gt1_path = cand
                        break
            if gt1_path.exists():
                gt1 = Image.open(gt1_path)
                gt1_tensor = TF.to_tensor(gt1)

        # Ground truth 2 (optional)
        gt2_tensor = None
        if self.second_gt_dir is not None and gt2_name is not None:
            gt2_path = self.second_gt_dir / gt2_name
            if gt2_path.exists():
                gt2 = Image.open(gt2_path)
                gt2_tensor = TF.to_tensor(gt2)

        masks = {"fov": fov_tensor}
        if gt1_tensor is not None:
            masks["gt"] = gt1_tensor
        if gt2_tensor is not None:
            masks["gt2"] = gt2_tensor

        if self.transform is not None:
            img_tensor, masks = self.transform(img_tensor, masks)

        fov_bin = (masks["fov"] > 0.5).float()
        sample = {
            "image": img_tensor,
            "fov": fov_bin,
            "filename": img_name,
        }

        if "gt" in masks:
            sample["gt"] = (masks["gt"] > 0.5).float()
        if "gt2" in masks:
            sample["gt2"] = (masks["gt2"] > 0.5).float()

        return sample

