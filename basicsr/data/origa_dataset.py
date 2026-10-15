import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from basicsr.data.transforms import RetinalSegmentationTransform
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ORIGADataset(Dataset):
    """ORIGA Retinal Dataset for Optic Disc and Cup Segmentation."""

    def __init__(
        self,
        opt: Optional[dict] = None,
        csv_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[RetinalSegmentationTransform] = None,
        disc_label_val: int = 1,
        cup_label_val: int = 2,
        image_extensions: Tuple[str, ...] = ("jpg", "png", "bmp", "tif"),
        mask_extensions: Tuple[str, ...] = ("png", "bmp", "jpg", "tif"),
        seed: int = 42,
        output_image_size: Optional[Tuple[int, int]] = (256, 256),
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if opt is not None:
            csv_path = opt.get("csv_path")
            images_dir = opt.get("images_dir")
            masks_dir = opt.get("masks_dir")
            split = opt.get("split", "train")
            disc_label_val = opt.get("disc_label_val", 1)
            cup_label_val = opt.get("cup_label_val", 2)
            seed = opt.get("seed", 42)
            output_image_size = tuple(opt.get("image_size", (256, 256)))
            is_train = opt.get("phase", "train") == "train"
            mean_vals = opt.get("mean", [0.5, 0.5, 0.5])
            std_vals = opt.get("std", [0.5, 0.5, 0.5])
            use_polar = opt.get("use_polar", True)
            rotation_deg = opt.get("rotation_deg", 15.0)

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
        self.disc_label_val = float(disc_label_val)
        self.cup_label_val = float(cup_label_val)
        self.image_extensions = image_extensions
        self.mask_extensions = mask_extensions
        self.output_image_size = output_image_size

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"ORIGA CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        train_df, test_df = self._build_official_split(df, seed=seed)

        if split == "train":
            base_df = train_df.reset_index(drop=True)
        elif split == "test":
            base_df = test_df.reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'test'")

        if indices is not None:
            self.df = base_df.iloc[indices].reset_index(drop=True)
        else:
            self.df = base_df

        self.filenames: List[str] = self.df["Filename"].tolist()
        self.glaucoma_labels: List[int] = self.df["Glaucoma"].astype(int).tolist()
        self.exp_cdr: np.ndarray = self.df["ExpCDR"].values if "ExpCDR" in self.df else np.zeros(len(self.df))

    @staticmethod
    def _build_official_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        uniques = df.get("Set", pd.Series([])).unique()
        train_df, test_df = None, None

        for u in uniques:
            cand = df[df["Set"] == u]
            if len(cand) == 325 and cand["Glaucoma"].sum() == 73:
                train_df = cand
            if len(cand) == 325 and cand["Glaucoma"].sum() == 95:
                test_df = cand

        if train_df is not None and test_df is not None:
            return train_df, test_df

        rng = np.random.RandomState(seed)
        pos = df[df["Glaucoma"] == 1].copy()
        neg = df[df["Glaucoma"] == 0].copy()

        pos = pos.sample(frac=1.0, random_state=seed)
        neg = neg.sample(frac=1.0, random_state=seed + 1)

        train_pos = pos.iloc[:73]
        test_pos = pos.iloc[73:]
        train_neg = neg.iloc[:252]
        test_neg = neg.iloc[252:]

        train_df = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = pd.concat([test_pos, test_neg]).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
        return train_df, test_df

    def _find_with_extensions(self, directory: Path, filename: str, extensions: Tuple[str, ...]) -> Path:
        p = directory / filename
        if p.exists():
            return p
        stem, _ = os.path.splitext(filename)
        for ext in extensions:
            cand = directory / f"{stem}.{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"{filename} not found in {directory}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        fname = row["Filename"]

        img_path = self._find_with_extensions(self.images_dir, fname, self.image_extensions)
        img = Image.open(img_path).convert("RGB")

        mask_path = self._find_with_extensions(self.masks_dir, fname, self.mask_extensions)
        mask = Image.open(mask_path).convert("L")

        if self.output_image_size is not None and self.transform is None:
            img = img.resize(self.output_image_size, Image.BILINEAR)
            mask = mask.resize(self.output_image_size, Image.NEAREST)

        img_tensor = TF.to_tensor(img)
        mask_np = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        masks = {"mask": mask_tensor}
        if self.transform is not None:
            img_tensor, masks = self.transform(img_tensor, masks)

        mask_tensor = masks["mask"].squeeze(0)
        disc = (mask_tensor == self.disc_label_val).float().unsqueeze(0)
        cup = (mask_tensor == self.cup_label_val).float().unsqueeze(0)
        gt = torch.cat([disc, cup], dim=0)
        fov = torch.ones_like(disc)

        return {
            "image": img_tensor,
            "gt": gt,
            "disc": disc,
            "cup": cup,
            "fov": fov,
            "filename": fname,
            "glaucoma": int(row.get("Glaucoma", 0)),
            "cdr": float(row.get("ExpCDR", 0.0)),
        }

