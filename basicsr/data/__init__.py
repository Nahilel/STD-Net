import copy
import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.data.drive_dataset import DRIVEDataset
from basicsr.data.origa_dataset import ORIGADataset
from basicsr.data.refuge_dataset import REFUGEDataset
from basicsr.data.transforms import RetinalSegmentationTransform, apply_clahe, compute_mean_std, polar_transform_tensor
from basicsr.utils.registry import DATASET_REGISTRY

__all__ = [
    "DRIVEDataset",
    "ORIGADataset",
    "REFUGEDataset",
    "RetinalSegmentationTransform",
    "apply_clahe",
    "polar_transform_tensor",
    "compute_mean_std",
    "build_dataset",
    "build_dataloader",
]


def build_dataset(dataset_opt: dict):
    dataset_opt = copy.deepcopy(dataset_opt)
    dataset_type = dataset_opt.get("type")
    dataset = DATASET_REGISTRY.get(dataset_type)(dataset_opt)
    return dataset


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, dataset_opt: dict, num_gpu: int = 1, dist: bool = False, sampler=None, seed: int = None):
    phase = dataset_opt.get("phase", "train")
    is_train = phase == "train"

    batch_size = dataset_opt.get("batch_size", 2 if is_train else 1)
    num_workers = dataset_opt.get("num_workers", 0)
    shuffle = is_train and sampler is None

    if seed is not None:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=0, seed=seed)
    else:
        init_fn = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=dataset_opt.get("pin_memory", False),
        worker_init_fn=init_fn,
    )
    return dataloader
