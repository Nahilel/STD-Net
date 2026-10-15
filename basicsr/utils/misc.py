import os
import random
import time
from os import path as osp
import numpy as np
import torch


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_and_rename(path: str):
    """Make directory and rename existing one if it exists."""
    if osp.exists(path):
        new_name = path + "_archived_" + get_time_str()
        print(f"Path already exists. Rename it to {new_name}", flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def make_exp_dirs(opt: dict):
    """Make experiment directories according to options."""
    path_opt = opt.get("path", {})
    if opt.get("is_train", True):
        experiments_root = path_opt.get("experiments_root")
        if experiments_root:
            os.makedirs(experiments_root, exist_ok=True)
        models_dir = path_opt.get("models")
        if models_dir:
            os.makedirs(models_dir, exist_ok=True)
        log_dir = path_opt.get("log")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        visualization_dir = path_opt.get("visualization")
        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)
    else:
        results_root = path_opt.get("results_root")
        if results_root:
            os.makedirs(results_root, exist_ok=True)
        visualization_dir = path_opt.get("visualization")
        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)


def scandir(dir_path: str, suffix=None, recursive: bool = False, full_path: bool = False):
    """Scan a directory to find the interested files."""
    if isinstance(suffix, str):
        suffix = (suffix,)

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if suffix is None or entry.name.endswith(suffix):
                    yield entry.path if full_path else osp.relpath(entry.path, root)
            elif recursive and entry.is_dir():
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def check_resume(opt: dict, resume_iter: int = None):
    """Check resume options and set resume flags."""
    resume_state = opt.get("path", {}).get("resume_state")
    if resume_state is not None:
        print(f"Resuming training from: {resume_state}", flush=True)

