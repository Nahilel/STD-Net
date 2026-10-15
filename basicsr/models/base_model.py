from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from os import path as osp
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from basicsr.utils.logger import get_root_logger


class BaseModel(ABC):
    """Base class for models in BasicSR."""

    def __init__(self, opt: dict):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() and opt.get("num_gpu", 1) > 0 else "cpu")
        self.is_train = opt.get("is_train", True)
        self.schedulers = []
        self.optimizers = []
        self.logger = get_root_logger()

    @abstractmethod
    def feed_data(self, data: dict):
        pass

    @abstractmethod
    def optimize_parameters(self, current_iter: int):
        pass

    def get_current_visuals(self) -> dict:
        return {}

    def save_network(self, net: nn.Module, net_label: str, current_iter: int, param_key: str = "params"):
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = osp.join(self.opt["path"]["models"], save_filename)

        net = net.module if hasattr(net, "module") else net
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save({param_key: state_dict}, save_path)
        self.logger.info(f"Model saved to: {save_path}")

    def save_best_network(self, net: nn.Module, net_label: str, param_key: str = "params"):
        save_filename = f"{net_label}_best.pth"
        save_path = osp.join(self.opt["path"]["models"], save_filename)

        net = net.module if hasattr(net, "module") else net
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save({param_key: state_dict}, save_path)
        self.logger.info(f"Best model saved to: {save_path}")

    def load_network(self, net: nn.Module, load_path: str, strict: bool = True, param_key: str = "params"):
        net = net.module if hasattr(net, "module") else net
        self.logger.info(f"Loading {net.__class__.__name__} model from {load_path}")
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None and param_key in load_net:
            load_net = load_net[param_key]
        # Remove 'module.' prefix if present
        new_state_dict = OrderedDict()
        for k, v in load_net.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict, strict=strict)

    def setup_schedulers(self):
        train_opt = self.opt["train"]
        scheduler_type = train_opt.get("scheduler", {}).get("type", "ReduceLROnPlateau")

        if scheduler_type == "ReduceLROnPlateau":
            for optimizer in self.optimizers:
                sched_opt = train_opt.get("scheduler", {})
                self.schedulers.append(
                    torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=sched_opt.get("mode", "min"),
                        factor=sched_opt.get("factor", 0.5),
                        patience=sched_opt.get("patience", 10),
                        min_lr=sched_opt.get("min_lr", 1e-6),
                    )
                )
        elif scheduler_type == "MultiStepLR":
            for optimizer in self.optimizers:
                sched_opt = train_opt.get("scheduler", {})
                self.schedulers.append(
                    torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=sched_opt.get("milestones", [50, 100]),
                        gamma=sched_opt.get("gamma", 0.5),
                    )
                )
        elif scheduler_type == "CosineAnnealingLR":
            for optimizer in self.optimizers:
                sched_opt = train_opt.get("scheduler", {})
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=sched_opt.get("T_max", 100),
                        eta_min=sched_opt.get("eta_min", 1e-6),
                    )
                )

    def get_current_learning_rate(self) -> List[float]:
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    def print_network(self, net: nn.Module):
        net = net.module if hasattr(net, "module") else net
        s, n = self._get_bare_model(net)
        net_str = f"Network [{net.__class__.__name__}] with parameters: {n:,d}\n{s}"
        self.logger.info(net_str)

    def _get_bare_model(self, net: nn.Module):
        net = net.module if hasattr(net, "module") else net
        num_params = sum(map(lambda x: x.numel(), net.parameters()))
        return str(net), num_params
