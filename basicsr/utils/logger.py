import datetime
import logging
import time
from os import path as osp
from basicsr.utils.misc import get_time_str

initialized_logger = {}


def get_root_logger(logger_name: str = "basicsr", log_level=logging.INFO, log_file: str = None) -> logging.Logger:
    """Get the root logger with stream handler and optional file handler."""
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger

    format_str = "%(asctime)s %(levelname)s: %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)

    logger.setLevel(log_level)
    initialized_logger[logger_name] = True
    return logger


class MessageLogger:
    """Message logger for training loop."""

    def __init__(self, opt: dict, current_epoch: int = 1, total_epochs: int = 100, tb_logger=None):
        self.exp_name = opt.get("name", "experiment")
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def __call__(self, log_dict: dict):
        epoch = log_dict.pop("epoch", self.current_epoch)
        lr = log_dict.pop("lr", None)

        msg = f"[{self.exp_name}] Epoch: [{epoch:3d}/{self.total_epochs:3d}] "
        if lr is not None:
            if isinstance(lr, (list, tuple)):
                lr_str = ", ".join([f"{x:.3e}" for x in lr])
            else:
                lr_str = f"{lr:.3e}"
            msg += f"lr: {lr_str} "

        for k, v in log_dict.items():
            if isinstance(v, float):
                msg += f"{k}: {v:.4f} "
            else:
                msg += f"{k}: {v} "

        self.logger.info(msg)

        if self.tb_logger:
            for k, v in log_dict.items():
                if isinstance(v, (int, float)):
                    self.tb_logger.add_scalar(k, v, epoch)


def init_tb_logger(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir=log_dir)
        return tb_logger
    except Exception:
        return None
