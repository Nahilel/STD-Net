import argparse
import datetime
import logging
import math
import os
import time
from os import path as osp
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import (
    MessageLogger,
    check_resume,
    copy_opt_file,
    dict2str,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    parse_options,
    set_random_seed,
)


def init_train_loggers(opt: dict):
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # Initialize TensorBoard logger
    tb_logger = None
    if opt.get("logger", {}).get("use_tb_logger", False):
        tb_logger = init_tb_logger(log_dir=osp.join(opt["root_path"], "tb_logger", opt["name"]))

    return logger, tb_logger


def create_train_val_dataloaders(opt: dict, logger):
    train_loader = None
    val_loader = None

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt.get("num_gpu", 1),
                dist=False,
                sampler=None,
                seed=opt.get("manual_seed", 42),
            )
            num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / dataset_opt["batch_size"])
            total_epochs = int(opt["train"]["total_epochs"])
            total_iters = total_epochs * num_iter_per_epoch
            logger.info(
                f"Training statistics: len(train_set)={len(train_set)}, "
                f"batch_size={dataset_opt['batch_size']}, iters/epoch={num_iter_per_epoch}, "
                f"total_epochs={total_epochs}, total_iters={total_iters}"
            )
        elif phase in ["val", "test"]:
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt.get("num_gpu", 1),
                dist=False,
                sampler=None,
                seed=opt.get("manual_seed", 42),
            )
            logger.info(f"Validation statistics: len(val_set)={len(val_set)}")

    return train_loader, val_loader, total_epochs


def main():
    # 1. Parse YAML options
    root_path = osp.abspath(osp.dirname(__file__))
    opt = parse_options(root_path, is_train=True)

    # 2. Make experiment directories
    make_exp_dirs(opt)
    copy_opt_file(opt["opt_file"], opt["path"]["experiments_root"])

    # 3. Seed and Loggers
    set_random_seed(opt.get("manual_seed", 42))
    logger, tb_logger = init_train_loggers(opt)

    # 4. Create Dataloaders
    train_loader, val_loader, total_epochs = create_train_val_dataloaders(opt, logger)

    # 5. Build Model
    model = build_model(opt)

    msg_logger = MessageLogger(opt, current_epoch=1, total_epochs=total_epochs, tb_logger=tb_logger)

    current_iter = 0
    best_val_metric = float("inf")
    best_epoch = 0

    val_freq = opt.get("train", {}).get("val_freq", 1)
    save_checkpoint_freq = opt.get("logger", {}).get("save_checkpoint_freq", 10)
    save_img = opt.get("val", {}).get("save_img", True)

    logger.info(f"--- Start Training {opt['name']} for {total_epochs} epochs ---")

    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        msg_logger.current_epoch = epoch

        # --- Training Loop ---
        for batch_idx, train_data in enumerate(train_loader):
            current_iter += 1
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

        # Log epoch train stats
        train_logs = model.get_current_log()
        train_logs["epoch"] = epoch
        train_logs["lr"] = model.get_current_learning_rate()
        train_logs["time"] = time.time() - epoch_start_time
        msg_logger(train_logs)

        # --- Validation Loop ---
        if val_loader is not None and (epoch % val_freq == 0 or epoch == total_epochs):
            val_metrics = model.nondist_validation(val_loader, current_epoch=epoch, tb_logger=tb_logger, save_img=save_img)

            # Schedulers (e.g. ReduceLROnPlateau monitoring val_loss or OE)
            val_loss = val_metrics.get("val_loss", train_logs.get("total", 0.0))
            for scheduler in model.schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

            val_log_msg = f"[Validation] Epoch: [{epoch:3d}/{total_epochs:3d}] "
            for k, v in val_metrics.items():
                val_log_msg += f"{k}: {v:.4f} "
                if tb_logger:
                    tb_logger.add_scalar(f"val/{k}", v, epoch)
            logger.info(val_log_msg)

            # Check best model
            # For ORIGA/REFUGE, track OE_total (lower is better); for DRIVE track val_loss (or -IOU)
            criterion_metric = val_metrics.get("OE_total", val_metrics.get("val_loss", float("inf")))
            if criterion_metric < best_val_metric:
                best_val_metric = criterion_metric
                best_epoch = epoch
                model.save_best_network(model.net_g, "net_g")
                logger.info(f"===> New best model at epoch {epoch}! Monitored metric: {best_val_metric:.4f}")

        # Periodic checkpoint save
        if epoch % save_checkpoint_freq == 0:
            model.save_network(model.net_g, "net_g", epoch)

    # Save latest model at end of training
    model.save_network(model.net_g, "net_g_latest", total_epochs)
    logger.info(f"--- Training Finished. Best Epoch: {best_epoch} with metric: {best_val_metric:.4f} ---")


if __name__ == "__main__":
    main()

