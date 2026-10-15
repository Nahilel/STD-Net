import logging
import os
from os import path as osp
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import dict2str, get_root_logger, get_time_str, make_exp_dirs, parse_options, set_random_seed


def main():
    root_path = osp.abspath(osp.dirname(__file__))
    opt = parse_options(root_path, is_train=False)

    make_exp_dirs(opt)
    set_random_seed(opt.get("manual_seed", 42))

    log_file = osp.join(opt["path"]["log"], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # Build Test Dataset and Dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt["datasets"].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt.get("num_gpu", 1),
            dist=False,
            sampler=None,
            seed=opt.get("manual_seed", 42),
        )
        logger.info(f"Number of test images in [{dataset_opt['name']}]: {len(test_set)}")
        test_loaders.append(test_loader)

    # Build Model
    model = build_model(opt)

    for test_loader in test_loaders:
        dataset_name = test_loader.dataset.__class__.__name__
        logger.info(f"Testing on dataset: {dataset_name}")
        test_metrics = model.nondist_validation(
            test_loader,
            current_epoch=0,
            tb_logger=None,
            save_img=opt.get("val", {}).get("save_img", True),
        )

        log_str = f"--- Test Results for [{opt['name']}] ---\n"
        for k, v in test_metrics.items():
            log_str += f"  {k:15s}: {v:.4f}\n"
        logger.info(log_str)

        # Write results to file
        result_txt = osp.join(opt["path"]["results_root"], "test_metrics.txt")
        with open(result_txt, "w") as f:
            f.write(log_str)
        logger.info(f"Results written to: {result_txt}")


if __name__ == "__main__":
    main()

