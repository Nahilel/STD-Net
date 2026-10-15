import argparse
import os
from collections import OrderedDict
from os import path as osp
import yaml


def ordered_yaml():
    """Support OrderedDict in yaml."""
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse_options(root_path: str = None, is_train: bool = True) -> dict:
    """Parse YAML options from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option YAML file.")
    parser.add_argument("--auto_resume", action="store_true", help="Auto resume from the latest checkpoint.")
    args = parser.parse_args()

    opt_file = args.opt
    Loader, _ = ordered_yaml()
    with open(opt_file, mode="r", encoding="utf-8") as f:
        opt = yaml.load(f, Loader=Loader)

    opt["is_train"] = is_train
    opt["auto_resume"] = args.auto_resume
    opt["opt_file"] = opt_file

    if root_path is None:
        root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    opt["root_path"] = root_path

    # Set up paths
    if "path" not in opt:
        opt["path"] = {}

    name = opt.get("name", "experiment")
    if is_train:
        experiments_root = osp.join(root_path, "experiments", name)
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["log"] = experiments_root
        opt["path"]["visualization"] = osp.join(experiments_root, "visualization")
    else:
        results_root = osp.join(root_path, "results", name)
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root
        opt["path"]["visualization"] = osp.join(results_root, "visualization")

    return opt


def parse_options_from_dict(opt: dict, root_path: str = None, is_train: bool = True) -> dict:
    """Parse options when given directly as a dictionary."""
    opt["is_train"] = is_train
    if root_path is None:
        root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    opt["root_path"] = root_path

    if "path" not in opt:
        opt["path"] = {}

    name = opt.get("name", "experiment")
    if is_train:
        experiments_root = osp.join(root_path, "experiments", name)
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["log"] = experiments_root
        opt["path"]["visualization"] = osp.join(experiments_root, "visualization")
    else:
        results_root = osp.join(root_path, "results", name)
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root
        opt["path"]["visualization"] = osp.join(results_root, "visualization")

    return opt


def dict2str(opt: dict, indent_level: int = 1) -> str:
    """dict to string for pretty printing."""
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + f": {v}\n"
    return msg


def copy_opt_file(opt_file: str, experiments_root: str):
    """Copy the option file to experiment folder for archiving."""
    import shutil
    from os import path as osp
    if osp.isfile(opt_file):
        os.makedirs(experiments_root, exist_ok=True)
        shutil.copy(opt_file, osp.join(experiments_root, osp.basename(opt_file)))

