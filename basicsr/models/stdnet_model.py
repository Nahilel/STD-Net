from collections import OrderedDict
from os import path as osp
from typing import Dict, Optional
import numpy as np
import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics.seg_metrics import calculate_drive_metrics, calculate_overlapping_error
from basicsr.models.base_model import BaseModel
from basicsr.utils.img_util import save_drive_visual_sample, save_origa_visual_sample
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class STDNetModel(BaseModel):
    """STD-Net Model wrapper for BasicSR."""

    def __init__(self, opt: dict):
        super().__init__(opt)

        # Build Generator/Segmentation Network
        self.net_g = build_network(opt["network_g"]).to(self.device)
        self.print_network(self.net_g)

        # Load pretrained weights if provided
        load_path = self.opt.get("path", {}).get("pretrain_network_g")
        if load_path is not None and osp.exists(load_path):
            self.load_network(self.net_g, load_path, strict=self.opt.get("path", {}).get("strict_load_g", True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt["train"]

        # Setup Loss
        loss_opt = train_opt.get("loss", {"type": "STDNetLoss", "mu": 0.001, "lam": 1.0})
        self.cri_stdnet = build_loss(loss_opt).to(self.device)

        # Setup Optimizer
        optim_opt = train_opt.get("optim_g", {"type": "Adam", "lr": 1e-3})
        optim_type = optim_opt.get("type", "Adam")
        lr = float(optim_opt.get("lr", 1e-3))
        weight_decay = float(optim_opt.get("weight_decay", 0.0))

        if optim_type == "Adam":
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_type == "SGD":
            momentum = float(optim_opt.get("momentum", 0.9))
            self.optimizer_g = torch.optim.SGD(self.net_g.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {optim_type} is not supported.")

        self.optimizers.append(self.optimizer_g)
        self.setup_schedulers()
        self.log_dict = OrderedDict()

    def feed_data(self, data: dict):
        self.image = data["image"].to(self.device)
        self.gt = data.get("gt")
        if self.gt is not None:
            self.gt = self.gt.to(self.device)
        self.fov = data.get("fov")
        if self.fov is not None:
            self.fov = self.fov.to(self.device)
        else:
            self.fov = torch.ones((self.image.shape[0], 1, self.image.shape[2], self.image.shape[3]), device=self.device)

        fname = data.get("filename", ["sample"])
        self.filename = fname[0] if isinstance(fname, list) else fname

    def optimize_parameters(self, current_iter: int = 0):
        self.net_g.train()
        self.optimizer_g.zero_grad()

        self.seg_final, self.seg5, self.seg6, self.seg7, self.texture, self.extracted_structure, self.structure = (
            self.net_g(self.image)
        )

        loss_dict = self.cri_stdnet(
            seg_outputs=(self.seg_final, self.seg5, self.seg6, self.seg7),
            gt=self.gt,
            fov=self.fov,
            structure=self.structure,
            texture=self.texture,
        )

        loss = loss_dict["total"]
        loss.backward()
        self.optimizer_g.step()

        self.log_dict = OrderedDict()
        for k, v in loss_dict.items():
            self.log_dict[k] = v.item()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.seg_final, self.seg5, self.seg6, self.seg7, self.texture, self.extracted_structure, self.structure = (
                self.net_g(self.image)
            )
            self.prob = torch.sigmoid(self.seg_final)

    def dist_validation(self, dataloader, current_epoch: int, tb_logger=None, save_img: bool = True):
        return self.nondist_validation(dataloader, current_epoch, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_epoch: int, tb_logger=None, save_img: bool = True) -> Dict[str, float]:
        self.net_g.eval()
        val_losses = []
        val_segs = []
        val_Ts = []
        val_Ss = []

        all_probs = []
        all_targets = []
        all_fovs = []

        all_disc_preds = []
        all_disc_gts = []
        all_cup_preds = []
        all_cup_gts = []

        num_classes = self.opt.get("network_g", {}).get("num_classes", 1)
        dataset_name = self.opt.get("name", "eval").upper()

        visual_saved = False
        vis_dir = self.opt.get("path", {}).get("visualization", "results/visualization")

        for idx, val_data in enumerate(tqdm(dataloader, desc=f"Validation [{dataset_name}]")):
            self.feed_data(val_data)
            self.test()

            if self.gt is not None:
                with torch.no_grad():
                    losses = self.cri_stdnet(
                        seg_outputs=(self.seg_final, self.seg5, self.seg6, self.seg7),
                        gt=self.gt,
                        fov=self.fov,
                        structure=self.structure,
                        texture=self.texture,
                    )
                val_losses.append(losses["total"].item())
                val_segs.append(losses["segmentation"].item())
                val_Ts.append(losses["texture"].item())
                val_Ss.append(losses["structure"].item())

                # Collect metric data
                if num_classes == 1:
                    all_probs.append(self.prob.detach().cpu().numpy().ravel())
                    all_targets.append(self.gt.detach().cpu().numpy().ravel())
                    all_fovs.append(self.fov.detach().cpu().numpy().ravel())
                elif num_classes == 2:
                    for b in range(self.image.shape[0]):
                        all_disc_preds.append(self.prob[b, 0].cpu())
                        all_disc_gts.append(self.gt[b, 0].cpu())
                        all_cup_preds.append(self.prob[b, 1].cpu())
                        all_cup_gts.append(self.gt[b, 1].cpu())

            # Save visual for the first sample
            if save_img and not visual_saved:
                visual_saved = True
                save_path = osp.join(vis_dir, f"{dataset_name}_epoch_{current_epoch:03d}.png")
                if num_classes == 1:
                    save_drive_visual_sample(
                        structure=self.structure[0],
                        texture=self.texture[0],
                        extracted_structure=self.extracted_structure[0],
                        prob=self.prob[0],
                        fov=self.fov[0] if self.fov is not None else None,
                        save_path=save_path,
                        title_prefix=f"{dataset_name} - Epoch {current_epoch}",
                        threshold=0.5,
                    )
                else:
                    save_origa_visual_sample(
                        structure=self.structure[0],
                        texture=self.texture[0],
                        extracted_structure=self.extracted_structure[0],
                        prob=self.prob[0],
                        fov=self.fov[0] if self.fov is not None else None,
                        save_path=save_path,
                        title_prefix=f"{dataset_name} - Epoch {current_epoch}",
                        threshold=0.5,
                    )

        # Aggregate metrics
        val_metrics = OrderedDict()
        if val_losses:
            val_metrics["val_loss"] = float(np.mean(val_losses))
            val_metrics["val_seg"] = float(np.mean(val_segs))
            val_metrics["val_T"] = float(np.mean(val_Ts))
            val_metrics["val_S"] = float(np.mean(val_Ss))

        if num_classes == 1 and all_probs:
            drive_metrics = calculate_drive_metrics(all_probs, all_targets, all_fovs)
            val_metrics.update(drive_metrics)
        elif num_classes == 2 and all_disc_preds:
            oe_metrics = calculate_overlapping_error(all_disc_preds, all_disc_gts, all_cup_preds, all_cup_gts)
            val_metrics.update(oe_metrics)

        return val_metrics

    def get_current_visuals(self) -> Dict[str, torch.Tensor]:
        out_dict = OrderedDict()
        out_dict["structure"] = self.structure[0].detach().cpu()
        out_dict["texture"] = self.texture[0].detach().cpu()
        out_dict["extracted_structure"] = self.extracted_structure[0].detach().cpu()
        out_dict["prob"] = self.prob[0].detach().cpu()
        if self.gt is not None:
            out_dict["gt"] = self.gt[0].detach().cpu()
        if self.fov is not None:
            out_dict["fov"] = self.fov[0].detach().cpu()
        return out_dict

    def get_current_log(self) -> dict:
        return self.log_dict
