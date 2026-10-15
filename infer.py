import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from basicsr.archs.stdnet_arch import STDNetFullModel
from basicsr.data.transforms import RetinalSegmentationTransform, apply_clahe
from basicsr.utils.img_util import save_drive_visual_sample, save_origa_visual_sample


def parse_args():
    parser = argparse.ArgumentParser(description="STD-Net Inference Script")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or image directory.")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to trained .pth model weights.")
    parser.add_argument("--output", "-o", type=str, default="results/inference", help="Output directory.")
    parser.add_argument("--task", type=str, choices=["vessel", "disc_cup"], default="vessel", help="Segmentation task type.")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Input size [H, W]. Default: 512 512 for vessel, 256 256 for disc_cup.")
    parser.add_argument("--apply_clahe", action="store_true", help="Apply CLAHE enhancement.")
    parser.add_argument("--use_polar", action="store_true", help="Apply polar coordinates transform.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Set default image size
    if args.image_size is None:
        img_size = (512, 512) if args.task == "vessel" else (256, 256)
    else:
        img_size = tuple(args.image_size)

    num_classes = 1 if args.task == "vessel" else 2

    # Build model
    model = STDNetFullModel(
        img_channels=3,
        texture_channels=3,
        hidden_dim=64,
        img_size=img_size[0],
        num_classes=num_classes,
    ).to(device)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    # Clean keys if needed
    cleaned_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_dict[name] = v
    model.load_state_dict(cleaned_dict)
    model.eval()

    # Setup transform
    mean = torch.tensor([0.485, 0.456, 0.406] if args.task == "vessel" else [0.5, 0.5, 0.5])
    std = torch.tensor([0.229, 0.224, 0.225] if args.task == "vessel" else [0.5, 0.5, 0.5])
    transform = RetinalSegmentationTransform(
        image_size=img_size,
        mean=mean,
        std=std,
        train=False,
        use_polar=args.use_polar,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in [".jpg", ".png", ".tif", ".tiff", ".bmp", ".jpeg"]])

    print(f"Running inference on {len(image_paths)} images...")

    for img_path in image_paths:
        img_pil = Image.open(img_path).convert("RGB")
        if args.apply_clahe:
            img_pil = apply_clahe(img_pil)

        img_tensor = TF.to_tensor(img_pil)
        dummy_mask = {"fov": torch.ones((1, img_tensor.shape[1], img_tensor.shape[2]))}
        img_tensor, dummy_mask = transform(img_tensor, dummy_mask)
        img_input = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            seg_final, seg5, seg6, seg7, texture, extracted_structure, structure = model(img_input)
            prob = torch.sigmoid(seg_final)[0]

        stem = img_path.stem

        # Save visual comparison
        vis_path = str(out_dir / f"{stem}_visual.png")
        if num_classes == 1:
            save_drive_visual_sample(
                structure=structure[0],
                texture=texture[0],
                extracted_structure=extracted_structure[0],
                prob=prob,
                fov=None,
                save_path=vis_path,
                title_prefix=stem,
                threshold=args.threshold,
            )
            # Save binary mask
            mask_bin = ((prob[0] >= args.threshold).float().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask_bin)
        else:
            save_origa_visual_sample(
                structure=structure[0],
                texture=texture[0],
                extracted_structure=extracted_structure[0],
                prob=prob,
                fov=None,
                save_path=vis_path,
                title_prefix=stem,
                threshold=args.threshold,
            )
            # Save disc and cup masks
            disc_bin = ((prob[0] >= args.threshold).float().cpu().numpy() * 255).astype(np.uint8)
            cup_bin = ((prob[1] >= args.threshold).float().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{stem}_disc.png"), disc_bin)
            cv2.imwrite(str(out_dir / f"{stem}_cup.png"), cup_bin)

        print(f"Processed: {img_path.name} -> Saved in {out_dir}")

    print("Inference completed!")


if __name__ == "__main__":
    main()

