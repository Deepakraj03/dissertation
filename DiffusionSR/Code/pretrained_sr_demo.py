"""
Stage 1 (nadir super-resolution) proof-of-concept using a pretrained,
off-the-shelf SR model (Swin2SR, Conde et al., ECCV 2022 workshops —
https://huggingface.co/caidas/swin2SR-classical-sr-x4-64), applied to real
HiRISE patches from the CycleGAN test set.

Standard SR evaluation protocol: take a real high-resolution patch, bicubic-
downsample it 4x to synthesize a low-resolution input, then compare two
upsampling methods against the real (ground-truth) high-resolution patch:
  1. naive bicubic upsampling (baseline)
  2. Swin2SR pretrained model (candidate)

This is a literature-adjacent, low-risk stand-in while the full
DiffusionSat-based Stage 1 fine-tune (Part 3+ of the DiffusionSR pipeline)
is still ahead.

Usage:
    python pretrained_sr_demo.py --input-dir ../../CycleGAN/Data/processed/hirise/test \
        --output-dir ../Data/pretrained_sr_demo --n 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import Swin2SRForImageSuperResolution

MODEL_NAME = "caidas/swin2SR-classical-sr-x4-64"
SCALE = 4


def load_model(device: torch.device) -> Swin2SRForImageSuperResolution:
    model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_NAME)
    model.to(device).eval()
    return model


def bicubic_downsample(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


def bicubic_upsample(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BICUBIC)


def swin2sr_upsample(model: Swin2SRForImageSuperResolution, lr_img: Image.Image,
                     device: torch.device) -> Image.Image:
    """Run the pretrained Swin2SR model on a greyscale LR patch, replicated
    to 3 channels (the model expects RGB) and averaged back to greyscale."""
    rgb = lr_img.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor).reconstruction  # (1, 3, H*scale, W*scale)

    out = out.squeeze(0).clamp(0, 1).cpu().numpy()
    out_gray = out.mean(axis=0)  # (H*scale, W*scale), average RGB -> greyscale
    out_uint8 = (out_gray * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_uint8, mode="L")


def make_comparison_grid(ground_truth: Image.Image, bicubic: Image.Image,
                         swin2sr: Image.Image) -> Image.Image:
    """Side-by-side [ground_truth | bicubic | swin2sr], all same size."""
    w, h = ground_truth.size
    grid = Image.new("L", (w * 3, h))
    grid.paste(ground_truth, (0, 0))
    grid.paste(bicubic, (w, 0))
    grid.paste(swin2sr, (w * 2, 0))
    return grid


def compute_metrics(ground_truth: np.ndarray, candidate: np.ndarray) -> dict:
    psnr = peak_signal_noise_ratio(ground_truth, candidate, data_range=255)
    ssim = structural_similarity(ground_truth, candidate, data_range=255)
    return {"psnr": float(psnr), "ssim": float(ssim)}


def run_demo(input_dir: Path, output_dir: Path, n: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))[:n]
    if not files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    results = []
    for path in files:
        ground_truth = Image.open(path).convert("L")
        gt_size = ground_truth.size

        lr = bicubic_downsample(ground_truth, SCALE)
        bicubic_sr = bicubic_upsample(lr, gt_size)
        swin2sr_sr = swin2sr_upsample(model, lr, device)
        if swin2sr_sr.size != gt_size:
            swin2sr_sr = swin2sr_sr.resize(gt_size, Image.BICUBIC)

        gt_arr = np.array(ground_truth)
        bicubic_metrics = compute_metrics(gt_arr, np.array(bicubic_sr))
        swin2sr_metrics = compute_metrics(gt_arr, np.array(swin2sr_sr))

        grid = make_comparison_grid(ground_truth, bicubic_sr, swin2sr_sr)
        grid_path = output_dir / f"{path.stem}_comparison.png"
        grid.save(grid_path)

        result = {
            "file": path.name,
            "bicubic": bicubic_metrics,
            "swin2sr": swin2sr_metrics,
            "comparison_grid": str(grid_path),
        }
        results.append(result)
        print(f"  {path.name}: bicubic PSNR={bicubic_metrics['psnr']:.2f} "
              f"SSIM={bicubic_metrics['ssim']:.3f} | "
              f"swin2sr PSNR={swin2sr_metrics['psnr']:.2f} "
              f"SSIM={swin2sr_metrics['ssim']:.3f}")

    summary = {
        "model": MODEL_NAME,
        "scale": SCALE,
        "n_images": len(results),
        "mean_bicubic_psnr": float(np.mean([r["bicubic"]["psnr"] for r in results])),
        "mean_bicubic_ssim": float(np.mean([r["bicubic"]["ssim"] for r in results])),
        "mean_swin2sr_psnr": float(np.mean([r["swin2sr"]["psnr"] for r in results])),
        "mean_swin2sr_ssim": float(np.mean([r["swin2sr"]["ssim"] for r in results])),
        "results": results,
    }
    summary_path = output_dir / "sr_demo_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    run_demo(Path(args.input_dir), Path(args.output_dir), args.n)


if __name__ == "__main__":
    main()
