"""
Quantify the end-to-end pipeline's real-DTM-chain vs estimated-DTM-chain
final outputs (2026-08-22 clean-crop run, `run_end_to_end_pipeline.py`),
which up to now were only compared qualitatively ("visibly different dune/
rock formation vs flatter plain"). Computes SSIM, PSNR, and LPIPS:

  1. real-DTM chain output vs estimated-DTM chain output directly -- turns
     "the DTM estimator's ~2m RMSE visibly affects the final image" into a
     number (lower SSIM / higher LPIPS = more divergence between the two
     chains).
  2. each chain's output vs the loose real reference photo -- NOT camera-
     matched to either render, so this is reported as a caveated sanity
     check, not a fidelity score; a real rover photo taken from a different
     pose than either synthetic camera cannot be a ground-truth target.

Grayscale + resize-to-common-size preprocessing matches
evaluate_mars_model.py's approach (true single-channel casting) for
consistency with Track 3's own SSIM/PSNR numbers, minus that script's global
brightness/contrast correction (not applicable here -- only 2-3 images
total, no "whole generated set" to pool statistics over).

Usage:
    python evaluate_e2e_pipeline.py \\
        --pipeline-dir ../Data/e2e_pipeline_output_newcrop \\
        --output ../Data/eval/e2e_pipeline_metrics.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).parent.parent
DEFAULT_PIPELINE_DIR = ROOT / "Data" / "e2e_pipeline_output_newcrop"
DEFAULT_OUTPUT = ROOT / "Data" / "eval" / "e2e_pipeline_metrics.json"

EVAL_SIZE = 256


def load_gray(path: Path, size: int = EVAL_SIZE) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    return np.array(im, dtype=np.uint8)


def load_rgb_tensor(path: Path, size: int = EVAL_SIZE) -> torch.Tensor:
    """LPIPS expects RGB in [-1, 1], (1, 3, H, W)."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def compute_pair_metrics(path_a: Path, path_b: Path, lpips_model, device) -> dict:
    gray_a, gray_b = load_gray(path_a), load_gray(path_b)
    ssim_score = float(structural_similarity(gray_a, gray_b, data_range=255))
    psnr_score = float(peak_signal_noise_ratio(gray_b, gray_a, data_range=255))
    with torch.no_grad():
        t_a = load_rgb_tensor(path_a).to(device)
        t_b = load_rgb_tensor(path_b).to(device)
        lpips_score = float(lpips_model(t_a, t_b).item())
    return {"ssim": ssim_score, "psnr": psnr_score, "lpips": lpips_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", type=str, default=str(DEFAULT_PIPELINE_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    pdir = Path(args.pipeline_dir)
    real_chain = pdir / "03a_real_dtm_chain_final_output.png"
    est_chain = pdir / "03b_estimated_dtm_chain_final_output.png"
    reference = pdir / "reference_real_photo_NOT_camera_matched.jpg"
    for p in (real_chain, est_chain, reference):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    results = {
        "real_chain_vs_estimated_chain": compute_pair_metrics(
            real_chain, est_chain, lpips_model, device),
        "real_chain_vs_loose_reference_NOT_camera_matched": compute_pair_metrics(
            real_chain, reference, lpips_model, device),
        "estimated_chain_vs_loose_reference_NOT_camera_matched": compute_pair_metrics(
            est_chain, reference, lpips_model, device),
        "eval_size": EVAL_SIZE,
        "interpretation_notes": [
            "real_chain_vs_estimated_chain is the load-bearing comparison: "
            "lower SSIM / higher LPIPS here is direct quantitative evidence "
            "that the DTM estimator's ~2m RMSE propagates to a visibly "
            "different final image, backing the qualitative claim in "
            "Chapter 5.",
            "Both *_vs_loose_reference rows compare against a real rover "
            "photo taken from a DIFFERENT, non-matching camera pose (no "
            "real rover pose exists for this synthetic crop/pose combination "
            "-- see Chapter 4's end-to-end pipeline section) -- treat as a "
            "coarse plausibility/genre check only, not a fidelity score.",
        ],
        "pipeline_dir": str(pdir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps(results, indent=2))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
