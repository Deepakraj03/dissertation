"""
Compute FID between generated CycleGAN translations and real rover test
images, using clean-fid's InceptionV3 feature extraction.

clean-fid 0.1.35's own frechet_distance() is broken on this environment's
scipy (1.18.0 removes sqrtm's `disp` kwarg and changes its return type from
a tuple to a plain array) — verified during planning. This module uses
clean-fid only for feature extraction and reimplements the standard FID
Frechet-distance formula to match current scipy's API.

Usage:
    python compute_fid.py \\
        --generated-dir ../Data/eval/generated_rover_rgb \\
        --real-dir ../Data/eval/real_rover_rgb \\
        --checkpoint epoch_025.pt \\
        --output ../Data/eval/fid_results.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent))
from cleanfid import fid as cleanfid_fid

ROOT = Path(__file__).parent.parent
DEFAULT_GENERATED_DIR = ROOT / "Data" / "eval" / "generated_rover_rgb"
DEFAULT_REAL_DIR = ROOT / "Data" / "eval" / "real_rover_rgb"
DEFAULT_OUTPUT = ROOT / "Data" / "eval" / "fid_results.json"


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """Standard FID Frechet-distance formula (Heusel et al., 2017),
    reimplemented to avoid clean-fid's broken call into scipy.linalg.sqrtm
    on current scipy versions (no `disp` kwarg, returns a plain array
    instead of a (result, errest) tuple)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_fid_score(generated_dir: Path, real_dir: Path) -> float:
    """Compute FID between two directories of RGB images using clean-fid's
    InceptionV3 feature extractor. See module docstring for why this
    doesn't call cleanfid.fid.compute_fid() directly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = cleanfid_fid.build_feature_extractor(
        "clean", device, use_dataparallel=False,
    )

    feats1 = cleanfid_fid.get_folder_features(
        str(generated_dir), feat_model, num_workers=0, device=device, mode="clean",
    )
    mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)

    feats2 = cleanfid_fid.get_folder_features(
        str(real_dir), feat_model, num_workers=0, device=device, mode="clean",
    )
    mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)

    return frechet_distance(mu1, sigma1, mu2, sigma2)


def write_results(score: float, checkpoint: str, n_generated: int,
                  n_real: int, out_path: Path) -> None:
    payload = {
        "fid": score,
        "checkpoint": checkpoint,
        "n_generated": n_generated,
        "n_real": n_real,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=str, default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--real-dir", type=str, default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint filename, recorded in the output JSON only")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    real_dir = Path(args.real_dir)
    n_generated = len(list(generated_dir.glob("*.png")))
    n_real = len(list(real_dir.glob("*.png")))

    print(f"Computing FID: {n_generated} generated vs {n_real} real images…")
    score = compute_fid_score(generated_dir, real_dir)
    print(f"FID: {score:.3f}")

    write_results(score, args.checkpoint, n_generated, n_real, Path(args.output))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
