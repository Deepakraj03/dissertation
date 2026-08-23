"""
Compute KID (Kernel Inception Distance, Binkowski et al. 2018) between
generated CycleGAN translations and real rover test images, using clean-fid's
InceptionV3 feature extraction -- the same extractor compute_fid.py already
uses for FID.

KID's estimator is unbiased and well-defined at any n>=2, unlike FID's
Gaussian-covariance estimate (unstable at small/uneven sample sizes). Added
so Track 1, Track 2, and Track 3 (ControlNet/Code/evaluate_mars_model.py,
which already computes KID for exactly this reason) share one common,
sample-size-robust distributional metric, while each track's original FID/
RMSE stays as its primary/literature-comparable number.

Usage:
    python compute_kid.py \\
        --generated-dir /scratch/dr00846/kid_eval/generated_rover_rgb \\
        --real-dir ../Data/processed/rover/test \\
        --checkpoint epoch_100.pt \\
        --output ../Data/eval/kid_results_track1.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cleanfid import fid as cleanfid_fid

ROOT = Path(__file__).parent.parent
DEFAULT_GENERATED_DIR = ROOT / "Data" / "eval" / "generated_rover_rgb"
DEFAULT_REAL_DIR = ROOT / "Data" / "eval" / "real_rover_rgb"
DEFAULT_OUTPUT = ROOT / "Data" / "eval" / "kid_results.json"


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3,
                      gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + coef0) ** degree


def compute_kid(real_features: np.ndarray, gen_features: np.ndarray,
                degree: int = 3, gamma: float | None = None,
                coef0: float = 1.0) -> float:
    """Unbiased polynomial-kernel MMD^2 (Binkowski et al. 2018). Same
    formula as ControlNet/Code/evaluate_mars_model.py's compute_kid --
    duplicated here rather than imported cross-package to keep CycleGAN/Code
    self-contained (matches this project's existing per-package script
    convention, e.g. compute_fid.py's own reimplemented frechet_distance)."""
    n = real_features.shape[0]
    m = gen_features.shape[0]

    k_xx = polynomial_kernel(real_features, real_features, degree, gamma, coef0)
    k_yy = polynomial_kernel(gen_features, gen_features, degree, gamma, coef0)
    k_xy = polynomial_kernel(real_features, gen_features, degree, gamma, coef0)

    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = k_xy.sum() / (n * m)

    return float(sum_xx + sum_yy - 2 * sum_xy)


def compute_kid_score(generated_dir: Path, real_dir: Path) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = cleanfid_fid.build_feature_extractor(
        "clean", device, use_dataparallel=False,
    )

    feats_gen = cleanfid_fid.get_folder_features(
        str(generated_dir), feat_model, num_workers=0, device=device, mode="clean",
    )
    feats_real = cleanfid_fid.get_folder_features(
        str(real_dir), feat_model, num_workers=0, device=device, mode="clean",
    )

    return compute_kid(feats_real, feats_gen)


def write_results(score: float, checkpoint: str, n_generated: int,
                  n_real: int, out_path: Path) -> None:
    payload = {
        "kid": score,
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

    print(f"Computing KID: {n_generated} generated vs {n_real} real images…")
    score = compute_kid_score(generated_dir, real_dir)
    print(f"KID: {score:.5f}")

    write_results(score, args.checkpoint, n_generated, n_real, Path(args.output))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
