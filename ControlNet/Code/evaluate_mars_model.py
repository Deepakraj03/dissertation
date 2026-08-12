"""
Comprehensive, domain-appropriate evaluation for the Mars ControlNet model
(HiRISE-derived condition map -> Navcam-style photorealistic translation).

FID proved misleading here (~438, n=9 real vs n=9 generated) for two
independent, compounding reasons: (1) FID's Gaussian-covariance estimate
is unstable at this sample size, and (2) a genuine RGB/ImageNet color-prior
mismatch -- Stable Diffusion's pretrained natural-photo prior imparts a
residual warm tint and darker overall brightness that real (true single-
channel) Navcam photos never have, which InceptionV3 features are highly
sensitive to independent of perceptual quality.

This script fixes both:
  - Forces both sets to true single-channel grayscale, then applies a
    single GLOBAL (whole-set, not per-pair) brightness/contrast
    correction before any metric is computed. Global, not per-pair,
    matters: correcting each generated image using knowledge of its own
    paired target would leak information into paired metrics (SSIM/PSNR)
    and inflate them unfairly -- this only ever uses aggregate statistics
    of the whole generated/real sets.
  - SSIM / PSNR (skimage): paired, pixel-level structural/fidelity
    metrics. Caveat: these measure how closely each output reconstructs
    its OWN specific real photo's exact content -- a stricter, different
    question than "is this a plausible scene given the geometry". The
    model hallucinates unseen content (specific rocks, sand ripples) it
    was never shown, so modest SSIM/PSNR here does not by itself mean
    poor quality -- see the printed interpretation notes.
  - KID (Kernel Inception Distance, Binkowski et al. 2018): unbiased
    polynomial-kernel MMD^2 between InceptionV3 feature sets, reusing the
    same clean-fid feature extractor compute_fid.py already validated on
    this environment. Unlike FID's Gaussian assumption, KID's estimator
    is unbiased and well-defined at small n -- appropriate for n=9.

Usage:
    python evaluate_mars_model.py
    python evaluate_mars_model.py --normalize histogram
    python evaluate_mars_model.py --condition-dir ../Data/processed/paired_controlnet_corpus/test \\
        --generated-dir ../Data/eval/generated_controlnet_rgb \\
        --output ../Data/eval/mars_model_eval.json --plots-dir ./evaluation_plots
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.exposure import match_histograms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).parent))
# image_utils.py is shared infrastructure that stays in CycleGAN/Code.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from cleanfid import fid as cleanfid_fid
from image_utils import to_rgb

ROOT = Path(__file__).parent.parent
DEFAULT_CONDITION_DIR = ROOT / "Data" / "processed" / "paired_controlnet_corpus" / "test"
DEFAULT_GENERATED_DIR = ROOT / "Data" / "eval" / "generated_controlnet_rgb"
DEFAULT_OUTPUT = ROOT / "Data" / "eval" / "mars_model_eval.json"
DEFAULT_PLOTS_DIR = ROOT / "evaluation_plots"


# --------------------------------------------------------------------------
# Preprocessing & normalization
# --------------------------------------------------------------------------

def load_grayscale(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load path as true single-channel luminance (never a channel-0
    extraction of an already-tinted RGB image), optionally resized."""
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def compute_aggregate_stats(images: list[np.ndarray]) -> tuple[float, float]:
    """Mean and std across every pixel of every image in the list -- a
    population statistic of the whole set, never of a single image."""
    stacked = np.concatenate([im.ravel() for im in images]).astype(np.float64)
    return float(stacked.mean()), float(stacked.std())


def normalize_mean_std(image: np.ndarray, src_mean: float, src_std: float,
                       ref_mean: float, ref_std: float) -> np.ndarray:
    """Affine-correct image from the (src_mean, src_std) population it was
    drawn from to the (ref_mean, ref_std) population. src/ref must be
    GLOBAL set statistics (see compute_aggregate_stats), never a specific
    image's own paired partner's statistics -- that would leak per-pair
    information into paired metrics computed after this."""
    corrected = (image.astype(np.float64) - src_mean) / (src_std + 1e-8) * ref_std + ref_mean
    return np.clip(corrected, 0, 255).astype(np.uint8)


def normalize_histogram_global(image: np.ndarray, pooled_reference: np.ndarray) -> np.ndarray:
    """Match image's histogram to pooled_reference -- a single reference
    built once from ALL real images pooled together (see main()), never
    from one specific paired real image. pooled_reference may be any 2D
    array (e.g. a flat pool of pixels reshaped to a column) -- only its
    intensity distribution matters, not its shape; skimage requires it
    only to share image's ndim, not its exact shape."""
    reference_2d = np.reshape(pooled_reference.astype(np.float64), (-1, 1))
    matched = match_histograms(image.astype(np.float64), reference_2d)
    return np.clip(matched, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Paired pixel-level metrics
# --------------------------------------------------------------------------

def compute_ssim_psnr(generated: np.ndarray, real: np.ndarray) -> dict:
    ssim_score, ssim_map = structural_similarity(generated, real, full=True, data_range=255)
    psnr_score = peak_signal_noise_ratio(real, generated, data_range=255)
    return {"ssim": float(ssim_score), "psnr": float(psnr_score), "ssim_map": ssim_map}


# --------------------------------------------------------------------------
# KID (unbiased polynomial-kernel MMD^2)
# --------------------------------------------------------------------------

def extract_features_from_arrays(images: list[np.ndarray], feat_model, device,
                                 tmp_dir: Path) -> np.ndarray:
    """images are (H, W) grayscale uint8 arrays. clean-fid's extractor
    operates on a folder (same pattern compute_fid.py uses), so this
    writes RGB (channel-replicated) PNGs to tmp_dir first."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stale in tmp_dir.glob("*.png"):
        stale.unlink()
    for i, im in enumerate(images):
        Image.fromarray(to_rgb(im), mode="RGB").save(tmp_dir / f"{i:04d}.png")
    return cleanfid_fid.get_folder_features(
        str(tmp_dir), feat_model, num_workers=0, device=device, mode="clean")


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3,
                      gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + coef0) ** degree


def compute_kid(real_features: np.ndarray, gen_features: np.ndarray,
                degree: int = 3, gamma: float | None = None,
                coef0: float = 1.0) -> float:
    """Unbiased polynomial-kernel MMD^2 between two feature sets (Binkowski
    et al. 2018, "Demystifying MMD GANs"). Uses every available sample as
    a single subset rather than averaging many random subsets (the
    subsetting in the original paper exists only to make computation
    tractable at 10k+ samples -- unnecessary and not applicable at n=9).
    Requires n>=2 and m>=2 (the unbiased within-set terms divide by
    n*(n-1) and m*(m-1))."""
    n = real_features.shape[0]
    m = gen_features.shape[0]

    k_xx = polynomial_kernel(real_features, real_features, degree, gamma, coef0)
    k_yy = polynomial_kernel(gen_features, gen_features, degree, gamma, coef0)
    k_xy = polynomial_kernel(real_features, gen_features, degree, gamma, coef0)

    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = k_xy.sum() / (n * m)

    return float(sum_xx + sum_yy - 2 * sum_xy)


# --------------------------------------------------------------------------
# Visual comparison grids
# --------------------------------------------------------------------------

def apply_heat_colormap(normalized: np.ndarray) -> np.ndarray:
    """normalized: (H, W) float array in [0, 1]. Hand-rolled black ->
    dark-red -> orange -> pale-yellow heat ramp -- no matplotlib
    dependency, matching this codebase's minimal dependency set."""
    control_points = np.array([
        [0.00, 0.00, 0.00],
        [0.50, 0.00, 0.00],
        [1.00, 0.25, 0.00],
        [1.00, 0.75, 0.00],
        [1.00, 1.00, 0.60],
    ])
    xs = np.linspace(0, 1, len(control_points))
    r = np.interp(normalized, xs, control_points[:, 0])
    g = np.interp(normalized, xs, control_points[:, 1])
    b = np.interp(normalized, xs, control_points[:, 2])
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def make_comparison_grid(condition: np.ndarray, real: np.ndarray,
                         generated: np.ndarray, ssim_map: np.ndarray,
                         out_path: Path, panel_size: int = 256) -> None:
    """4-panel grid: condition map, real target, generated output, and a
    1-SSIM structural-residual heatmap (brighter = more structurally
    different from the real photo at that location)."""
    labels = ["Condition (input)", "Real (ground truth)",
             "Generated (ControlNet)", "1 - SSIM residual"]

    cond_img = Image.fromarray(condition).convert("RGB").resize(
        (panel_size, panel_size), Image.LANCZOS)
    real_img = Image.fromarray(real).convert("RGB").resize(
        (panel_size, panel_size), Image.LANCZOS)
    gen_img = Image.fromarray(generated).convert("RGB").resize(
        (panel_size, panel_size), Image.LANCZOS)

    residual = 1.0 - ssim_map
    span = residual.max() - residual.min()
    residual_norm = (residual - residual.min()) / (span if span > 1e-8 else 1.0)
    residual_rgb = apply_heat_colormap(residual_norm)
    residual_img = Image.fromarray(residual_rgb).resize(
        (panel_size, panel_size), Image.NEAREST)

    panels = [cond_img, real_img, gen_img, residual_img]

    gap = 4
    label_h = 22
    grid = Image.new("RGB", (panel_size * 4 + gap * 3, panel_size + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(grid)
    x = 0
    for panel, label in zip(panels, labels):
        grid.paste(panel, (x, label_h))
        draw.text((x + 4, 5), label, fill=(225, 225, 225))
        x += panel_size + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def find_test_triples(condition_dir: Path, generated_dir: Path) -> list[dict]:
    """Pair every generated_dir/<id>.png with condition_dir/<id>_condition.png
    and condition_dir/<id>_target.jpg. Skips a generated image with no
    matching pair on disk rather than raising, since a partial eval run
    (e.g. --limit during generation) is a normal, non-exceptional case."""
    triples = []
    for gen_path in sorted(generated_dir.glob("*.png")):
        product_id = gen_path.stem
        condition_path = condition_dir / f"{product_id}_condition.png"
        real_path = condition_dir / f"{product_id}_target.jpg"
        if condition_path.exists() and real_path.exists():
            triples.append({"id": product_id, "condition": condition_path,
                            "real": real_path, "generated": gen_path})
    return triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-dir", type=str, default=str(DEFAULT_CONDITION_DIR))
    parser.add_argument("--generated-dir", type=str, default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--plots-dir", type=str, default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument("--normalize", choices=["mean_std", "histogram", "none"],
                        default="mean_std",
                        help="Global (whole-set) brightness/color correction "
                             "applied to the generated set before any metric.")
    parser.add_argument("--eval-size", type=int, default=512,
                        help="Common resolution real/generated images are "
                             "resized to before SSIM/PSNR/KID.")
    args = parser.parse_args()

    condition_dir = Path(args.condition_dir)
    generated_dir = Path(args.generated_dir)
    plots_dir = Path(args.plots_dir)

    triples = find_test_triples(condition_dir, generated_dir)
    if not triples:
        print(f"No matching triples found between {condition_dir} and {generated_dir}")
        sys.exit(1)
    print(f"Found {len(triples)} test triples")

    size = (args.eval_size, args.eval_size)
    generated_imgs = [load_grayscale(t["generated"], size) for t in triples]
    real_imgs = [load_grayscale(t["real"], size) for t in triples]
    condition_imgs = [load_grayscale(t["condition"]) for t in triples]

    if args.normalize == "mean_std":
        gen_mean, gen_std = compute_aggregate_stats(generated_imgs)
        real_mean, real_std = compute_aggregate_stats(real_imgs)
        print(f"Generated set (pre-norm): mean={gen_mean:.1f} std={gen_std:.1f}")
        print(f"Real set:                 mean={real_mean:.1f} std={real_std:.1f}")
        generated_norm = [normalize_mean_std(im, gen_mean, gen_std, real_mean, real_std)
                          for im in generated_imgs]
    elif args.normalize == "histogram":
        pooled_reference = np.concatenate([im.ravel() for im in real_imgs])
        generated_norm = [normalize_histogram_global(im, pooled_reference)
                          for im in generated_imgs]
    else:
        generated_norm = generated_imgs

    per_sample = []
    ssim_maps = []
    for t, gen, real in zip(triples, generated_norm, real_imgs):
        m = compute_ssim_psnr(gen, real)
        per_sample.append({"id": t["id"], "ssim": m["ssim"], "psnr": m["psnr"]})
        ssim_maps.append(m["ssim_map"])

    mean_ssim = float(np.mean([p["ssim"] for p in per_sample]))
    mean_psnr = float(np.mean([p["psnr"] for p in per_sample]))

    print("Extracting InceptionV3 features for KID…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = cleanfid_fid.build_feature_extractor("clean", device, use_dataparallel=False)
    tmp_real = plots_dir / "_tmp_kid_real"
    tmp_gen = plots_dir / "_tmp_kid_gen"
    real_features = extract_features_from_arrays(real_imgs, feat_model, device, tmp_real)
    gen_features = extract_features_from_arrays(generated_norm, feat_model, device, tmp_gen)
    kid_score = compute_kid(real_features, gen_features)
    shutil.rmtree(tmp_real, ignore_errors=True)
    shutil.rmtree(tmp_gen, ignore_errors=True)

    print(f"Writing comparison grids to {plots_dir}/ …")
    for t, cond, real, gen, ssim_map in zip(triples, condition_imgs, real_imgs,
                                            generated_norm, ssim_maps):
        make_comparison_grid(cond, real, gen, ssim_map,
                             plots_dir / f"{t['id']}_comparison.png")

    results = {
        "n_samples": len(triples),
        "normalize_method": args.normalize,
        "eval_size": args.eval_size,
        "mean_ssim": mean_ssim,
        "mean_psnr": mean_psnr,
        "kid": kid_score,
        "per_sample": per_sample,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interpretation_notes": [
            "SSIM/PSNR measure exact pixel-level reconstruction of each "
            "specific real photo -- a stricter, different question from "
            "whether the output is a plausible scene given the geometry. "
            "The model hallucinates unseen content (specific rocks, sand "
            "ripples) it was never shown, so modest SSIM/PSNR here does "
            "not by itself mean poor quality.",
            "KID (unbiased polynomial-kernel MMD^2) is well-defined at "
            "small n, unlike FID's Gaussian-covariance estimate -- "
            "preferred over FID for this test set size.",
            f"Generated images were normalized ('{args.normalize}') using "
            "GLOBAL whole-set statistics before any metric was computed, "
            "never per-pair, so paired metrics (SSIM/PSNR) cannot be "
            "inflated by peeking at each image's own real target.",
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nn={len(triples)}  mean SSIM={mean_ssim:.4f}  "
         f"mean PSNR={mean_psnr:.2f} dB  KID={kid_score:.5f}")
    print(f"Results written to {out_path}")
    print(f"Comparison grids written to {plots_dir}/")


if __name__ == "__main__":
    main()
