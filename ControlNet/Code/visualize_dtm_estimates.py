#!/usr/bin/env python
"""One-off diagnostic (same precedent as render_pitch_bucket_samples.py):
run the trained DTM estimator over all real held-out Gale patches and save
a visual [ortho | real height | estimated height] comparison PNG per
patch, labeled with its RMSE, so results can be inspected by eye rather
than only as an aggregate number. Not a pipeline component -- ad hoc
reporting tool."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_dtm_estimator import (
    compute_rmse, find_gale_orig_patch_pairs, load_trained_pipeline, run_inference,
    DEFAULT_GALE_DTM_PRODUCT_ID,
)
from height_encoding import decode_height_from_unit_range


def height_to_uint8(height: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (height - vmin) / (vmax - vmin) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--unet-lora-checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--gale-product-id", type=str, default=DEFAULT_GALE_DTM_PRODUCT_ID)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_gale_orig_patch_pairs(args.test_dir, args.gale_product_id)
    pipeline = load_trained_pipeline(
        args.base_model, args.controlnet_checkpoint, args.unet_lora_checkpoint,
        device=args.device,
    )

    results = []
    for condition_path, target_path in pairs:
        ortho_image = Image.open(condition_path).convert("L")
        real_height = np.load(target_path)

        estimated_unit_range = run_inference(pipeline, ortho_image)
        valid_real = real_height[~np.isnan(real_height)]
        vmin, vmax = float(valid_real.min()), float(valid_real.max())
        estimated_height = decode_height_from_unit_range(
            estimated_unit_range, {"min": vmin, "max": vmax})

        rmse = compute_rmse(estimated_height, real_height)

        real_vis = height_to_uint8(np.nan_to_num(real_height, nan=vmin), vmin, vmax)
        est_vis = height_to_uint8(estimated_height, vmin, vmax)
        ortho_arr = np.array(ortho_image)

        comparison = np.concatenate([ortho_arr, real_vis, est_vis], axis=1)
        base = condition_path.name.removesuffix("_condition.png")
        out_path = out_dir / f"{base}_rmse{rmse:.3f}_ortho_real_estimated.png"
        Image.fromarray(comparison).save(out_path)
        results.append((rmse, out_path))
        print(f"{base}: RMSE = {rmse:.4f} m -> {out_path.name}")

    results.sort(key=lambda r: r[0])
    print("\nBest (lowest RMSE) patches:")
    for rmse, path in results[:3]:
        print(f"  {rmse:.4f} m  {path.name}")


if __name__ == "__main__":
    main()
