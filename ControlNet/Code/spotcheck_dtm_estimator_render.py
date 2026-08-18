#!/usr/bin/env python
"""Qualitative "shelf artifact" spot-check: render a ground-level view from
the trained DTM estimator's output and compare it side-by-side against a
render from the real stereo DTM, at the same synthetic camera position on
the same real HiRISE product (DTEEC_040770_1755_039280_1755_A01, Gale
Crater -- the project's one held-out real product with both a real stereo
DTM and, separately, real rover-pose target photos in the original 27-pair
paired_controlnet_corpus).

Deliberately does NOT attempt to re-derive one of those 27 real rover
poses exactly (would need the real sol for each product_id to fetch its
PDS3 label -- not recoverable from the product_id text alone, and the
public mars.nasa.gov raw-images search API did not resolve it for spot
checks against real product IDs from this corpus). Instead renders both
DTMs from the same synthetic camera position at the crop's center, using
a steep pitch (-15 deg, within the real MAX_ABS_PITCH_DEG=20 cutoff)
deliberately chosen because the original 2026-08-09/2026-08-13 shelf
findings concentrated in steeper down-look poses -- this directly tests
whether swapping in the estimated DTM introduces or worsens that artifact
relative to the real DTM, holding render geometry fixed. One real target
photo from the original pairing (any pose on this same product) is copied
alongside purely as loose visual reference to real ground texture at this
site -- explicitly NOT camera-matched, and labelled as such.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))

from dtm_coverage import query_dtm_coverage, signed_lon_to_0_360
from dtm_arrays import load_dtm_arrays
from fetch_hirise_dtm import fetch_dtm_and_orthos
from render_ground_view import render_ground_view
from height_encoding import decode_height_from_unit_range

from evaluate_dtm_estimator import load_trained_pipeline, run_inference

GALE_PRODUCT_ID = "DTEEC_040770_1755_039280_1755_A01"
GALE_LAT_MIN, GALE_LAT_MAX = -8.0, -3.0
GALE_LON_MIN, GALE_LON_MAX = 135.0, 140.0
SPOTCHECK_PITCH_DEG = -15.0


def crop_window_centered(raster_shape: tuple, center_row: float, center_col: float,
                         size: int) -> tuple:
    """(row_start, col_start) for a size x size crop centered as close as
    possible to (center_row, center_col), clamped so the crop stays fully
    inside a raster of raster_shape = (rows, cols)."""
    rows, cols = raster_shape
    row_start = int(round(center_row - size / 2))
    col_start = int(round(center_col - size / 2))
    row_start = max(0, min(row_start, rows - size))
    col_start = max(0, min(col_start, cols - size))
    return row_start, col_start


def find_gale_dtm_record():
    records = query_dtm_coverage(
        GALE_LAT_MIN, GALE_LAT_MAX,
        signed_lon_to_0_360(GALE_LON_MIN), signed_lon_to_0_360(GALE_LON_MAX),
    )
    for record in records:
        if record.product_id == GALE_PRODUCT_ID:
            return record
    raise ValueError(f"{GALE_PRODUCT_ID} not found in real Gale DTM coverage query")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet-checkpoint", type=str, required=True)
    parser.add_argument("--unet-lora-checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--reference-photo", type=str, default=None,
                        help="Optional real target photo (any real pose on"
                             " this product) copied alongside for loose,"
                             " NOT camera-matched visual reference")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = find_gale_dtm_record()
    fetch_result = fetch_dtm_and_orthos(record, scratch_dir, scratch_dir)
    if fetch_result["status"] != "ok":
        raise RuntimeError(f"fetch_dtm_and_orthos failed: {fetch_result['status']}")
    dtm_path = Path(fetch_result["dtm_path"])
    ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

    try:
        arrays = load_dtm_arrays(dtm_path, ortho_path)
        rows, cols = arrays.heightmap.shape
        row_start, col_start = crop_window_centered(
            (rows, cols), rows / 2, cols / 2, args.patch_size)
        row_end, col_end = row_start + args.patch_size, col_start + args.patch_size

        real_height_crop = arrays.heightmap[row_start:row_end, col_start:col_end]
        albedo_crop = arrays.albedo[row_start:row_end, col_start:col_end]

        camera_row = camera_col = args.patch_size / 2

        real_render = render_ground_view(
            real_height_crop, albedo_crop, arrays.pixel_scale_m,
            camera_row=camera_row, camera_col=camera_col, heading_deg=0.0,
            pitch_deg=SPOTCHECK_PITCH_DEG,
        )

        pipeline = load_trained_pipeline(
            args.base_model, args.controlnet_checkpoint, args.unet_lora_checkpoint,
            device=args.device,
        )
        ortho_image = Image.fromarray(albedo_crop).convert("L")
        estimated_unit_range = run_inference(pipeline, ortho_image)
        valid_real = real_height_crop[~np.isnan(real_height_crop)]
        reference_meta = {"min": float(valid_real.min()), "max": float(valid_real.max())}
        estimated_height_crop = decode_height_from_unit_range(
            estimated_unit_range, reference_meta)

        estimated_render = render_ground_view(
            estimated_height_crop, albedo_crop, arrays.pixel_scale_m,
            camera_row=camera_row, camera_col=camera_col, heading_deg=0.0,
            pitch_deg=SPOTCHECK_PITCH_DEG,
        )
    finally:
        dtm_path.unlink(missing_ok=True)
        for p in fetch_result.get("ortho_paths", {}).values():
            Path(p).unlink(missing_ok=True)

    Image.fromarray(real_render).save(out_dir / "real_dtm_render.png")
    Image.fromarray(estimated_render).save(out_dir / "estimated_dtm_render.png")
    Image.fromarray(albedo_crop).save(out_dir / "ortho_condition.png")

    comparison = np.concatenate([real_render, estimated_render], axis=1)
    Image.fromarray(comparison).save(out_dir / "real_vs_estimated_comparison.png")

    if args.reference_photo:
        shutil.copy(args.reference_photo,
                    out_dir / "reference_real_photo_NOT_camera_matched.jpg")

    print(f"Spot-check renders written to {out_dir}")


if __name__ == "__main__":
    main()
