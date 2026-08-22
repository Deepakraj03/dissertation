#!/usr/bin/env python
"""Scan the real Gale HiRISE DTM/ortho for a crop location whose crude
render actually has ground structure near the camera, instead of always
using run_end_to_end_pipeline.py's default raster-center crop -- which,
per project_dtm_estimator's 2026-08-20 finding, stayed dominated by the
sky/shelf artifact at both -15deg and -5deg pitch, giving no real test of
whether the DTM estimator's ~2m RMSE visibly propagates to the final
ControlNet output.

Reuses assemble_geometry_corpus.py's random-camera-position sampler and
ground_entropy quality metric -- the same mechanism that filtered the
geometry corpus -- but ranks candidates by score instead of just
accept/reject against a fixed threshold, and writes the single best one out
as a prepare_e2e_crop.py-compatible .npz (real_height_crop/albedo_crop/
pixel_scale_m) for run_end_to_end_pipeline.py --precomputed-crop-npz.

Heading is fixed at 0deg for every candidate (matching what
run_end_to_end_pipeline.py always renders at) -- scoring a candidate at a
heading that render won't actually use would make the ranking meaningless.
Scores are computed against the FULL heightmap/ortho (not a pre-cropped
window), then the winning position is cropped afterward: render_ground_view
only samples within max_range_m (30m) of the camera, well inside a
patch_size/2 margin at these DTMs' ~1-2m/px resolution, so the post-hoc crop
reproduces the same render.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))

from dtm_arrays import load_dtm_arrays
from fetch_hirise_dtm import fetch_dtm_and_orthos
from render_ground_view import render_ground_view
from assemble_geometry_corpus import sample_camera_positions, ground_entropy

from spotcheck_dtm_estimator_render import crop_window_centered, find_gale_dtm_record


def score_candidates(heightmap: np.ndarray, albedo: np.ndarray, pixel_scale_m: float,
                     candidates: list, pitch_deg: float) -> list:
    """Render each (row, col) candidate at heading=0/pitch_deg against the
    full heightmap/albedo and score it by ground_entropy (higher = more
    real ground structure actually drawn, not sky). Candidates
    render_ground_view can't render (camera on/adjacent to a NaN DTM cell)
    are silently skipped, matching assemble_geometry_corpus.py's handling."""
    results = []
    for row, col in candidates:
        try:
            render = render_ground_view(
                heightmap, albedo, pixel_scale_m,
                camera_row=row, camera_col=col, heading_deg=0.0, pitch_deg=pitch_deg,
            )
        except ValueError:
            continue
        results.append({
            "row": row, "col": col,
            "score": ground_entropy(render),
            "sky_frac": float((render == 0).mean()),
        })
    return results


def pick_best_candidate(scored: list) -> dict:
    if not scored:
        raise ValueError("no valid candidates to pick from")
    return max(scored, key=lambda c: c["score"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--pitch-deg", type=float, default=-15.0,
                        help="Pitch to score candidates at -- match whatever"
                             " run_end_to_end_pipeline.py --pitch-deg you"
                             " intend to run with the resulting crop")
    parser.add_argument("--n-candidates", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-npz", type=str, required=True)
    args = parser.parse_args()

    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching real Gale DTM/ortho...")
    record = find_gale_dtm_record()
    fetch_result = fetch_dtm_and_orthos(record, scratch_dir, scratch_dir)
    if fetch_result["status"] != "ok":
        raise RuntimeError(f"fetch_dtm_and_orthos failed: {fetch_result['status']}")
    dtm_path = Path(fetch_result["dtm_path"])
    ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

    try:
        arrays = load_dtm_arrays(dtm_path, ortho_path)
        rows, cols = arrays.heightmap.shape

        default_row, default_col = rows / 2, cols / 2
        margin_px = args.patch_size // 2 + 5
        sampled = sample_camera_positions(
            arrays.heightmap.shape, margin_px=margin_px, n=args.n_candidates, seed=args.seed)
        candidates = [(default_row, default_col)] + [(row, col) for row, col, _ in sampled]

        scored = score_candidates(
            arrays.heightmap, arrays.albedo, arrays.pixel_scale_m, candidates, args.pitch_deg)
        if not scored:
            raise RuntimeError("every candidate position failed to render (all NaN neighborhoods?)")

        default_scored = next(
            (c for c in scored if c["row"] == default_row and c["col"] == default_col), None)
        if default_scored is not None:
            print(f"Default center crop: ground_entropy={default_scored['score']:.3f} "
                 f"sky_frac={default_scored['sky_frac']:.2f}")

        best = pick_best_candidate(scored)
        print(f"Best of {len(scored)} scored candidates: row={best['row']:.0f} "
             f"col={best['col']:.0f} ground_entropy={best['score']:.3f} "
             f"sky_frac={best['sky_frac']:.2f}")

        row_start, col_start = crop_window_centered(
            arrays.heightmap.shape, best["row"], best["col"], args.patch_size)
        row_end, col_end = row_start + args.patch_size, col_start + args.patch_size

        real_height_crop = arrays.heightmap[row_start:row_end, col_start:col_end]
        albedo_crop = arrays.albedo[row_start:row_end, col_start:col_end]
        pixel_scale_m = arrays.pixel_scale_m
    finally:
        dtm_path.unlink(missing_ok=True)
        for p in fetch_result.get("ortho_paths", {}).values():
            Path(p).unlink(missing_ok=True)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, real_height_crop=real_height_crop, albedo_crop=albedo_crop,
             pixel_scale_m=np.float64(pixel_scale_m))
    print(f"Saved crop ({real_height_crop.shape}) to {out_npz}"
         f" ({out_npz.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
