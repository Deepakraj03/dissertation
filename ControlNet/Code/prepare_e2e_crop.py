#!/usr/bin/env python
"""Fetch the real Gale HiRISE DTM/ortho and crop the same patch
run_end_to_end_pipeline.py uses, saving just the small cropped arrays to a
.npz file -- no GPU needed for this step. Exists so the ~940MB raw
DTM+ortho download (302MB DTM + 294MB + 343MB orthos, both source
observations) never has to sit on a GPU machine's disk at all -- run this
on any machine with free disk and network access, then ship the tiny
(~1MB) .npz to wherever GPU inference actually runs."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))

from dtm_arrays import load_dtm_arrays
from fetch_hirise_dtm import fetch_dtm_and_orthos

from spotcheck_dtm_estimator_render import crop_window_centered, find_gale_dtm_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
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
        row_start, col_start = crop_window_centered(
            (rows, cols), rows / 2, cols / 2, args.patch_size)
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
