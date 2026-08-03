"""Orchestrate the geometry-mediated corpus build: for each of the 8
landing-region HiRISE stereo DTMs, download it, render many candidate
ground-level views via render_ground_view, keep the ones that pass the
existing entropy quality filter, and assemble a train/val/test split —
same disk-quota-aware one-product-at-a-time lifecycle as hirise_fullres.py.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dtm_coverage import DtmCoverageRecord
from dtm_arrays import load_dtm_arrays
from fetch_hirise_dtm import fetch_dtm_and_orthos
from render_ground_view import render_ground_view
from preprocess import entropy, ENTROPY_TH, save_patch, split_and_move

# Landing-region bounds from Plan 1's "Coverage check result"
# (docs/superpowers/plans/2026-08-02-hirise-dtm-coverage-and-fetch.md) —
# the 8 products whose footprint overlaps this box.
LANDING_LAT_MIN, LANDING_LAT_MAX = 17.5, 19.0
LANDING_LON_MIN, LANDING_LON_MAX = 334.5, 336.5


def sample_camera_positions(heightmap_shape: tuple[int, int], margin_px: int,
                            n: int, seed: int) -> list[tuple[float, float, float]]:
    """Sample n distinct (row, col, heading_deg) camera positions at least
    margin_px from any edge of a heightmap_shape array. Returns [] if the
    margin leaves no valid interior (matches hirise_fullres.py's
    sample_patch_positions convention)."""
    h, w = heightmap_shape
    max_row, max_col = h - 1 - margin_px, w - 1 - margin_px
    if max_row < margin_px or max_col < margin_px:
        return []

    rng = random.Random(seed)
    positions_set = set()
    max_attempts = n * 10
    attempts = 0
    while len(positions_set) < n and attempts < max_attempts:
        row = rng.randint(margin_px, max_row)
        col = rng.randint(margin_px, max_col)
        positions_set.add((row, col))
        attempts += 1

    return [(row, col, rng.uniform(0, 360)) for row, col in positions_set]


def process_dtm_product(record: DtmCoverageRecord, scratch_dir: Path,
                        staging_dir: Path, n_crops: int = 200,
                        seed: int = 0) -> dict:
    """Download record's DTM+ortho, render up to n_crops candidate ground
    views at random camera positions (oversampling n_crops*4 candidates and
    skipping any that land on a NaN/nodata heightmap cell), keep the ones
    passing the entropy filter, save them to staging_dir, delete the raw
    downloaded files, and return a status dict."""
    fetch_result = fetch_dtm_and_orthos(record, scratch_dir, scratch_dir)
    if fetch_result["status"] != "ok":
        return {"product_id": record.product_id, "status": fetch_result["status"],
                "crops_saved": 0}

    dtm_path = Path(fetch_result["dtm_path"])
    ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

    try:
        arrays = load_dtm_arrays(dtm_path, ortho_path)

        staging_dir.mkdir(parents=True, exist_ok=True)
        # Oversample candidates 4x n_crops to compensate for both nodata-gap
        # skips and entropy-filter rejections — mirrors the oversampling
        # pattern already used by extract_qualifying_patches in
        # hirise_fullres.py, rather than assuming every candidate qualifies.
        positions = sample_camera_positions(
            arrays.heightmap.shape, margin_px=35, n=n_crops * 4, seed=seed,
        )

        saved = 0
        nan_skipped = 0
        for i, (row, col, heading) in enumerate(positions):
            if saved >= n_crops:
                break
            if np.isnan(arrays.heightmap[int(row), int(col)]):
                nan_skipped += 1
                continue
            crop = render_ground_view(
                arrays.heightmap, arrays.albedo, arrays.pixel_scale_m,
                camera_row=row, camera_col=col, heading_deg=heading,
            )
            if entropy(crop) < ENTROPY_TH:
                continue
            save_patch(crop, staging_dir / f"{record.product_id}_p{i:04d}.png")
            saved += 1

        return {"product_id": record.product_id, "status": "ok",
                "crops_saved": saved, "nan_skipped": nan_skipped}
    finally:
        dtm_path.unlink(missing_ok=True)
        for p in fetch_result["ortho_paths"].values():
            Path(p).unlink(missing_ok=True)


def _in_landing_region(rec: dict) -> bool:
    lat_overlap = (LANDING_LAT_MIN <= rec["min_lat"] <= LANDING_LAT_MAX or
                  LANDING_LAT_MIN <= rec["max_lat"] <= LANDING_LAT_MAX)
    lon_overlap = (LANDING_LON_MIN <= rec["min_lon"] <= LANDING_LON_MAX or
                  LANDING_LON_MIN <= rec["max_lon"] <= LANDING_LON_MAX)
    return lat_overlap and lon_overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-crops", type=int, default=200,
                        help="Candidate crops rendered per DTM product")
    parser.add_argument("--coverage-report",
                        default=None,
                        help="Path to dtm_coverage_oxia_planum.json (default: CycleGAN/Data/HiRISE_index/dtm_coverage_oxia_planum.json)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    report_path = Path(args.coverage_report) if args.coverage_report else (
        root / "Data" / "HiRISE_index" / "dtm_coverage_oxia_planum.json"
    )
    report = json.loads(report_path.read_text())
    landing_records = [DtmCoverageRecord(**r) for r in report["records"]
                       if _in_landing_region(r)]
    print(f"Processing {len(landing_records)} landing-region DTM products…")

    scratch_dir = root / "Data" / "_geometry_corpus_scratch"
    staging_dir = root / "Data" / "processed" / "geometry_corpus" / "_staging"
    out_dir = root / "Data" / "processed" / "geometry_corpus"

    manifest = []
    for record in landing_records:
        print(f"\nProcessing {record.product_id}…")
        result = process_dtm_product(record, scratch_dir, staging_dir,
                                     n_crops=args.n_crops)
        print(f"  {result['status']} — {result.get('crops_saved', 0)} crops")
        manifest.append(result)

    staged = sorted(staging_dir.glob("*.png"))
    split_counts = split_and_move(staged, out_dir)
    print(f"\nFinal split: {split_counts}")

    manifest_path = out_dir / "geometry_corpus_manifest.json"
    manifest_path.write_text(json.dumps({
        "products": manifest, "split_counts": split_counts,
    }, indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
