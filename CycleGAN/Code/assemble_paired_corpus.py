"""Orchestrate Phase 1's paired ControlNet corpus build: group candidate
Navcam rover poses by which Gale Crater HiRISE DTM covers them, render each
pose's real predicted ground view, pair it with the real photo, and split
into train/val/test — same one-DTM-on-disk-at-a-time lifecycle as
assemble_geometry_corpus.py, extended to also fetch each pose's real target
photo per group."""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dtm_coverage import DtmCoverageRecord, query_dtm_coverage, signed_lon_to_0_360
from dtm_arrays import load_dtm_arrays
from dtm_geo_lookup import (
    find_covering_dtm, latlon_to_dtm_pixel, compass_heading_to_render_heading,
)
from download_hirise import REGIONS
from fetch_hirise_dtm import fetch_dtm_and_orthos
from hirise_fullres import download_with_verify
from parse_rover_pose import RoverPose, fetch_and_parse_pose
from preprocess import save_patch, ENTROPY_TH
from render_ground_view import render_ground_view
from rover_localization import download_localization_csv, parse_localization_csv
from check_pose_feasibility import list_navcam_products_for_sol
from assemble_geometry_corpus import ground_entropy

import rasterio

NAVCAM_JPG_BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_navcam_raw/EXTRAS/FULL"


def fetch_target_photo(product_id: str, sol: int, dest_path: Path) -> bool:
    """Download product_id's real full-frame browse JPG — same archive path
    and filename convention download_rover.py already uses successfully."""
    url = f"{NAVCAM_JPG_BASE}/SOL{sol:05d}/{product_id}.JPG"
    return download_with_verify(url, dest_path)


def group_products_by_covering_dtm(poses: list[RoverPose],
                                   dtm_records: list[DtmCoverageRecord],
                                   ) -> dict[str, list[RoverPose]]:
    grouped: dict[str, list[RoverPose]] = {}
    for pose in poses:
        record = find_covering_dtm(pose.latitude, pose.longitude, dtm_records)
        if record is None:
            continue
        grouped.setdefault(record.product_id, []).append(pose)
    return grouped


def process_dtm_group(dtm_record: DtmCoverageRecord, poses: list[RoverPose],
                      scratch_dir: Path, out_dir: Path) -> dict:
    """Download dtm_record's DTM+ortho once, render every pose in poses
    from its real (row, col, heading) on that DTM, fetch each pose's real
    target photo, save both to out_dir, then delete the raw DTM/ortho
    files regardless of outcome — including when fetch_dtm_and_orthos
    fails partway (e.g. DTM downloaded fine but an ortho download fails),
    which is why the expected raw file paths are computed up front from
    scratch_dir's naming convention rather than only trusted from a
    (possibly absent, on failure) fetch_result."""
    expected_dtm_path = scratch_dir / f"{dtm_record.product_id}.IMG"
    expected_ortho_paths = [
        scratch_dir / f"{obs_id}_ORTHO.JP2"
        for obs_id in (dtm_record.obs_id_a, dtm_record.obs_id_b) if obs_id
    ]
    fetch_result = {}
    try:
        fetch_result = fetch_dtm_and_orthos(dtm_record, scratch_dir, scratch_dir)
        if fetch_result["status"] != "ok":
            return {"product_id": dtm_record.product_id,
                   "status": fetch_result["status"], "pairs_saved": 0,
                   "saved_ids": []}

        dtm_path = Path(fetch_result["dtm_path"])
        ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

        arrays = load_dtm_arrays(dtm_path, ortho_path)
        with rasterio.open(dtm_path) as src:
            transform = src.transform

        out_dir.mkdir(parents=True, exist_ok=True)
        saved_ids = []
        for pose in poses:
            pixel = latlon_to_dtm_pixel(dtm_path, pose.latitude, pose.longitude)
            if pixel is None:
                continue
            row, col = pixel
            heading = compass_heading_to_render_heading(
                pose.compass_heading_deg, transform,
            )
            try:
                condition_map = render_ground_view(
                    arrays.heightmap, arrays.albedo, arrays.pixel_scale_m,
                    camera_row=row, camera_col=col, heading_deg=heading,
                )
            except ValueError:
                continue

            if ground_entropy(condition_map) < ENTROPY_TH:
                # Uninformative render: either genuine sky, or real terrain
                # sitting in a gap the chosen ortho doesn't cover (both
                # paint as sky_value=0 — see dtm_arrays.py's ortho-resample
                # nodata fill). Skip before spending a network call on the
                # target photo for a candidate we're about to discard.
                continue

            target_path = out_dir / f"{pose.product_id}_target.jpg"
            if not fetch_target_photo(pose.product_id, pose.sol, target_path):
                continue

            save_patch(condition_map, out_dir / f"{pose.product_id}_condition.png")
            saved_ids.append(pose.product_id)

        return {"product_id": dtm_record.product_id, "status": "ok",
               "pairs_saved": len(saved_ids), "saved_ids": saved_ids}
    finally:
        Path(fetch_result.get("dtm_path", expected_dtm_path)).unlink(missing_ok=True)
        ortho_paths_to_clean = list(fetch_result.get("ortho_paths", {}).values()) \
            or expected_ortho_paths
        for p in ortho_paths_to_clean:
            Path(p).unlink(missing_ok=True)


def split_pairs_and_move(product_ids: list[str], staging_dir: Path,
                         out_dir: Path, seed: int = 0,
                         train_frac: float = 0.8, val_frac: float = 0.1) -> dict:
    """Shuffle product_ids and move each pair's condition+target files
    together into the same train/val/test split subdir under out_dir."""
    ids = list(product_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    for split_name, split_ids in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for pid in split_ids:
            shutil.move(str(staging_dir / f"{pid}_condition.png"),
                       str(split_dir / f"{pid}_condition.png"))
            shutil.move(str(staging_dir / f"{pid}_target.jpg"),
                       str(split_dir / f"{pid}_target.jpg"))
    return {"total_pairs": n,
           "split_counts": {k: len(v) for k, v in splits.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sols", type=int, nargs="+",
                        default=list(range(1, 4700, 200)),
                        help="Sols to sample candidate Navcam images from")
    parser.add_argument("--per-sol", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    index_dir = root / "Data" / "HiRISE_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    cfg = REGIONS["gale_crater"]
    west = signed_lon_to_0_360(cfg["lon_min"])
    east = signed_lon_to_0_360(cfg["lon_max"])
    print("Querying Gale Crater DTM coverage…")
    dtm_records = query_dtm_coverage(cfg["lat_min"], cfg["lat_max"], west, east)
    print(f"Found {len(dtm_records)} DTM product(s)")

    csv_path = index_dir / "localized_interp.csv"
    if not csv_path.exists() and not download_localization_csv(csv_path):
        print("ERROR: localization CSV download failed")
        sys.exit(1)
    localization = parse_localization_csv(csv_path)

    print(f"Gathering candidate poses across sols {args.sols}…")
    poses = []
    for sol in args.sols:
        try:
            products = list_navcam_products_for_sol(sol)[:args.per_sol]
        except Exception:
            continue
        for product_id in products:
            pose = fetch_and_parse_pose(product_id, sol, localization)
            if pose is not None:
                poses.append(pose)
    print(f"{len(poses)} candidate poses parsed")

    grouped = group_products_by_covering_dtm(poses, dtm_records)
    print(f"{len(grouped)} DTM group(s) with at least one covered pose")

    scratch_dir = root / "Data" / "_paired_corpus_scratch"
    staging_dir = root / "Data" / "processed" / "paired_controlnet_corpus" / "_staging"
    out_dir = root / "Data" / "processed" / "paired_controlnet_corpus"

    manifest = []
    saved_ids = []
    for dtm_product_id, group_poses in grouped.items():
        record = next(r for r in dtm_records if r.product_id == dtm_product_id)
        print(f"\nProcessing DTM {dtm_product_id} ({len(group_poses)} candidate poses)…")
        result = process_dtm_group(record, group_poses, scratch_dir, staging_dir)
        print(f"  {result['status']} — {result.get('pairs_saved', 0)} pairs")
        manifest.append(result)
        saved_ids.extend(result.get("saved_ids", []))

    split_counts = split_pairs_and_move(saved_ids, staging_dir, out_dir)
    print(f"\nFinal split: {split_counts}")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dtm_product_id", "status", "pairs_saved"])
        for row in manifest:
            writer.writerow([row["product_id"], row["status"], row.get("pairs_saved", 0)])
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
