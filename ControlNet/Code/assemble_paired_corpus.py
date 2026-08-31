"""Orchestrate Phase 1's paired ControlNet corpus build: group candidate
Navcam rover poses by which Gale Crater HiRISE DTM covers them, render each
pose's real predicted ground view, pair it with the real photo, and split
into train/val/test — same one-DTM-on-disk-at-a-time lifecycle as
assemble_geometry_corpus.py, extended to also fetch each pose's real target
photo per group."""

import argparse
import csv
import dataclasses
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# dtm_coverage.py, dtm_arrays.py, download_hirise.py, fetch_hirise_dtm.py,
# hirise_fullres.py, preprocess.py, render_ground_view.py, and
# assemble_geometry_corpus.py are shared HiRISE/DTM infrastructure that
# stays in CycleGAN/Code (also used by CycleGAN's own Track 2 geometry
# corpus) -- bridge to it rather than duplicating it into ControlNet/Code.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from dtm_coverage import DtmCoverageRecord, query_dtm_coverage, signed_lon_to_0_360
from dtm_arrays import load_dtm_arrays
from dtm_geo_lookup import (
    find_covering_dtm, latlon_to_dtm_pixel, compass_heading_to_render_heading,
    _parse_wkt_polygon, _point_in_polygon,
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

# Found 2026-08-09: even with correct pitch modeling (render_ground_view's
# pitch_deg), a real steep down-look pose (close range) renders as a
# low-detail, DTM-resolution-limited "shelf" pattern that doesn't match the
# real photo's fine near-field detail -- a spatial-resolution mismatch
# between the ~1m/px HiRISE DTM and cm-scale close-range content, not
# something the entropy filter alone catches (both real examples that
# exposed this passed entropy). Most real Navcam full-frame poses are
# steeper than this (median real elevation ~-38 deg, see the 2026-08-09
# candidate survey), so this trades corpus yield for per-pair fidelity.
#
# Recalibrated 2026-08-13 against real Gale Crater samples (see
# render_pitch_bucket_samples.py, 2026-08-13-multi-site-data-expansion.md
# Task 4): the 2026-08-09 shelf artifact was checked in three wider pitch
# buckets and was found in 5/5 20-25deg samples, 5/5 25-35deg samples, and
# 5/5 35-50deg samples (full landed totals agree: 6/6, 5/5, 8/8) -- every
# inspected condition map showed either the same blocky low-detail "shelf"
# as the real photo's near-field texture, or, in several 20-25deg and most
# 35-50deg cases, a complete black-frame render (render_ground_view's
# per-column ray march hits a NaN albedo sample on its first in-bounds
# step at these steeper elevation angles and aborts the column before
# drawing anything -- an even more severe version of the same DTM/ortho
# coverage-limited failure, not a separate bug specific to this tool).
# Since the shelf/failure pattern was in the MAJORITY (not the minority)
# of every bucket checked, including the bucket immediately past the
# existing cutoff, there is no wider threshold the 2026-08-09 tradeoff
# rationale supports -- MAX_ABS_PITCH_DEG stays at 20.0.
MAX_ABS_PITCH_DEG = 20.0


def render_pose_condition_map(dtm_path: Path, arrays, transform,
                              pose: RoverPose) -> "np.ndarray | None":
    """Render pose's real predicted ground view against arrays (already
    loaded via load_dtm_arrays), or None if pose's lat/lon falls outside
    dtm_path's raster or the render hits a NaN-adjacent DTM cell. Extracted
    from process_dtm_group's inline loop body so the pitch-bucket
    investigation tool (render_pitch_bucket_samples.py) renders through
    the exact same path production does, not a parallel reimplementation
    that could silently diverge."""
    pixel = latlon_to_dtm_pixel(dtm_path, pose.latitude, pose.longitude)
    if pixel is None:
        return None
    row, col = pixel
    heading = compass_heading_to_render_heading(pose.compass_heading_deg, transform)
    try:
        return render_ground_view(
            arrays.heightmap, arrays.albedo, arrays.pixel_scale_m,
            camera_row=row, camera_col=col, heading_deg=heading,
            pitch_deg=pose.pitch_deg,
        )
    except ValueError:
        return None


def fetch_target_photo(product_id: str, sol: int, dest_path: Path) -> bool:
    """Download product_id's real full-frame browse JPG — same archive path
    and filename convention download_rover.py already uses successfully."""
    url = f"{NAVCAM_JPG_BASE}/SOL{sol:05d}/{product_id}.JPG"
    return download_with_verify(url, dest_path)


def group_products_by_covering_dtm(poses: list[RoverPose],
                                   dtm_records: list[DtmCoverageRecord],
                                   ) -> dict[str, list[RoverPose]]:
    """Same real point-in-polygon coverage test as find_covering_dtm
    (dtm_geo_lookup.py), but each record's WKT footprint is parsed once
    here rather than once per (pose, record) pair — found 2026-08-31 to be
    the dominant real cost of this step at dense-sampling scale (20,281
    poses x up to 25 DTMs was re-parsing the same handful of polygons
    roughly half a million times); does not change dtm_geo_lookup.py's own
    shared find_covering_dtm, which other callers still use unmodified."""
    parsed_records = [
        (record, _parse_wkt_polygon(record.footprint_wkt)) if record.footprint_wkt
        else (record, None)
        for record in dtm_records
    ]
    grouped: dict[str, list[RoverPose]] = {}
    for pose in poses:
        record = None
        for candidate, polygon in parsed_records:
            if polygon is not None:
                if _point_in_polygon(pose.longitude, pose.latitude, polygon):
                    record = candidate
                    break
            elif (candidate.min_lat <= pose.latitude <= candidate.max_lat and
                    candidate.min_lon <= pose.longitude <= candidate.max_lon):
                record = candidate
                break
        if record is None:
            continue
        grouped.setdefault(record.product_id, []).append(pose)
    return grouped


def process_dtm_group(dtm_record: DtmCoverageRecord, poses: list[RoverPose],
                      scratch_dir: Path, out_dir: Path,
                      max_pitch_deg: float = MAX_ABS_PITCH_DEG,
                      entropy_th: float = ENTROPY_TH,
                      source_mission: str = "MSL") -> dict:
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
                   "saved_ids": [], "source_mission": source_mission}

        dtm_path = Path(fetch_result["dtm_path"])
        ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))

        arrays = load_dtm_arrays(dtm_path, ortho_path)
        with rasterio.open(dtm_path) as src:
            transform = src.transform

        out_dir.mkdir(parents=True, exist_ok=True)
        saved_ids = []
        rejections = {"pitch": 0, "render_failed": 0, "low_entropy": 0, "fetch_failed": 0}
        for pose in poses:
            if abs(pose.pitch_deg) > max_pitch_deg:
                rejections["pitch"] += 1
                continue
            condition_map = render_pose_condition_map(dtm_path, arrays, transform, pose)
            if condition_map is None:
                rejections["render_failed"] += 1
                continue

            if ground_entropy(condition_map) < entropy_th:
                # Uninformative render: either genuine sky, or real terrain
                # sitting in a gap the chosen ortho doesn't cover (both
                # paint as sky_value=0 — see dtm_arrays.py's ortho-resample
                # nodata fill). Skip before spending a network call on the
                # target photo for a candidate we're about to discard.
                rejections["low_entropy"] += 1
                continue

            target_path = out_dir / f"{pose.product_id}_target.jpg"
            if not fetch_target_photo(pose.product_id, pose.sol, target_path):
                rejections["fetch_failed"] += 1
                continue

            save_patch(condition_map, out_dir / f"{pose.product_id}_condition.png")
            saved_ids.append(pose.product_id)

        return {"product_id": dtm_record.product_id, "status": "ok",
               "pairs_saved": len(saved_ids), "saved_ids": saved_ids,
               "source_mission": source_mission, "rejections": rejections,
               "candidates": len(poses)}
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
    together into the same train/val/test split subdir under out_dir.

    Clears each split subdir before writing into it: a re-run against a
    different total product_ids count reshuffles which product IDs land
    in which split, so leaving a prior run's files in place would let old
    and new pairs silently mix (or the same ID sit stale in a split this
    run didn't put it in). Task 8's implementer hit exactly this and
    manually deleted stale files before running -- this makes re-running
    safe by default instead of relying on that being remembered."""
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
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for pid in split_ids:
            shutil.move(str(staging_dir / f"{pid}_condition.png"),
                       str(split_dir / f"{pid}_condition.png"))
            shutil.move(str(staging_dir / f"{pid}_target.jpg"),
                       str(split_dir / f"{pid}_target.jpg"))
    return {"total_pairs": n,
           "split_counts": {k: len(v) for k, v in splits.items()}}


def gather_candidate_poses(sols: list[int], per_sol: int,
                           localization: dict) -> list[RoverPose]:
    """List up to per_sol full-frame Navcam products per sol in sols,
    parse each into a real pose, skipping (not raising on) a sol whose
    listing request fails or a product whose label doesn't parse --
    same posture as every other fetch step in this pipeline."""
    poses = []
    for sol in sols:
        try:
            products = list_navcam_products_for_sol(sol)[:per_sol]
        except Exception:
            continue
        for product_id in products:
            pose = fetch_and_parse_pose(product_id, sol, localization)
            if pose is not None:
                poses.append(pose)
    return poses


def query_region_dtm_coverage(region_key: str) -> list[DtmCoverageRecord]:
    """REGIONS[region_key]'s bounding box, queried for real HiRISE stereo
    DTM coverage. Thin wrapper so main() is driven by --region instead of
    the previously hardcoded 'gale_crater' key."""
    cfg = REGIONS[region_key]
    west = signed_lon_to_0_360(cfg["lon_min"])
    east = signed_lon_to_0_360(cfg["lon_max"])
    return query_dtm_coverage(cfg["lat_min"], cfg["lat_max"], west, east)


def build_arg_parser() -> argparse.ArgumentParser:
    """Separated from main() so tests exercise the real, fully-configured
    parser (choices, defaults, help text) instead of a hand-typed copy
    that could silently drift from what main() actually runs — e.g. the
    --region choices excluding oxia_planum, which a regression test needs
    to check against the genuine parser to mean anything."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sols", type=int, nargs="+",
                        default=list(range(1, 4700, 50)),
                        help="Sols to sample candidate Navcam images from")
    parser.add_argument("--per-sol", type=int, default=8)
    parser.add_argument("--max-pitch-deg", type=float, default=MAX_ABS_PITCH_DEG,
                        help="Max abs(pitch_deg) for a pose to be rendered "
                             "(see MAX_ABS_PITCH_DEG's module comment)")
    parser.add_argument("--entropy-th", type=float, default=ENTROPY_TH,
                        help="Minimum ground_entropy (bits) for a rendered "
                             "condition map to be kept (preprocess.py default: "
                             f"{ENTROPY_TH})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the default "
                             "Data/processed/paired_controlnet_corpus output "
                             "location, so a filter-tuning experiment doesn't "
                             "overwrite the corpus currently in use.")
    # Oxia Planum is explicitly out of scope — no real rover target photos
    # exist there, and this pipeline requires paired condition maps + real targets.
    parser.add_argument("--region", type=str, default="gale_crater",
                        choices=[k for k in REGIONS if k != "oxia_planum"],
                        help="REGIONS key to query DTM coverage for")
    parser.add_argument("--cache-poses-to", type=Path, default=None,
                        help="Write gathered candidate poses to this JSON path "
                             "after the (network-bound) gather step, so a later "
                             "run can replay filter changes via --load-poses-from "
                             "without re-fetching every label over the network.")
    parser.add_argument("--load-poses-from", type=Path, default=None,
                        help="Load candidate poses from a JSON file written by a "
                             "prior --cache-poses-to run instead of gathering them "
                             "from the network. --sols/--per-sol are ignored when set.")
    return parser


def write_manifest(manifest: list[dict], saved_ids: list[str],
                   manifest_path: Path) -> bool:
    """Write manifest rows (including the spec's source_mission provenance
    column) to manifest_path — unless saved_ids is empty, in which case
    warn and skip the write entirely rather than overwriting a non-empty
    tracked manifest.csv with a bare-header file (e.g. --region
    jezero_crater currently has no working pose-fetch path and produces 0
    pairs). Returns True if the manifest was written, False if skipped."""
    if len(saved_ids) == 0:
        print(f"WARNING: 0 pairs produced — check region/sols/pitch settings; "
             f"NOT overwriting {manifest_path}")
        return False
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dtm_product_id", "status", "pairs_saved", "source_mission"])
        for row in manifest:
            writer.writerow([row["product_id"], row["status"], row.get("pairs_saved", 0),
                             row.get("source_mission", "MSL")])
    return True


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    index_dir = root / "Data" / "HiRISE_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying {args.region} DTM coverage…")
    dtm_records = query_region_dtm_coverage(args.region)
    print(f"Found {len(dtm_records)} DTM product(s)")

    csv_path = index_dir / "localized_interp.csv"
    if not csv_path.exists() and not download_localization_csv(csv_path):
        print("ERROR: localization CSV download failed")
        sys.exit(1)
    localization = parse_localization_csv(csv_path)

    if args.load_poses_from is not None:
        print(f"Loading cached candidate poses from {args.load_poses_from}…")
        raw = json.loads(args.load_poses_from.read_text())
        poses = [RoverPose(**p) for p in raw]
        print(f"{len(poses)} candidate poses loaded (no network fetch)")
    else:
        print(f"Gathering candidate poses across sols {args.sols}…")
        poses = gather_candidate_poses(args.sols, args.per_sol, localization)
        print(f"{len(poses)} candidate poses parsed")
        if args.cache_poses_to is not None:
            args.cache_poses_to.parent.mkdir(parents=True, exist_ok=True)
            args.cache_poses_to.write_text(
                json.dumps([dataclasses.asdict(p) for p in poses]))
            print(f"Cached {len(poses)} poses to {args.cache_poses_to}")

    grouped = group_products_by_covering_dtm(poses, dtm_records)
    print(f"{len(grouped)} DTM group(s) with at least one covered pose")

    out_dir = args.output_dir if args.output_dir is not None else (
        root / "Data" / "processed" / "paired_controlnet_corpus")
    scratch_dir = root / "Data" / "_paired_corpus_scratch"
    staging_dir = out_dir / "_staging"
    # Clear any leftover staging content from a prior run (e.g. one that
    # crashed partway through) before writing into it -- found 2026-08-31
    # to otherwise silently mix stale pairs from an interrupted run into a
    # fresh one's output, since split_pairs_and_move only clears the final
    # train/val/test dirs, not this intermediate directory.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    manifest = []
    saved_ids = []
    for dtm_product_id, group_poses in grouped.items():
        record = next(r for r in dtm_records if r.product_id == dtm_product_id)
        print(f"\nProcessing DTM {dtm_product_id} ({len(group_poses)} candidate poses)…")
        result = process_dtm_group(record, group_poses, scratch_dir, staging_dir,
                                   max_pitch_deg=args.max_pitch_deg,
                                   entropy_th=args.entropy_th)
        print(f"  {result['status']} — {result.get('pairs_saved', 0)} pairs"
              f"  (rejections: {result.get('rejections', {})})")
        manifest.append(result)
        saved_ids.extend(result.get("saved_ids", []))

    split_counts = split_pairs_and_move(saved_ids, staging_dir, out_dir)
    print(f"\nFinal split: {split_counts}")

    manifest_path = out_dir / "manifest.csv"
    if write_manifest(manifest, saved_ids, manifest_path):
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
