"""One-time investigation tool: render real candidate poses across pitch
buckets steeper than the current MAX_ABS_PITCH_DEG=20.0 cutoff, saving each
next to its real target photo for manual visual comparison, to determine
whether the known DTM-resolution "shelf" artifact (see
assemble_paired_corpus.py's MAX_ABS_PITCH_DEG comment) reappears at wider
thresholds before committing to one. Not part of the production pipeline --
run once, inspect output, then update MAX_ABS_PITCH_DEG by hand."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from dtm_coverage import query_dtm_coverage, signed_lon_to_0_360
from dtm_arrays import load_dtm_arrays
from download_hirise import REGIONS
from fetch_hirise_dtm import fetch_dtm_and_orthos
from hirise_fullres import download_with_verify

from assemble_paired_corpus import (
    fetch_target_photo, gather_candidate_poses, group_products_by_covering_dtm,
    render_pose_condition_map,
)
from parse_rover_pose import RoverPose
from preprocess import save_patch
from rover_localization import download_localization_csv, parse_localization_csv

import rasterio
from PIL import Image

DEFAULT_BUCKETS = [(20.0, 25.0), (25.0, 35.0), (35.0, 50.0)]


def bucket_poses_by_pitch(poses: list[RoverPose],
                          buckets: list[tuple[float, float]],
                          ) -> dict[str, list[RoverPose]]:
    """Assign each pose to the first bucket (low, high) where
    low <= abs(pitch_deg) < high. A pose whose abs(pitch_deg) falls in no
    bucket is excluded (e.g. poses already under MAX_ABS_PITCH_DEG, which
    the current pipeline already handles fine and don't need investigating).
    Every bucket key is always present in the result, even if empty."""
    result = {f"{low}-{high}": [] for low, high in buckets}
    for pose in poses:
        abs_pitch = abs(pose.pitch_deg)
        for low, high in buckets:
            if low <= abs_pitch < high:
                result[f"{low}-{high}"].append(pose)
                break
    return result


def save_side_by_side(condition_map, target_path: Path, out_path: Path) -> None:
    """condition_map (grayscale np.ndarray) next to the real target photo,
    for fast visual comparison during the manual recalibration pass."""
    cond_img = Image.fromarray(condition_map).convert("RGB")
    target_img = Image.open(target_path).convert("RGB").resize(cond_img.size)
    combined = Image.new("RGB", (cond_img.width * 2, cond_img.height))
    combined.paste(cond_img, (0, 0))
    combined.paste(target_img, (cond_img.width, 0))
    combined.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sols", type=int, nargs="+",
                        default=list(range(1, 4700, 100)))
    parser.add_argument("--per-sol", type=int, default=6)
    parser.add_argument("--samples-per-bucket", type=int, default=5)
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).parent.parent / "Data" /
                                   "pitch_recalibration_samples"))
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    index_dir = root / "Data" / "HiRISE_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = REGIONS["gale_crater"]
    west = signed_lon_to_0_360(cfg["lon_min"])
    east = signed_lon_to_0_360(cfg["lon_max"])
    print("Querying Gale Crater DTM coverage…")
    dtm_records = query_dtm_coverage(cfg["lat_min"], cfg["lat_max"], west, east)

    csv_path = index_dir / "localized_interp.csv"
    if not csv_path.exists() and not download_localization_csv(csv_path):
        print("ERROR: localization CSV download failed")
        sys.exit(1)
    localization = parse_localization_csv(csv_path)

    print(f"Gathering candidate poses across sols {args.sols}…")
    poses = gather_candidate_poses(args.sols, args.per_sol, localization)
    print(f"{len(poses)} candidate poses parsed")

    grouped = group_products_by_covering_dtm(poses, dtm_records)
    buckets = DEFAULT_BUCKETS
    scratch_dir = root / "Data" / "_pitch_recal_scratch"

    for dtm_product_id, group_poses in grouped.items():
        record = next(r for r in dtm_records if r.product_id == dtm_product_id)
        bucketed = bucket_poses_by_pitch(group_poses, buckets)
        if not any(bucketed.values()):
            continue

        print(f"\nDTM {dtm_product_id}: fetching for pitch-bucket sampling…")
        fetch_result = fetch_dtm_and_orthos(record, scratch_dir, scratch_dir)
        if fetch_result["status"] != "ok":
            print(f"  skip ({fetch_result['status']})")
            continue
        try:
            dtm_path = Path(fetch_result["dtm_path"])
            ortho_path = Path(next(iter(fetch_result["ortho_paths"].values())))
            arrays = load_dtm_arrays(dtm_path, ortho_path)
            with rasterio.open(dtm_path) as src:
                transform = src.transform

            for bucket_key, bucket_poses in bucketed.items():
                for pose in bucket_poses[:args.samples_per_bucket]:
                    condition_map = render_pose_condition_map(
                        dtm_path, arrays, transform, pose)
                    if condition_map is None:
                        continue
                    target_path = scratch_dir / f"{pose.product_id}_target.jpg"
                    if not fetch_target_photo(pose.product_id, pose.sol, target_path):
                        continue
                    bucket_dir = out_dir / bucket_key
                    bucket_dir.mkdir(parents=True, exist_ok=True)
                    save_side_by_side(
                        condition_map, target_path,
                        bucket_dir / f"{pose.product_id}_side_by_side.png")
                    target_path.unlink(missing_ok=True)
                    print(f"  [{bucket_key}] saved {pose.product_id}")
        finally:
            Path(fetch_result.get("dtm_path", "")).unlink(missing_ok=True)
            for p in fetch_result.get("ortho_paths", {}).values():
                Path(p).unlink(missing_ok=True)

    print(f"\nDone. Inspect side-by-side comparisons under {out_dir}/<bucket>/")


if __name__ == "__main__":
    main()
