"""Jezero Crater / Mars 2020 (Perseverance) version of
assemble_paired_corpus.py's real pose-matched paired ControlNet corpus
builder. Reuses assemble_paired_corpus.py's DTM-group orchestration
(process_dtm_group, group_products_by_covering_dtm, split_pairs_and_move,
write_manifest) unchanged, since RoverPose's shape is identical between
missions and that orchestration only depends on those fields, not on
mission-specific parsing. Only the three genuinely mission-specific pieces
are swapped in: candidate-pose gathering (real PDS4 label parsing via
parse_m2020_navcam_pose.py, real position lookup via
m2020_rover_localization.py) and the real target-photo fetch (the
mars2020_navcam_ops_calibrated bundle's browse/ PNG mirror, not MSL's JPG
archive).

Real access path, label schema, and browse-image path all verified
2026-08-29 against real fetched products -- see parse_m2020_navcam_pose.py's
module docstring for the verification detail. Candidate poses are drawn
from the real DTM-overlap check already run this session: 2 of 32 Jezero
DTM groups (DTEEC_045994_1985_046060_1985_U01,
DTEEC_022680_1985_022746_1985_A01) cover 214 real Perseverance sols."""

import argparse
import csv
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))

import assemble_paired_corpus as apc
from dtm_coverage import query_dtm_coverage, signed_lon_to_0_360
from download_hirise import REGIONS
from hirise_fullres import download_with_verify
from m2020_rover_localization import (
    LOCALIZATION_CSV_URL, download_localization_csv, parse_localization_csv,
)
from parse_m2020_navcam_pose import RoverPose, fetch_and_parse_pose

NAVCAM_BROWSE_BASE = (
    "https://planetarydata.jpl.nasa.gov/img/data/mars2020/"
    "mars2020_navcam_ops_calibrated/browse/sol"
)

# Full-frame product-type prefixes worth candidate-checking, from the real
# inventory's LID product-type distribution (2026-08-29): nlf/nrf (~106.7k,
# the general engineering full-frame product) and nlb/nrb/nlr/nrr (~1.2k,
# the smaller stereo-pair-specific type this module's pose parser was
# originally verified against). nlg/nrg (~75k) and nlm/nrm (~26.5k)
# excluded: 'g' is a distinct (likely lower-fidelity/compressed) product
# type not yet verified to carry the same label geometry fields this
# parser depends on, and 'm' is the thumbnail/medium type MSL's own
# pipeline also excludes (see check_pose_feasibility.py's "_F, not _T"
# comment) -- narrower candidate set now, easy to widen once f-type yield
# is characterised.
FULL_FRAME_PREFIXES = ("nlf_", "nrf_", "nlb_", "nrb_", "nlr_", "nrr_")


FULL_FRAME_LBL_RE = re.compile(r'href="(N[LR][BGFRM]_[^"]+FDR_N[^"]+)\.xml"')


def load_candidate_sols(inventory_csv: Path) -> list[int]:
    """Parse the real collection_data_inventory.csv (one LID per real
    product) into a sorted list of every sol that has at least one
    FULL_FRAME_PREFIXES / fdr_n candidate. Used only to know WHICH sols are
    worth a real directory listing -- the LID's own product-id text is not
    used to construct a filename (found 2026-08-29: a trailing version-ish
    suffix, "03" for early-mission products, differs later in the mission
    -- e.g. sol 658's real filenames end "...0A0295J" with no equivalent
    suffix at all -- so reconstructing filenames from the LID is not safe
    across the mission; list_navcam_products_for_sol below does a real,
    exact listing instead for whichever sols this function says to check)."""
    sols: set[int] = set()
    with open(inventory_csv) as f:
        for line in f:
            if not line.startswith("P,"):
                continue
            lid = line.strip().split(",", 1)[1].rsplit("::", 1)[0]
            stem = lid.rsplit(":data:", 1)[-1]
            if not stem.startswith(FULL_FRAME_PREFIXES) or "fdr_n" not in stem:
                continue
            parts = stem.split("_")
            try:
                sols.add(int(parts[1]))
            except (IndexError, ValueError):
                continue
    return sorted(sols)


def list_navcam_products_for_sol(sol: int, per_sol_hint: int = 20) -> list[str]:
    """Real directory listing of sol's data/sol/{sol}/ids/fdr/ncam/ path,
    returning full-frame ('FDR_N' variant, matching this module's parser)
    product-id stems -- exact real filenames, not reconstructed from the
    inventory LID (see load_candidate_sols). Mirrors
    check_pose_feasibility.list_navcam_products_for_sol's MSL approach."""
    url = f"https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_navcam_ops_calibrated/data/sol/{sol:05d}/ids/fdr/ncam/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    stems = FULL_FRAME_LBL_RE.findall(r.text)
    return list(dict.fromkeys(stems))[:per_sol_hint]  # dedupe (dup <a> entries), keep order


def fetch_m2020_target_photo(product_id: str, sol: int, dest_path: Path) -> bool:
    """Real target-photo fetch for M2020: the ops_calibrated bundle's
    browse/ PNG mirror of the same sol/ids/fdr/ncam/ path the real label
    lives at (verified 2026-08-29 -- see this module's docstring). Saved
    with a .jpg extension to match assemble_paired_corpus.py's existing
    target_path naming convention even though the real source format is
    PNG; save_patch-adjacent code only cares about openability via PIL,
    which handles either format transparently."""
    url = f"{NAVCAM_BROWSE_BASE}/{sol:05d}/ids/fdr/ncam/{product_id}.png"
    return download_with_verify(url, dest_path)


def gather_m2020_candidate_poses(sols: list[int], per_sol: int,
                                  localization: dict) -> list[RoverPose]:
    """For each sol, get real product filenames via a live directory
    listing (list_navcam_products_for_sol), then fetch+parse each one's
    real label. Skips (not raises on) a sol whose listing request fails,
    same posture as gather_candidate_poses in assemble_paired_corpus.py."""
    poses = []
    for sol in sols:
        try:
            products = list_navcam_products_for_sol(sol, per_sol)
        except Exception:
            continue
        for product_id in products:
            pose = fetch_and_parse_pose(product_id, sol, localization)
            if pose is not None:
                poses.append(pose)
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-csv", type=Path, required=True,
                        help="Path to a pre-downloaded collection_data_inventory.csv "
                        "(see this module's docstring for the real source URL).")
    parser.add_argument("--sols", type=int, nargs="+", default=None,
                        help="Sols to sample. Default: every sol with at least "
                        "one candidate product in the inventory.")
    parser.add_argument("--per-sol", type=int, default=8)
    parser.add_argument("--max-pitch-deg", type=float, default=apc.MAX_ABS_PITCH_DEG)
    parser.add_argument("--region", type=str, default="jezero_crater")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Reuse assemble_paired_corpus.py's orchestration unchanged by
    # monkey-patching its module-level, late-bound fetch_target_photo
    # reference -- process_dtm_group looks this name up at call time, so
    # this is a clean swap, not a fork of that function.
    apc.fetch_target_photo = fetch_m2020_target_photo

    print(f"Loading candidate sols from {args.inventory_csv}…")
    candidate_sols = load_candidate_sols(args.inventory_csv)
    print(f"{len(candidate_sols)} distinct sols have at least one full-frame candidate")

    sols = args.sols if args.sols is not None else candidate_sols
    print(f"Checking {len(sols)} sols (up to {args.per_sol} products each, "
          f"via a real directory listing per sol)…")

    cfg = REGIONS[args.region]
    west = signed_lon_to_0_360(cfg["lon_min"])
    east = signed_lon_to_0_360(cfg["lon_max"])
    print(f"Querying {args.region} DTM coverage…")
    dtm_records = query_dtm_coverage(cfg["lat_min"], cfg["lat_max"], west, east)
    print(f"Found {len(dtm_records)} DTM product(s)")

    csv_path = args.output_dir.parent / "m2020_places.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() and not download_localization_csv(csv_path):
        print("ERROR: M2020 localization CSV download failed")
        sys.exit(1)
    localization = parse_localization_csv(csv_path)
    print(f"{len(localization)} (site, drive) localization entries loaded")

    print("Gathering and parsing real candidate poses (this fetches one real "
          "directory listing per sol plus one real XML label per candidate "
          "-- the slow, network-bound step)…")
    poses = gather_m2020_candidate_poses(sols, args.per_sol, localization)
    print(f"{len(poses)} candidate poses parsed")

    grouped = apc.group_products_by_covering_dtm(poses, dtm_records)
    print(f"{len(grouped)} DTM group(s) with at least one covered pose")

    scratch_dir = args.output_dir.parent / "_paired_corpus_scratch_m2020"
    staging_dir = args.output_dir / "_staging"

    manifest = []
    saved_ids = []
    for dtm_product_id, group_poses in grouped.items():
        record = next(r for r in dtm_records if r.product_id == dtm_product_id)
        print(f"\nProcessing DTM {dtm_product_id} ({len(group_poses)} candidate poses)…")
        result = apc.process_dtm_group(record, group_poses, scratch_dir, staging_dir,
                                       max_pitch_deg=args.max_pitch_deg,
                                       source_mission="M2020")
        print(f"  {result['status']} — {result.get('pairs_saved', 0)} pairs")
        manifest.append(result)
        saved_ids.extend(result.get("saved_ids", []))

    split_counts = apc.split_pairs_and_move(saved_ids, staging_dir, args.output_dir)
    print(f"\nFinal split: {split_counts}")

    manifest_path = args.output_dir / "manifest.csv"
    if apc.write_manifest(manifest, saved_ids, manifest_path):
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
