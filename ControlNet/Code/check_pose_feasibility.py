"""Phase 1 feasibility check: real Gale Crater HiRISE DTM coverage count +
real Navcam pose-label parse success rate, before any bulk fetching. Mirrors
the original geometry-mediated design's Plan 1 coverage check."""

import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
# download_hirise.py and dtm_coverage.py are shared infrastructure that
# stay in CycleGAN/Code.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from download_hirise import REGIONS
from dtm_coverage import query_dtm_coverage, signed_lon_to_0_360
from parse_rover_pose import fetch_and_parse_pose
from rover_localization import (
    LOCALIZATION_CSV_URL, download_localization_csv, parse_localization_csv,
)

NAVCAM_DATA_BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_navcam_raw/DATA"
FULL_FRAME_LBL_RE = re.compile(r'href="(N[LR][AB]_\w+EDR_F\w+\.LBL)"')


def list_navcam_products_for_sol(sol: int) -> list[str]:
    """List full-frame ("_F" variant, not "_T" thumbnail) Navcam product IDs
    available for a given sol, from the label-bearing DATA/ archive path."""
    url = f"{NAVCAM_DATA_BASE}/SOL{sol:05d}/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    filenames = FULL_FRAME_LBL_RE.findall(r.text)
    return [Path(f).stem for f in dict.fromkeys(filenames)]  # dedupe, keep order


def run_feasibility_check(sols: list[int], per_sol: int,
                          localization: dict) -> dict:
    attempted = 0
    parsed = 0
    for sol in sols:
        try:
            products = list_navcam_products_for_sol(sol)[:per_sol]
        except Exception:
            continue
        for product_id in products:
            attempted += 1
            pose = fetch_and_parse_pose(product_id, sol, localization)
            if pose is not None:
                parsed += 1

    return {
        "sols_checked": sols,
        "products_attempted": attempted,
        "products_parsed": parsed,
        "parse_success_rate": (parsed / attempted) if attempted else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sols", type=int, nargs="+",
                        default=[46, 500, 1200, 2000, 3000],
                        help="Sample sols to check (spread across the mission)")
    parser.add_argument("--per-sol", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    index_dir = root / "Data" / "HiRISE_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Querying real Gale Crater HiRISE stereo DTM coverage…")
    cfg = REGIONS["gale_crater"]
    west = signed_lon_to_0_360(cfg["lon_min"])
    east = signed_lon_to_0_360(cfg["lon_max"])
    dtm_records = query_dtm_coverage(cfg["lat_min"], cfg["lat_max"], west, east)
    print(f"Found {len(dtm_records)} DTM product(s) covering Gale Crater")

    print("Downloading MSL rover localization table…")
    csv_path = index_dir / "localized_interp.csv"
    if not download_localization_csv(csv_path):
        print("ERROR: localization CSV download failed")
        sys.exit(1)
    localization = parse_localization_csv(csv_path)
    print(f"Parsed {len(localization)} (site, drive) localization entries")

    print(f"Checking pose-label parse rate across sols {args.sols}…")
    pose_report = run_feasibility_check(args.sols, args.per_sol, localization)
    print(f"Pose parse success rate: {pose_report['parse_success_rate']:.1%} "
         f"({pose_report['products_parsed']}/{pose_report['products_attempted']})")

    report = {
        "gale_crater_dtm_count": len(dtm_records),
        "pose_feasibility": pose_report,
    }
    out_path = index_dir / "controlnet_phase1_feasibility_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
