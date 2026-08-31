"""Fetch and parse NASA's official Mars 2020 (Perseverance) rover
localization table (the PLACES-equivalent product), which maps each
(site, drive) pair to a real planetodetic lat/lon/elevation. Mirrors
rover_localization.py's MSL interface exactly, since parse_m2020_navcam_pose.py
needs the same (site, drive) -> position lookup MSL's pipeline already uses;
the two missions' localization products differ only in column names and
host, not in structure -- see this module's LOCALIZATION_CSV_URL docstring
for the real, verified difference (planetodetic_latitude vs MSL's
planetocentric_latitude, confirmed 2026-08-29 by inspecting real column
headers)."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from hirise_fullres import download_with_verify

LOCALIZATION_CSV_URL = (
    "https://pds-geosciences.wustl.edu/m2020/"
    "urn-nasa-pds-mars2020_rover_places/data_localizations/best_interp.csv"
)


@dataclass
class SiteDrivePose:
    site: int
    drive: int
    latitude: float
    longitude: float
    elevation: float
    sol: int


def download_localization_csv(dest_path: Path) -> bool:
    return download_with_verify(LOCALIZATION_CSV_URL, dest_path)


def parse_localization_csv(csv_path: Path) -> dict[tuple[int, int], SiteDrivePose]:
    """Parse frame=="ROVER" rows into a (site, drive) -> SiteDrivePose dict.
    Verified 2026-08-29: every real row in this table already has pose==-1
    (the single default entry per site/drive, unlike MSL's table which mixes
    -1-default rows with other pose sub-indices) -- so no pose disambiguation
    is needed here, but the check is kept for parity with
    rover_localization.py and as a safety net against a future data release
    changing that."""
    result: dict[tuple[int, int], SiteDrivePose] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["frame"] != "ROVER":
                continue
            key = (int(row["site"]), int(row["drive"]))
            is_default_pose = int(row["pose"]) == -1
            if key in result and not is_default_pose:
                continue
            result[key] = SiteDrivePose(
                site=key[0],
                drive=key[1],
                latitude=float(row["planetodetic_latitude"]),
                longitude=float(row["longitude"]),
                elevation=float(row["elevation"]),
                sol=int(row["sol"]),
            )
    return result
