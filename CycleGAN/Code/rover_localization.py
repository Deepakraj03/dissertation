"""Fetch and parse NASA's official MSL rover localization table (the
"PLACES" product), which maps each (site, drive) pair Curiosity has
visited to a real planetocentric lat/lon/elevation/yaw. Per-image PDS3
labels only carry (site, drive) — this table is what turns that into an
actual position on Mars."""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hirise_fullres import download_with_verify

LOCALIZATION_CSV_URL = (
    "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/"
    "data_localizations/localized_interp.csv"
)


@dataclass
class SiteDrivePose:
    site: int
    drive: int
    latitude: float
    longitude: float
    elevation: float
    yaw_deg: float
    sol: int


def download_localization_csv(dest_path: Path) -> bool:
    return download_with_verify(LOCALIZATION_CSV_URL, dest_path)


def parse_localization_csv(csv_path: Path) -> dict[tuple[int, int], SiteDrivePose]:
    """Parse frame=="ROVER" rows into a (site, drive) -> SiteDrivePose dict.
    When multiple `pose` sub-index rows exist for the same (site, drive),
    the pose==-1 row (the site/drive's default entry) wins if present,
    otherwise the first row encountered is kept."""
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
                latitude=float(row["planetocentric_latitude"]),
                longitude=float(row["longitude"]),
                elevation=float(row["elevation"]),
                yaw_deg=float(row["yaw"]),
                sol=int(row["sol"]),
            )
    return result
