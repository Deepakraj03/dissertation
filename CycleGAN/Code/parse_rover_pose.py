"""Fetch and parse one Navcam EDR product's PDS3 label to recover the
rover's (site, drive) and mast pointing at capture time, then combine with
rover_localization.py's lookup to get a final absolute pose."""

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from rover_localization import SiteDrivePose

NAVCAM_LABEL_BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_navcam_raw/DATA"

ROVER_MOTION_COUNTER_RE = re.compile(
    r"ROVER_MOTION_COUNTER\s*=\s*\(([^)]+)\)"
)
RSM_GROUP_RE = re.compile(
    r"GROUP\s*=\s*RSM_ARTICULATION_STATE_PARMS(.*?)END_GROUP\s*=\s*RSM_ARTICULATION_STATE_PARMS",
    re.DOTALL,
)
ARTICULATION_ANGLE_RE = re.compile(
    r"ARTICULATION_DEVICE_ANGLE\s*=\s*\(([^)]+)\)", re.DOTALL
)


@dataclass
class RoverPose:
    product_id: str
    sol: int
    site: int
    drive: int
    latitude: float
    longitude: float
    mast_azimuth_deg: float
    compass_heading_deg: float


def label_url_for(product_id: str, sol: int) -> str:
    return f"{NAVCAM_LABEL_BASE}/SOL{sol:05d}/{product_id}.LBL"


def parse_navcam_label(label_text: str) -> dict:
    """Extract site, drive, and RSM mast azimuth-measured (degrees) from a
    Navcam PDS3 label's text. Raises ValueError if either required field is
    missing or malformed, so callers can distinguish a real parse failure
    from a network error."""
    counter_match = ROVER_MOTION_COUNTER_RE.search(label_text)
    if not counter_match:
        raise ValueError("ROVER_MOTION_COUNTER not found in label")
    fields = [f.strip() for f in counter_match.group(1).split(",")]
    site, drive = int(fields[0]), int(fields[1])

    rsm_group_match = RSM_GROUP_RE.search(label_text)
    if not rsm_group_match:
        raise ValueError("RSM_ARTICULATION_STATE_PARMS group not found in label")
    angle_match = ARTICULATION_ANGLE_RE.search(rsm_group_match.group(1))
    if not angle_match:
        raise ValueError("ARTICULATION_DEVICE_ANGLE not found in RSM group")
    # First value is AZIMUTH-MEASURED per ARTICULATION_DEVICE_ANGLE_NAME's
    # documented ordering; strip the "<rad>" unit suffix before parsing.
    first_value = angle_match.group(1).split(",")[0]
    azimuth_rad = float(first_value.replace("<rad>", "").strip())

    return {
        "site": site,
        "drive": drive,
        "mast_azimuth_deg": math.degrees(azimuth_rad),
    }


def fetch_and_parse_pose(product_id: str, sol: int,
                         localization: dict[tuple[int, int], SiteDrivePose],
                         ) -> "RoverPose | None":
    """Fetch product_id's label, parse it, and join against localization by
    (site, drive). Returns None (never raises) on any failure — fetch error,
    unparseable label, or a (site, drive) not present in localization —
    since the caller processes many candidates and skip-on-failure is the
    expected common case, not exceptional."""
    try:
        r = requests.get(label_url_for(product_id, sol), timeout=30)
        r.raise_for_status()
        parsed = parse_navcam_label(r.text)
    except Exception:
        return None

    site_drive = localization.get((parsed["site"], parsed["drive"]))
    if site_drive is None:
        return None

    compass_heading = (site_drive.yaw_deg + parsed["mast_azimuth_deg"]) % 360.0

    return RoverPose(
        product_id=product_id,
        sol=sol,
        site=parsed["site"],
        drive=parsed["drive"],
        latitude=site_drive.latitude,
        longitude=site_drive.longitude,
        mast_azimuth_deg=parsed["mast_azimuth_deg"],
        compass_heading_deg=compass_heading,
    )
