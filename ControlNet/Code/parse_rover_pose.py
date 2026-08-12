"""Fetch and parse one Navcam EDR product's PDS3 label to recover the
rover's (site, drive) and real camera boresight pointing at capture time,
then combine with rover_localization.py's lookup to get a final absolute
pose."""

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
SITE_DERIVED_GEOMETRY_RE = re.compile(
    r"GROUP\s*=\s*SITE_DERIVED_GEOMETRY_PARMS(.*?)END_GROUP\s*=\s*SITE_DERIVED_GEOMETRY_PARMS",
    re.DOTALL,
)
INSTRUMENT_AZIMUTH_RE = re.compile(r"INSTRUMENT_AZIMUTH\s*=\s*([\-\d.]+)")
INSTRUMENT_ELEVATION_RE = re.compile(r"INSTRUMENT_ELEVATION\s*=\s*([\-\d.]+)")


@dataclass
class RoverPose:
    product_id: str
    sol: int
    site: int
    drive: int
    latitude: float
    longitude: float
    compass_heading_deg: float
    pitch_deg: float


def label_url_for(product_id: str, sol: int) -> str:
    return f"{NAVCAM_LABEL_BASE}/SOL{sol:05d}/{product_id}.LBL"


def parse_navcam_label(label_text: str) -> dict:
    """Extract site, drive, and the real (JPL-derived, absolute site-frame)
    camera boresight azimuth/elevation from a Navcam PDS3 label's text.
    Uses SITE_DERIVED_GEOMETRY_PARMS's INSTRUMENT_AZIMUTH/INSTRUMENT_ELEVATION
    rather than composing RSM joint angles with rover body yaw ourselves —
    JPL's own geometry pipeline already resolves the full kinematic chain
    (RSM + rover + coordinate frames) into this single absolute value, and
    it's the only source that carries real elevation at all (found
    2026-08-09: most real Navcam full-frame shots are steep down-look
    arm-workspace shots, not horizon shots — elevation is essential, not
    optional). Raises ValueError if any required field is missing or
    malformed, so callers can distinguish a real parse failure from a
    network error."""
    counter_match = ROVER_MOTION_COUNTER_RE.search(label_text)
    if not counter_match:
        raise ValueError("ROVER_MOTION_COUNTER not found in label")
    fields = [f.strip() for f in counter_match.group(1).split(",")]
    site, drive = int(fields[0]), int(fields[1])

    geom_match = SITE_DERIVED_GEOMETRY_RE.search(label_text)
    if not geom_match:
        raise ValueError("SITE_DERIVED_GEOMETRY_PARMS group not found in label")
    az_match = INSTRUMENT_AZIMUTH_RE.search(geom_match.group(1))
    el_match = INSTRUMENT_ELEVATION_RE.search(geom_match.group(1))
    if not az_match or not el_match:
        raise ValueError(
            "INSTRUMENT_AZIMUTH/ELEVATION not found in SITE_DERIVED_GEOMETRY_PARMS group")

    return {
        "site": site,
        "drive": drive,
        "azimuth_deg": float(az_match.group(1)),
        "elevation_deg": float(el_match.group(1)),
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

    return RoverPose(
        product_id=product_id,
        sol=sol,
        site=parsed["site"],
        drive=parsed["drive"],
        latitude=site_drive.latitude,
        longitude=site_drive.longitude,
        compass_heading_deg=parsed["azimuth_deg"] % 360.0,
        pitch_deg=parsed["elevation_deg"],
    )
