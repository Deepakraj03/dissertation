"""Fetch and parse one Mars 2020 (Perseverance) Navcam calibrated product's
real PDS4 XML label to recover the rover's (site, drive) and real camera
boresight pointing at capture time, then combine with
m2020_rover_localization.py's lookup to get a final absolute pose.

Mirrors parse_rover_pose.py's MSL/PDS3 interface (same RoverPose shape,
same fetch-then-join pattern) so downstream code (render_pose_condition_map,
assemble_paired_corpus.py's grouping logic) works unmodified against either
mission -- only the label format and access path differ, not the geometry
this project's rendering pipeline actually needs.

Real access path and label schema verified 2026-08-29 against a real
product (NLB_0001_0667035586_056FDR_N0010052AUT_04096_0A02I3J03, sol 1):
direct file guessing against the msl-style path/bundle combination used by
this project's earlier Jezero exploration (discover_jezero_navcam_archive.py)
failed (redirects/404s) because it guessed the wrong bundle
(mars2020_navcam_ops_raw) and the wrong internal path convention. The real,
working combination is the *_ops_calibrated bundle at the sol/ids/fdr/ncam/
path below.

Label structure: two <geom:Geometry_Lander> blocks share one <geom:Geometry>
element, distinguished by <geom:geometry_state> ("Telemetry", the real
reported geometry we want, vs "Initial", pre-onboard-update and not used
here). The Telemetry block itself carries two <geom:Derived_Geometry>
entries: one in ROVER_NAV_FRAME (rover-relative, no solar geometry) and one
in SITE_FRAME (absolute, carries solar_azimuth/solar_elevation) -- the
SITE_FRAME one is the MSL SITE_DERIVED_GEOMETRY_PARMS equivalent this
project's renderer needs, and is identified here by the presence of
solar_azimuth (only the SITE_FRAME entry has it), not by frame-type text
matching, since testing for the field this code actually consumes is more
robust to schema variation than testing an unrelated descriptive field."""

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from m2020_rover_localization import SiteDrivePose

NAVCAM_LABEL_BASE = (
    "https://planetarydata.jpl.nasa.gov/img/data/mars2020/"
    "mars2020_navcam_ops_calibrated/data/sol"
)

GEOM_NS = "http://pds.nasa.gov/pds4/geom/v1"
NS = {"geom": GEOM_NS}


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
    """product_id must be the FDR_N (non-linearized) full-frame variant's
    stem, e.g. NLB_0001_0667035586_056FDR_N0010052AUT_04096_0A02I3J03 --
    matches the real directory layout's data/sol/{sol:05d}/ids/fdr/ncam/
    path, verified against a real fetched product."""
    return f"{NAVCAM_LABEL_BASE}/{sol:05d}/ids/fdr/ncam/{product_id}.xml"


def parse_navcam_label(label_text: str) -> dict:
    """Extract site, drive, and the real absolute (SITE_FRAME) camera
    boresight azimuth/elevation from a Navcam PDS4 label's XML text. Raises
    ValueError if any required element is missing, so callers can
    distinguish a real parse failure from a network error, same posture as
    parse_rover_pose.py's PDS3 parser."""
    root = ET.fromstring(label_text)

    telemetry_block = None
    for gl in root.iter(f"{{{GEOM_NS}}}Geometry_Lander"):
        state = gl.find("geom:geometry_state", NS)
        if state is not None and state.text == "Telemetry":
            telemetry_block = gl
            break
    if telemetry_block is None:
        raise ValueError("no Telemetry-state Geometry_Lander block found")

    mc = telemetry_block.find("geom:Motion_Counter", NS)
    if mc is None:
        raise ValueError("Motion_Counter not found in Telemetry geometry block")
    indices = {}
    for idx in mc.findall("geom:Motion_Counter_Index", NS):
        iid = idx.find("geom:index_id", NS)
        val = idx.find("geom:index_value_number", NS)
        if iid is not None and val is not None:
            indices[iid.text] = val.text
    if "SITE" not in indices or "DRIVE" not in indices:
        raise ValueError(f"SITE/DRIVE not found in Motion_Counter: {indices}")
    site, drive = int(indices["SITE"]), int(indices["DRIVE"])

    site_frame_geom = None
    for dg in telemetry_block.findall("geom:Derived_Geometry", NS):
        if dg.find("geom:solar_azimuth", NS) is not None:
            site_frame_geom = dg
            break
    if site_frame_geom is None:
        raise ValueError(
            "no SITE_FRAME Derived_Geometry (identified by solar_azimuth "
            "presence) found in Telemetry geometry block")

    az = site_frame_geom.find("geom:instrument_azimuth", NS)
    el = site_frame_geom.find("geom:instrument_elevation", NS)
    if az is None or el is None:
        raise ValueError("instrument_azimuth/elevation not found in SITE_FRAME Derived_Geometry")

    return {
        "site": site,
        "drive": drive,
        "azimuth_deg": float(az.text),
        "elevation_deg": float(el.text),
    }


def fetch_and_parse_pose(product_id: str, sol: int,
                         localization: dict[tuple[int, int], SiteDrivePose],
                         ) -> "RoverPose | None":
    """Fetch product_id's label, parse it, and join against localization by
    (site, drive). Returns None (never raises) on any failure -- fetch
    error, unparseable label, or a (site, drive) not present in
    localization -- mirroring parse_rover_pose.py's skip-on-failure
    posture for bulk candidate processing."""
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
