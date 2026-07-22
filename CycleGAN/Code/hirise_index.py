"""
Parse the HiRISE RDR cumulative index for per-observation geographic
footprints.

Index source: https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB
Column layout verified against RDRCUMINDEX.LBL (54 comma-separated,
quoted columns; despite the .LBL declaring fixed byte offsets, the actual
file is CSV).
"""

from dataclasses import dataclass
from pathlib import Path

import csv
import requests

INDEX_URL = "https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB"

# Zero-based column indices, verified against RDRCUMINDEX.LBL.
COL_FILE_NAME_SPEC  = 1
COL_OBSERVATION_ID  = 4
COL_MIN_LATITUDE    = 35
COL_MAX_LATITUDE    = 36
COL_MIN_LONGITUDE   = 37
COL_MAX_LONGITUDE   = 38
COL_MAP_PROJECTION  = 41
EXPECTED_COLUMNS    = 54


@dataclass
class Footprint:
    obs_id: str
    min_lat: float
    max_lat: float
    min_lon: float  # normalised to [-180, 180]
    max_lon: float  # normalised to [-180, 180]
    projection: str
    file_name_spec: str  # e.g. "RDR/ESP/ORB_.../OBS_ID/OBS_ID_RED.JP2"


def _to_signed_lon(lon_0_360: float) -> float:
    """Convert 0-360 East longitude (index convention) to -180..180
    (Murray Lab tile-grid convention)."""
    return lon_0_360 - 360.0 if lon_0_360 > 180.0 else lon_0_360


def parse_index_row(fields: list[str]) -> Footprint:
    return Footprint(
        obs_id=fields[COL_OBSERVATION_ID].strip(),
        min_lat=float(fields[COL_MIN_LATITUDE]),
        max_lat=float(fields[COL_MAX_LATITUDE]),
        min_lon=_to_signed_lon(float(fields[COL_MIN_LONGITUDE])),
        max_lon=_to_signed_lon(float(fields[COL_MAX_LONGITUDE])),
        projection=fields[COL_MAP_PROJECTION].strip(),
        file_name_spec=fields[COL_FILE_NAME_SPEC].strip(),
    )


def load_index(tab_path: Path) -> dict[str, Footprint]:
    """Parse the full cumulative index into obs_id -> Footprint."""
    index: dict[str, Footprint] = {}
    with open(tab_path, newline="", encoding="latin-1") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if len(row) < EXPECTED_COLUMNS:
                continue
            fp = parse_index_row(row)
            index[fp.obs_id] = fp
    return index


def download_index(dest_dir: Path) -> Path:
    """Download RDRCUMINDEX.TAB if not already cached. Returns its path.

    Downloads to a .part temp file and only renames it to the final path
    once the stream completes without error, so an interrupted download
    (e.g. a mid-stream connection reset) never leaves a truncated file
    that a later run's "already downloaded" check would mistake for a
    complete index."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "RDRCUMINDEX.TAB"
    if dest_path.exists() and dest_path.stat().st_size > 10_000_000:
        return dest_path

    tmp_path = dest_dir / "RDRCUMINDEX.TAB.part"
    r = requests.get(
        INDEX_URL, timeout=300, stream=True,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp_path.rename(dest_path)
    return dest_path
