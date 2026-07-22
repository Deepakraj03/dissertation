"""
Resolve and download tiles from the Murray Lab Global CTX Mosaic of Mars.

Tile grid verified against the live directory listing at
https://murray-lab.caltech.edu/CTX/V01/tiles/ — tiles are named
MurrayLab_GlobalCTXMosaic_V01_E{lon}_N{lat}.zip on a 4-degree grid, where
{lon} is the signed cell longitude zero-padded to 3 digits (e.g. "-060",
"004", "000") and {lat} to 2 digits (e.g. "-56", "08", "00"), identifying
the cell's lower-left (southwest) corner.
"""

import math
import shutil
import zipfile
from pathlib import Path

import requests
import urllib3

TILE_BASE_URL = "https://murray-lab.caltech.edu/CTX/V01/tiles"
TILE_STEP_DEG = 4

# murray-lab.caltech.edu serves a legitimate certificate (InCommon RSA OV
# SSL CA 3, correctly matching murray-web.gps.caltech.edu, valid at time
# of writing) but does not send its intermediate certificate in the TLS
# handshake, so verification fails with "unable to get local issuer
# certificate" regardless of client. Confirmed via manual `openssl
# s_client` inspection during debugging — this is a server
# misconfiguration, not a spoofed/self-signed cert. Verification is
# disabled only for requests to this one domain, which serves public,
# non-sensitive scientific imagery (no credentials ever sent here).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _tile_component(value: int, width: int) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):0{width}d}"


def tile_name_for(lon: float, lat: float) -> str:
    """Return the Murray Lab tile filename (no extension) for the
    4-degree cell containing the given signed (-180..180) lon/lat."""
    cell_lon = math.floor(lon / TILE_STEP_DEG) * TILE_STEP_DEG
    cell_lat = math.floor(lat / TILE_STEP_DEG) * TILE_STEP_DEG
    lon_str = _tile_component(cell_lon, 3)
    lat_str = _tile_component(cell_lat, 2)
    return f"MurrayLab_GlobalCTXMosaic_V01_E{lon_str}_N{lat_str}"


def tiles_for_bbox(min_lon: float, max_lon: float,
                    min_lat: float, max_lat: float) -> list[str]:
    """Distinct tile names covering a bounding box. Callers must skip
    footprints where len(result) > 1 (out of scope for this pass)."""
    names = {
        tile_name_for(lon, lat)
        for lon in (min_lon, max_lon)
        for lat in (min_lat, max_lat)
    }
    return sorted(names)


def _download_with_one_retry(url: str, dest_path: Path) -> None:
    """Download url to dest_path. On failure, retry exactly once before
    letting the exception propagate (spec: "retry once, then skip and
    log" — the skip/log happens in the caller, process_one).

    Downloads to a .part temp file and only renames it to dest_path once
    the stream completes without error, so an interrupted download never
    leaves a truncated file at dest_path that a later run's "already
    downloaded" check (in download_tile) would mistake for a complete
    file — same class of bug as hirise_index.download_index, fixed the
    same way."""
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            r = requests.get(
                url, timeout=600, stream=True, verify=False,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            )
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp_path.rename(dest_path)
            return
        except Exception as e:
            last_error = e
    raise last_error


def download_tile(tile_name: str, dest_dir: Path) -> Path:
    """Download and extract a tile zip if not already cached locally.
    Returns the path to the extracted .tif."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tif_path = dest_dir / f"{tile_name}.tif"
    if tif_path.exists():
        return tif_path

    zip_path = dest_dir / f"{tile_name}.zip"
    if not zip_path.exists():
        url = f"{TILE_BASE_URL}/{tile_name}.zip"
        _download_with_one_retry(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        tif_members = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        if not tif_members:
            raise FileNotFoundError(f"No .tif found inside {zip_path}")
        # Stream in chunks rather than .read() the whole ~2.25GB file into
        # memory at once (this machine has only 7GB RAM).
        with zf.open(tif_members[0]) as src, open(tif_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 24)

    # Each tile is ~1.7GB zipped + ~2.25GB extracted; delete the zip once
    # the tif is safely extracted so tiles don't cost ~4GB each forever.
    zip_path.unlink()

    return tif_path
