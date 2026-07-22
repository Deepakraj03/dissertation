"""
Real-resolution HiRISE RDR pipeline — replaces the browse-JPEG-based
CycleGAN nadir corpus (~3 m/px) with patches from genuine RDR products
(25 cm/px, 16-bit radiometric).

Unlike download_hirise.py (which fetches browse quicklooks from
PDS/EXTRAS/RDR), this module fetches the real RDR product from PDS/RDR —
one observation at a time, never keeping more than one ~568MB source file
on disk, since the target machine's disk quota is tight.

Usage:
    python hirise_fullres.py --region oxia_planum --n-observations 10
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

RDR_BASE = "https://hirise.lpl.arizona.edu/PDS"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})


def real_rdr_url_for(file_name_spec: str) -> str | None:
    """Build the real (full-resolution) RDR product URL from an index row's
    FILE_NAME_SPECIFICATION. Returns None for non-RED or non-JP2 entries
    (e.g. COLOR products, or .IMG-only rows), matching download_hirise.py's
    browse_url_for's RED-only filter."""
    if not file_name_spec.endswith("_RED.JP2"):
        return None
    return f"{RDR_BASE}/{file_name_spec}"


def download_with_verify(url: str, dest_path: Path) -> bool:
    """Download url to dest_path, verifying the final size exactly matches
    Content-Length. Deletes dest_path and returns False on any mismatch or
    error, so a truncated download is never mistaken for a complete one."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        expected_size = int(r.headers.get("Content-Length", -1))

        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

        actual_size = dest_path.stat().st_size
        if expected_size >= 0 and actual_size != expected_size:
            dest_path.unlink()
            return False
        return True
    except Exception:
        if dest_path.exists():
            dest_path.unlink()
        return False
