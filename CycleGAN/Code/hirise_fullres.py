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

sys.path.insert(0, str(Path(__file__).parent))

RDR_BASE = "https://hirise.lpl.arizona.edu/PDS"


def real_rdr_url_for(file_name_spec: str) -> str | None:
    """Build the real (full-resolution) RDR product URL from an index row's
    FILE_NAME_SPECIFICATION. Returns None for non-RED or non-JP2 entries
    (e.g. COLOR products, or .IMG-only rows), matching download_hirise.py's
    browse_url_for's RED-only filter."""
    if not file_name_spec.endswith("_RED.JP2"):
        return None
    return f"{RDR_BASE}/{file_name_spec}"
