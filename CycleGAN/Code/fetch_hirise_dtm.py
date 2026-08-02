"""Download a HiRISE stereo DTM raster and its companion orthorectified
images (already co-registered to the DTM by HiRISE's own photogrammetric
pipeline — no separate co-registration step needed). One product's files
on disk at a time, same disk-quota-aware lifecycle as hirise_fullres.py."""

import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from hirise_fullres import download_with_verify
from dtm_coverage import DtmCoverageRecord

OBS_ID_PATTERN = re.compile(r"[A-Z]{3}_\d{6}_\d{4}")


def parse_productfiles_html(html: str) -> dict[str, list[str]]:
    """Map each observation ID found in ORTHO.JP2 links to its list of
    ortho URLs. A DTM's productfiles page lists an A and C variant per
    source observation; both are kept, caller picks one (prefer 'A')."""
    ortho_urls = re.findall(r'href="([^"]+_ORTHO\.JP2)"', html)
    by_obs: dict[str, list[str]] = {}
    for url in ortho_urls:
        filename = url.rsplit("/", 1)[-1]
        match = OBS_ID_PATTERN.search(filename)
        if match:
            by_obs.setdefault(match.group(0), []).append(url)
    return by_obs


def fetch_dtm_and_orthos(record: DtmCoverageRecord, scratch_dir: Path,
                         dest_dir: Path) -> dict:
    """Download record's DTM .IMG and one ortho .JP2 per source
    observation (preferring the '_A_' variant) into dest_dir, deleting
    scratch copies as soon as each file is verified. Returns a status dict
    matching hirise_fullres.process_observation's shape."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    dtm_dest = dest_dir / f"{record.product_id}.IMG"
    if not download_with_verify(record.dtm_url, dtm_dest):
        return {"product_id": record.product_id, "status": "dtm_download_failed"}

    try:
        r = requests.get(record.files_url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return {"product_id": record.product_id, "status": f"productfiles_fetch_failed: {e}"}

    by_obs = parse_productfiles_html(r.text)
    ortho_paths = {}
    for obs_id in (record.obs_id_a, record.obs_id_b):
        candidates = by_obs.get(obs_id, [])
        preferred = next((u for u in candidates if "_A_" in u), None) or (
            candidates[0] if candidates else None
        )
        if preferred is None:
            return {"product_id": record.product_id,
                    "status": f"no_ortho_found_for_{obs_id}"}
        ortho_dest = dest_dir / f"{obs_id}_ORTHO.JP2"
        if not download_with_verify(preferred, ortho_dest):
            return {"product_id": record.product_id,
                    "status": f"ortho_download_failed_for_{obs_id}"}
        ortho_paths[obs_id] = str(ortho_dest)

    return {
        "product_id": record.product_id,
        "status": "ok",
        "dtm_path": str(dtm_dest),
        "ortho_paths": ortho_paths,
    }
