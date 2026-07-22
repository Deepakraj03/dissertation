"""
HiRISE nadir image downloader — selects observations by real geographic
footprint (from the PDS RDR cumulative index) rather than the observation
ID's trailing digits.

BUG FIX (see project history): the original version filtered candidate
observations by crawling PDS directory listings and parsing a "lat code"
from the last 4 digits of the observation ID, assuming
LLLL = round((lat_deg + 90) * 10). That assumption is wrong for real
HiRISE observation IDs — an audit of the 304 previously-downloaded images
found 0/299 were actually within their claimed region; all were
south-polar (real lat -87 to -68) regardless of which of the 4 target
regions was requested. It also never constrained longitude at all, so
even a correct lat-code would have matched anywhere in that latitude
band globally. This version filters using each observation's real
lat/lon footprint from RDRCUMINDEX.TAB, with real region bounding boxes
sourced from published landing-site/feature coordinates.

Downloads RED-channel browse JPEGs (2048x~6000 px, ~5 m/px effective
resolution — this is a downsampled preview, NOT full-resolution HiRISE;
see Data/CTX_HiRISE_pairs work for genuine high-resolution sourcing).

Usage:
    python download_hirise.py                  # Oxia Planum, 300 images
    python download_hirise.py --region all      # all 4 regions
    python download_hirise.py --max 100 --workers 8
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, str(Path(__file__).parent))
from hirise_index import Footprint, download_index, load_index

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "Data" / "HiRISE" / "raw"
INDEX_DIR   = Path(__file__).parent.parent / "Data" / "HiRISE_index"
BROWSE_BASE = "https://hirise.lpl.arizona.edu/PDS/EXTRAS/RDR"

# Real regional bounding boxes (lat/lon in degrees, longitude signed
# -180..180), sourced from published landing-site/feature coordinates:
#   - Oxia Planum: ExoMars Rosalind Franklin landing site, ~18.2N 335.45E,
#     region commonly described as 16-24N, 335-345E.
#   - Jezero Crater: Mars 2020 Perseverance landing site, ~18.38N 77.58E.
#   - Gale Crater: Curiosity landing site, ~-4.5S 137.4E.
#   - Hellas Basin: basin centre ~-42.7S 70E (huge feature; box below
#     covers a representative sub-region, not the whole ~2300km basin).
REGIONS = {
    "oxia_planum": {
        "lat_min": 16.0, "lat_max": 24.0,
        "lon_min": -25.0, "lon_max": -15.0,
        "description": "ExoMars RFM primary landing site",
    },
    "jezero_crater": {
        "lat_min": 16.0, "lat_max": 21.0,
        "lon_min": 75.0, "lon_max": 80.0,
        "description": "Mars 2020 Perseverance landing site",
    },
    "gale_crater": {
        "lat_min": -8.0, "lat_max": -3.0,
        "lon_min": 135.0, "lon_max": 140.0,
        "description": "Curiosity rover area (diverse terrain)",
    },
    "hellas_basin": {
        "lat_min": -45.0, "lat_max": -35.0,
        "lon_min": 60.0, "lon_max": 80.0,
        "description": "Diverse ancient terrain",
    },
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})


def footprint_in_region(fp: Footprint, region: dict) -> bool:
    """True if the footprint's centre point falls within the region's
    real lat/lon bounding box."""
    center_lat = (fp.min_lat + fp.max_lat) / 2
    center_lon = (fp.min_lon + fp.max_lon) / 2
    return (region["lat_min"] <= center_lat <= region["lat_max"]
            and region["lon_min"] <= center_lon <= region["lon_max"])


def browse_url_for(file_name_spec: str) -> str | None:
    """Build the browse JPEG URL from an index row's
    FILE_NAME_SPECIFICATION, e.g.
    'RDR/ESP/ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_RED.JP2'
    -> '.../ESP_039912_1095_RED.browse.jpg'. Returns None for non-RED-JP2
    entries (e.g. COLOR products), which have no matching browse image
    under this path pattern."""
    if not file_name_spec.endswith(".JP2"):
        return None
    rel_path = file_name_spec[len("RDR/"):-len(".JP2")] + ".browse.jpg"
    return f"{BROWSE_BASE}/{rel_path}"


def download_image(url: str, out_path: Path) -> bool:
    """Download a single browse JPEG. Returns True on success."""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True
    try:
        r = SESSION.get(url, timeout=120, stream=True)
        r.raise_for_status()
        content = r.content
        if not content[:3] == b"\xff\xd8\xff":
            return False
        out_path.write_bytes(content)
        return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
        return False


def collect_targets(regions_to_query: list[tuple[str, dict]],
                    index: dict[str, Footprint],
                    max_per_region: int) -> list[tuple[str, Path, str]]:
    """
    Select observations whose real footprint (from the index) falls
    within each requested region, and build (url, out_path, obs_id) for
    each. Returns a deduplicated list capped at max_per_region per
    region.
    """
    tasks: list[tuple[str, Path, str]] = []
    already_seen: set[str] = set()

    for region_name, region_cfg in regions_to_query:
        region_dir = DATA_DIR / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Searching {region_name} "
              f"(lat {region_cfg['lat_min']}-{region_cfg['lat_max']}, "
              f"lon {region_cfg['lon_min']}-{region_cfg['lon_max']})…")

        region_tasks: list[tuple[str, Path, str]] = []
        for obs_id, fp in index.items():
            if len(region_tasks) >= max_per_region:
                break
            if obs_id in already_seen:
                continue
            if not footprint_in_region(fp, region_cfg):
                continue
            browse_url = browse_url_for(fp.file_name_spec)
            if browse_url is None:
                continue
            already_seen.add(obs_id)
            out_path = region_dir / f"{obs_id}_RED.browse.jpg"
            region_tasks.append((browse_url, out_path, obs_id))

        print(f"    Found {len(region_tasks)} candidate images")
        tasks.extend(region_tasks)

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="oxia_planum",
                        choices=list(REGIONS.keys()) + ["all"],
                        help="Target geographic region")
    parser.add_argument("--max",     type=int, default=300,
                        help="Max images per region")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel download threads")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []

    print("Loading HiRISE footprint index (downloads ~166MB if not cached)…")
    index_path = download_index(INDEX_DIR)
    index = load_index(index_path)
    print(f"Index loaded: {len(index)} observations")

    regions_to_query = (
        list(REGIONS.items()) if args.region == "all"
        else [(args.region, REGIONS[args.region])]
    )

    print("Selecting HiRISE observation targets…")
    tasks = collect_targets(regions_to_query, index, max_per_region=args.max)
    print(f"\nTotal targets: {len(tasks)}")

    already_paths = {m["path"] for m in manifest}
    tasks = [(u, p, o) for u, p, o in tasks if str(p) not in already_paths]
    print(f"New downloads:  {len(tasks)}")

    if not tasks:
        print("Nothing new to download.")
        return

    print(f"Downloading with {args.workers} workers…\n")
    downloaded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_image, url, path): (obs_id, path)
            for url, path, obs_id in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            obs_id, path = futures[fut]
            ok = fut.result()
            if ok:
                downloaded += 1
                sz_kb = path.stat().st_size // 1024 if path.exists() else 0
                manifest.append({
                    "obs_id": obs_id,
                    "path":   str(path),
                    "size_kb": sz_kb,
                })
            else:
                failed += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} — {downloaded} ok, {failed} failed")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {downloaded} downloaded, {failed} failed.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
