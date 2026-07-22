"""
Co-register existing downloaded HiRISE browse images with matching
low-resolution crops from the Murray Lab Global CTX Mosaic.

Produces paired (lr.tif, hr.jpg) crops under Data/CTX_HiRISE_pairs/{obs_id}/
for the DiffusionSat fine-tune data pipeline (Part 2+, not built here),
plus manifest.json / skipped.json recording what succeeded and why
anything was skipped.

Usage:
    python coregister_ctx_hirise.py --smoke-test 5   # first 5 images only
    python coregister_ctx_hirise.py                  # full batch
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).parent))
from hirise_index import Footprint, download_index, load_index
from ctx_mosaic import download_tile, tiles_for_bbox

ROOT        = Path(__file__).parent.parent
HIRISE_RAW  = ROOT / "Data" / "HiRISE" / "raw"
INDEX_DIR   = ROOT / "Data" / "HiRISE_index"
TILE_DIR    = ROOT / "Data" / "CTX_tiles"
PAIRS_DIR   = ROOT / "Data" / "CTX_HiRISE_pairs"

OBS_ID_RE = re.compile(r"(ESP|PSP)_\d{6}_\d{4}")

# Murray Lab CTX mosaic tiles use a projected Mars equirectangular CRS in
# metres (Mars_2015_Ocentric_Equirectangular_clon_0, standard_parallel_1=0,
# central_meridian=0), not raw lon/lat degrees. This radius is the tile's
# own SPHEROID semi-major axis (confirmed via rasterio on a real
# downloaded tile during planning). With standard_parallel_1=0 the
# projection reduces to this simple form — verified empirically against
# the real tile's reported bounds (see test_lonlat_to_projected_matches_real_tile_bounds).
MARS_RADIUS_M = 3396190.0


def lonlat_to_projected(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Convert planetocentric lon/lat (degrees) to the CTX mosaic's
    projected equirectangular CRS (metres)."""
    x = math.radians(lon_deg) * MARS_RADIUS_M
    y = math.radians(lat_deg) * MARS_RADIUS_M
    return x, y


def find_downloaded_images(hirise_raw_dir: Path) -> list[tuple[str, Path]]:
    """Return (obs_id, path) for every downloaded HiRISE browse JPEG that
    still exists on disk, read from the single flat manifest.json that
    download_hirise.py writes directly in hirise_raw_dir (verified
    against Code/download_hirise.py's DATA_DIR/manifest_path logic —
    there is no per-region manifest)."""
    results: list[tuple[str, Path]] = []
    manifest_path = hirise_raw_dir / "manifest.json"
    if not manifest_path.exists():
        return results
    entries = json.loads(manifest_path.read_text())
    for entry in entries:
        path = Path(entry["path"])
        if not path.exists():
            continue
        match = OBS_ID_RE.search(entry["obs_id"])
        if match:
            results.append((match.group(0), path))
    return results


def crop_ctx_to_footprint(tif_path: Path, footprint: Footprint,
                          out_path: Path) -> None:
    """Crop a CTX mosaic GeoTIFF to a footprint's lat/lon bounding box.

    The footprint's geographic bounds are converted to the tile's
    projected CRS (see lonlat_to_projected) before computing the crop
    window. Raises ValueError if the resulting window is not fully
    contained within the tile's pixel bounds — this should not happen
    for footprints that passed the tiles_for_bbox single-tile check, so
    a failure here indicates a real problem (e.g. a footprint that
    only *appears* single-tile due to floating-point edge effects) and
    must not be allowed to silently produce a truncated crop.
    """
    min_x, min_y = lonlat_to_projected(footprint.min_lon, footprint.min_lat)
    max_x, max_y = lonlat_to_projected(footprint.max_lon, footprint.max_lat)

    with rasterio.open(tif_path) as src:
        window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)

        if (window.col_off < 0 or window.row_off < 0
                or window.col_off + window.width > src.width
                or window.row_off + window.height > src.height):
            raise ValueError(
                f"crop window {window} falls outside tile bounds "
                f"({src.width}x{src.height}) for {footprint.obs_id}"
            )

        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)


def process_one(obs_id: str, hirise_path: Path, index: dict[str, Footprint],
                tile_dir: Path, pairs_dir: Path) -> tuple[bool, str]:
    """Attempt to produce one co-registered (lr.tif, hr.jpg) pair.
    Returns (success, status_string)."""
    footprint = index.get(obs_id)
    if footprint is None:
        return False, "no_index_match"

    tiles = tiles_for_bbox(footprint.min_lon, footprint.max_lon,
                           footprint.min_lat, footprint.max_lat)
    if len(tiles) > 1:
        return False, "spans_multiple_tiles"

    try:
        tif_path = download_tile(tiles[0], tile_dir)
    except Exception as e:
        return False, f"tile_download_failed: {e}"

    pair_dir = pairs_dir / obs_id
    try:
        crop_ctx_to_footprint(tif_path, footprint, pair_dir / "lr.tif")
    except Exception as e:
        return False, f"crop_failed: {e}"

    pair_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(hirise_path, pair_dir / "hr.jpg")
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", type=int, default=None,
                        help="Only process the first N images (visual QA run)")
    args = parser.parse_args()

    print("Loading HiRISE footprint index (downloads ~166MB if not cached)…")
    index_path = download_index(INDEX_DIR)
    index = load_index(index_path)
    print(f"Index loaded: {len(index)} observations")

    images = find_downloaded_images(HIRISE_RAW)
    if args.smoke_test:
        images = images[:args.smoke_test]
    print(f"Processing {len(images)} downloaded HiRISE images…")

    manifest, skipped = [], []
    for obs_id, hirise_path in images:
        ok, status = process_one(obs_id, hirise_path, index, TILE_DIR, PAIRS_DIR)
        if ok:
            manifest.append({
                "obs_id": obs_id,
                "lr": str(PAIRS_DIR / obs_id / "lr.tif"),
                "hr": str(PAIRS_DIR / obs_id / "hr.jpg"),
            })
        else:
            skipped.append({"obs_id": obs_id, "reason": status})
        print(f"  {obs_id}: {status}")

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    (PAIRS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (PAIRS_DIR / "skipped.json").write_text(json.dumps(skipped, indent=2))
    print(f"\nDone. {len(manifest)} pairs created, {len(skipped)} skipped.")
    print(f"Manifest: {PAIRS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
