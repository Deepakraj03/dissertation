"""Query the PDS Orbital Data Explorer (ODE) REST API for HiRISE stereo
DTM coverage over a lat/lon region.

Verified endpoint (2026-08-02):
https://oderest.rsl.wustl.edu/live2/?target=mars&ihid=MRO&iid=HIRISE&pt=DTM
&output=JSON&results=m&minlat=..&maxlat=..&westlon=..&eastlon=..
Longitudes are 0-360 East (same convention as RDRCUMINDEX, opposite of
download_hirise.py's REGIONS which uses signed -180..180 — callers must
convert, see query_dtm_coverage).
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from download_hirise import REGIONS

ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2/"


@dataclass
class DtmCoverageRecord:
    product_id: str
    dtm_url: str
    obs_id_a: str
    obs_id_b: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    comment: str
    files_url: str
    footprint_wkt: str = ""


def build_ode_dtm_query_url(min_lat: float, max_lat: float,
                            west_lon_360: float, east_lon_360: float) -> str:
    params = (
        f"target=mars&ihid=MRO&iid=HIRISE&pt=DTM&output=JSON&results=m"
        f"&minlat={min_lat}&maxlat={max_lat}"
        f"&westlon={west_lon_360}&eastlon={east_lon_360}"
    )
    return f"{ODE_BASE_URL}?{params}"


def _extract_obs_ids(product: dict) -> tuple[str, str]:
    """Extract the two source observation IDs from a product's ODE_notes
    'Index Record' line, which quotes them as the 8th/9th CSV fields."""
    notes = product.get("ODE_notes", {}).get("ODE_note", [])
    for note in notes:
        if "Index Record" in note or note.strip().startswith('"'):
            quoted = re.findall(r'"([^"]*)"', note)
            # Index Record fields: [0]=archive, [1]=path, [2]=host, [3]=inst,
            # [4]=product_id, [5]=version, [6]=target, [7]=comment,
            # [8]=obs_id_a, [9]=obs_id_b, [10]=..., [11]=product_type
            if len(quoted) >= 10:
                return quoted[8], quoted[9]
    raise ValueError(f"could not find obs ID pair in ODE_notes: {notes}")


def parse_ode_response(response_json: dict) -> list[DtmCoverageRecord]:
    results = response_json.get("ODEResults", {})
    products_container = results.get("Products")
    if not products_container:
        return []
    product = products_container.get("Product", [])
    # ODE collapses a single result to a bare dict instead of a one-item list.
    if isinstance(product, dict):
        products = [product]
    else:
        products = product

    records = []
    for p in products:
        obs_a, obs_b = _extract_obs_ids(p)
        records.append(DtmCoverageRecord(
            product_id=p["LabelURL"].rsplit("/", 1)[-1].removesuffix(".IMG"),
            dtm_url=p["LabelURL"],
            obs_id_a=obs_a,
            obs_id_b=obs_b,
            min_lat=float(p["Minimum_latitude"]),
            max_lat=float(p["Maximum_latitude"]),
            min_lon=float(p["Westernmost_longitude"]),
            max_lon=float(p["Easternmost_longitude"]),
            comment=p.get("Comment", ""),
            files_url=p["FilesURL"],
            footprint_wkt=p.get("Footprint_geometry", ""),
        ))
    return records


def query_dtm_coverage(min_lat: float, max_lat: float,
                       west_lon_360: float, east_lon_360: float) -> list[DtmCoverageRecord]:
    url = build_ode_dtm_query_url(min_lat, max_lat, west_lon_360, east_lon_360)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return parse_ode_response(r.json())


def _query_band_recursive(min_lat: float, max_lat: float, query_fn,
                          min_band_deg: float) -> list[DtmCoverageRecord]:
    records = query_fn(min_lat, max_lat, 0.0, 360.0)
    if len(records) < 100 or (max_lat - min_lat) <= min_band_deg:
        return records
    mid = (min_lat + max_lat) / 2
    return (_query_band_recursive(min_lat, mid, query_fn, min_band_deg)
           + _query_band_recursive(mid, max_lat, query_fn, min_band_deg))


def query_global_dtm_coverage(band_deg: float = 10.0, min_band_deg: float = 0.5,
                              query_fn=query_dtm_coverage) -> list[DtmCoverageRecord]:
    """Full global HiRISE stereo DTM catalog, working around the ODE API's
    100-result-per-query cap. Verified 2026-08-14: a single global query
    truncates at 100; even 10-degree latitude bands still cap near the
    equator (Mars's most HiRISE-imaged latitudes). Recursively halves any
    band that comes back at exactly 100 (the cap signature) down to
    min_band_deg, rather than hardcoding a fixed split depth that could
    silently under-cover a denser future catalog. Full 0-360 longitude is
    queried in every band -- splitting was only ever needed on latitude in
    manual testing. Deduplicates by product_id, since a DTM footprint
    straddling a band boundary is returned by both adjacent queries."""
    all_records: list[DtmCoverageRecord] = []
    lat = -90.0
    while lat < 90.0:
        band_max = min(lat + band_deg, 90.0)
        all_records.extend(_query_band_recursive(lat, band_max, query_fn, min_band_deg))
        lat = band_max

    seen: set[str] = set()
    deduped = []
    for r in all_records:
        if r.product_id not in seen:
            seen.add(r.product_id)
            deduped.append(r)
    return deduped


def signed_lon_to_0_360(lon: float) -> float:
    """Convert -180..180 signed longitude to 0-360 East, matching ODE's
    convention (inverse of hirise_index.py's _to_signed_lon)."""
    return lon + 360.0 if lon < 0.0 else lon


def build_coverage_report(region_name: str, lat_min: float, lat_max: float,
                          lon_min: float, lon_max: float,
                          records: list[DtmCoverageRecord]) -> dict:
    west = signed_lon_to_0_360(lon_min)
    east = signed_lon_to_0_360(lon_max)
    return {
        "region": region_name,
        "query_bounds": {
            "min_lat": lat_min, "max_lat": lat_max,
            "west_lon_360": west, "east_lon_360": east,
        },
        "count": len(records),
        "records": [asdict(r) for r in records],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="oxia_planum", choices=list(REGIONS.keys()))
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: Data/HiRISE_index/dtm_coverage_<region>.json)")
    args = parser.parse_args()

    cfg = REGIONS[args.region]
    west = signed_lon_to_0_360(cfg["lon_min"])
    east = signed_lon_to_0_360(cfg["lon_max"])
    print(f"Querying ODE for {args.region} DTM coverage "
          f"(lat {cfg['lat_min']}-{cfg['lat_max']}, lon0-360 {west}-{east})…")
    records = query_dtm_coverage(cfg["lat_min"], cfg["lat_max"], west, east)
    print(f"Found {len(records)} DTM product(s)")

    report = build_coverage_report(args.region, cfg["lat_min"], cfg["lat_max"],
                                   cfg["lon_min"], cfg["lon_max"], records)
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent.parent / "Data" / "HiRISE_index"
        / f"dtm_coverage_{args.region}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
