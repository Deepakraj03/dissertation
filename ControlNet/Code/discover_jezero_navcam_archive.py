"""Investigate real Mars 2020 (Jezero Crater) Navcam archive access and
label format -- does NOT build a working ingestion path. See
docs/superpowers/plans/2026-08-13-multi-site-data-expansion.md Task 7 for
the real findings this builds on (direct planetarydata.jpl.nasa.gov file
fetches 302-redirect to a landing page; a real PLACES-equivalent
localization product exists at pds-geosciences.wustl.edu). Produces a JSON
report a follow-up plan would need before designing real Jezero ingestion.
"""

import json
import sys
from pathlib import Path

import requests

BROWSER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# Real candidate roots found during 2026-08-13 planning research -- see
# this module's docstring. Each is tried against a real known product
# path; the goal is finding which (if any) serves real file content
# rather than a redirect or 404.
CANDIDATE_FILE_ROOTS = [
    "https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_navcam_ops_raw",
    "https://pdsimage2.wr.usgs.gov/archive/mars2020/mars2020_navcam_ops_raw",
]
SAMPLE_PRODUCT_PATH = "data/sol/00001/ids/edr/NLB_0001_0667035586_056EDR_N0010052AUT_04096_00_2I3J03"
PLACES_CANDIDATE_URL = (
    "https://pds-geosciences.wustl.edu/m2020/"
    "urn-nasa-pds-mars2020_rover_places/data_localizations/"
)
GEOMETRY_KEYWORDS = ["solar", "elevation", "azimuth", "geometry", "pointing"]


def classify_fetch_result(response) -> str:
    """real_content: looks like an actual PDS3/PDS4 label, not a redirect
    landing page. html_not_data: 200 but generic HTML (the exact failure
    mode found for planetarydata.jpl.nasa.gov's individual M2020 files
    during planning). http_error: non-200 status."""
    if response.status_code != 200:
        return "http_error"
    text_start = response.text[:200].lstrip()
    if text_start.startswith("PDS_VERSION_ID") or text_start.startswith("<?xml"):
        return "real_content"
    return "html_not_data"


def probe_file_roots(roots: list[str], sample_path: str,
                     session: "requests.Session") -> dict:
    results = {}
    for root in roots:
        for ext in (".xml", ".IMG"):
            url = f"{root}/{sample_path}{ext}"
            try:
                resp = session.get(url, timeout=30, allow_redirects=True)
                results[url] = classify_fetch_result(resp)
            except Exception as e:
                results[url] = f"exception: {e}"
    return results


def probe_places_localization(url: str, session: "requests.Session") -> dict:
    try:
        resp = session.get(url, timeout=30)
        return {"url": url, "status_code": resp.status_code,
               "content_length": len(resp.text),
               "listing_sample": resp.text[:500]}
    except Exception as e:
        return {"url": url, "exception": str(e)}


def find_geometry_keywords(label_text: str) -> list[str]:
    lower = label_text.lower()
    return [kw for kw in GEOMETRY_KEYWORDS if kw in lower]


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_USER_AGENT})

    print("Probing candidate M2020 Navcam file roots…")
    file_root_results = probe_file_roots(CANDIDATE_FILE_ROOTS, SAMPLE_PRODUCT_PATH, session)
    for url, verdict in file_root_results.items():
        print(f"  {verdict}: {url}")

    print("\nProbing PLACES localization product…")
    places_result = probe_places_localization(PLACES_CANDIDATE_URL, session)
    print(f"  {places_result}")

    real_content_urls = [url for url, v in file_root_results.items() if v == "real_content"]
    geometry_keywords_found = []
    if real_content_urls:
        label_text = session.get(real_content_urls[0], timeout=30).text
        geometry_keywords_found = find_geometry_keywords(label_text)
        print(f"\nReal label fetched from {real_content_urls[0]}")
        print(f"Geometry-related keywords found: {geometry_keywords_found}")
    else:
        print("\nNo candidate root served real file content -- "
             "geometry field presence could not be checked this run.")

    report = {
        "file_root_probe": file_root_results,
        "places_probe": places_result,
        "real_content_urls": real_content_urls,
        "geometry_keywords_found": geometry_keywords_found,
        "viable_for_ingestion": bool(real_content_urls) and bool(geometry_keywords_found),
    }

    root = Path(__file__).parent.parent
    out_path = root / "Data" / "HiRISE_index" / "jezero_archive_discovery_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
