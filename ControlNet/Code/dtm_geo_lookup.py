"""Geometry glue between a real lat/lon (from parse_rover_pose.py) and a
specific HiRISE DTM's pixel grid, plus converting a compass heading into
render_ground_view.py's own heading_deg convention."""

import math
import sys
from pathlib import Path

import pyproj
import rasterio

# dtm_coverage.py is shared infrastructure that stays in CycleGAN/Code
# (also used by CycleGAN's own assemble_geometry_corpus.py) -- bridge to
# it rather than duplicating it into ControlNet/Code.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CycleGAN" / "Code"))
from dtm_coverage import DtmCoverageRecord


def _parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    """Parse ODE's 'POLYGON ((lon lat, lon lat, ...))' format into a list
    of (lon, lat) vertices. No shapely dependency — this codebase keeps a
    minimal dependency set (see the plan's Tech Stack), and ODE's polygon
    format is simple enough to parse directly."""
    coords_str = wkt.split("((")[1].split("))")[0]
    points = []
    for pair in coords_str.split(","):
        lon_str, lat_str = pair.strip().split()
        points.append((float(lon_str), float(lat_str)))
    return points


def _point_in_polygon(lon: float, lat: float,
                      polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test (odd-even rule)."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (
                lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def find_covering_dtm(lat: float, lon_signed: float,
                      records: list[DtmCoverageRecord]) -> DtmCoverageRecord | None:
    """Which record (if any) covers this lat/lon. Prefers a real
    point-in-polygon test against footprint_wkt (ODE's real DTM footprint —
    found 2026-08-09 to be a narrow rotated strip, much smaller than its own
    axis-aligned bounding box; a bbox-only test wrongly accepted real rover
    positions sitting in the bbox's empty corners, outside the actual strip).
    Falls back to the bbox test only when footprint_wkt is missing.
    Assumes records' min_lon/max_lon (and footprint_wkt's own longitudes)
    are already directly comparable to lon_signed — true for Gale Crater
    (~135-140E, no dateline crossing), see this plan's Global Constraints
    for why that's safe here specifically."""
    for record in records:
        if record.footprint_wkt:
            polygon = _parse_wkt_polygon(record.footprint_wkt)
            if _point_in_polygon(lon_signed, lat, polygon):
                return record
        elif (record.min_lat <= lat <= record.max_lat and
                record.min_lon <= lon_signed <= record.max_lon):
            return record
    return None


def latlon_to_dtm_pixel(dtm_path: Path, lat: float,
                        lon_signed: float) -> tuple[float, float] | None:
    """Convert a real lat/lon into dtm_path's own pixel (row, col). Returns
    None if the resulting pixel falls outside the raster's bounds."""
    with rasterio.open(dtm_path) as src:
        transformer = pyproj.Transformer.from_crs(
            src.crs.geodetic_crs, src.crs, always_xy=True,
        )
        x, y = transformer.transform(lon_signed, lat)
        col, row = ~src.transform * (x, y)

        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        return (row, col)


def compass_heading_to_render_heading(compass_deg: float, transform) -> float:
    """Convert a real compass bearing (0=North, 90=East, clockwise) into
    render_ground_view's heading_deg convention (0=+row direction,
    90=+col direction), using the DTM's own affine transform so this works
    regardless of a particular raster's row/col orientation. Assumes an
    axis-aligned (unrotated) raster, matching dtm_arrays.py's existing
    treatment of these HiRISE products."""
    theta = math.radians(compass_deg)
    dr = math.cos(theta) / transform.e
    dc = math.sin(theta) / transform.a
    return math.degrees(math.atan2(dc, dr)) % 360.0
