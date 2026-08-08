"""Geometry glue between a real lat/lon (from parse_rover_pose.py) and a
specific HiRISE DTM's pixel grid, plus converting a compass heading into
render_ground_view.py's own heading_deg convention."""

import math
from pathlib import Path

import pyproj
import rasterio

from dtm_coverage import DtmCoverageRecord


def find_covering_dtm(lat: float, lon_signed: float,
                      records: list[DtmCoverageRecord]) -> DtmCoverageRecord | None:
    """Cheap bounding-box pre-filter: which record (if any) covers this
    lat/lon. Assumes records' min_lon/max_lon are already directly
    comparable to lon_signed — true for Gale Crater (~135-140E, no dateline
    crossing), see this plan's Global Constraints for why that's safe here
    specifically."""
    for record in records:
        if (record.min_lat <= lat <= record.max_lat and
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
