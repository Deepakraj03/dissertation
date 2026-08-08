import math
from collections import namedtuple

import pytest
import pyproj
import rasterio
import numpy as np

from dtm_geo_lookup import (
    find_covering_dtm, latlon_to_dtm_pixel, compass_heading_to_render_heading,
)
from dtm_coverage import DtmCoverageRecord


def _make_record(product_id, min_lat, max_lat, min_lon, max_lon):
    return DtmCoverageRecord(
        product_id=product_id, dtm_url="", obs_id_a="", obs_id_b="",
        min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,
        comment="", files_url="",
    )


def test_find_covering_dtm_returns_matching_record():
    records = [
        _make_record("A", min_lat=-5.0, max_lat=-4.0, min_lon=137.0, max_lon=138.0),
        _make_record("B", min_lat=10.0, max_lat=11.0, min_lon=140.0, max_lon=141.0),
    ]
    result = find_covering_dtm(lat=-4.5, lon_signed=137.5, records=records)
    assert result.product_id == "A"


def test_find_covering_dtm_returns_none_when_uncovered():
    records = [_make_record("A", min_lat=-5.0, max_lat=-4.0, min_lon=137.0, max_lon=138.0)]
    result = find_covering_dtm(lat=20.0, lon_signed=50.0, records=records)
    assert result is None


def test_compass_heading_to_render_heading_standard_north_up_raster():
    # Standard north-up HiRISE-style raster: a>0 (col -> east), e<0 (row -> south).
    Affine = namedtuple("Affine", ["a", "e"])
    transform = Affine(a=1.0, e=-1.0)

    # Compass South (180deg) -> render's +row direction -> heading_deg 0.
    assert compass_heading_to_render_heading(180.0, transform) == pytest.approx(0.0, abs=1e-6)
    # Compass East (90deg) -> render's +col direction -> heading_deg 90.
    assert compass_heading_to_render_heading(90.0, transform) == pytest.approx(90.0, abs=1e-6)
    # Compass North (0deg) -> render's -row direction -> heading_deg 180.
    assert compass_heading_to_render_heading(0.0, transform) == pytest.approx(180.0, abs=1e-6)
    # Compass West (270deg) -> render's -col direction -> heading_deg 270.
    assert compass_heading_to_render_heading(270.0, transform) == pytest.approx(270.0, abs=1e-6)


def test_latlon_to_dtm_pixel_roundtrips_through_a_real_geotiff(tmp_path):
    # Build a tiny real north-up GeoTIFF in a Mars equirectangular projected CRS
    # (not geographic EPSG:4326) so this test exercises the real rasterio/pyproj
    # CRS-transformation path, not a degenerate identity transform.
    dtm_path = tmp_path / "test_dtm.tif"
    width, height = 100, 100
    pixel_size_m = 100.0  # 100m pixels

    # Mars equirectangular CRS centered at (0, 0) with IAU mean radius
    mars_eqc_proj_str = (
        "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 "
        "+R=3396190 +units=m +no_defs"
    )
    mars_crs = pyproj.CRS.from_proj4(mars_eqc_proj_str)

    # Origin at a known point in projected coordinates (meters)
    origin_x, origin_y = 1000000.0, -2000000.0
    transform = rasterio.transform.from_origin(
        origin_x, origin_y, pixel_size_m, pixel_size_m,
    )

    data = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=data.dtype, crs=mars_crs, transform=transform,
    ) as dst:
        dst.write(data, 1)

    # Compute the center point in projected coordinates (meters)
    center_x = origin_x + (width / 2) * pixel_size_m
    center_y = origin_y - (height / 2) * pixel_size_m

    # Transform back to lat/lon via pyproj to get exact coordinates
    transformer_to_latlon = pyproj.Transformer.from_crs(
        mars_crs, mars_crs.geodetic_crs, always_xy=True,
    )
    center_lon, center_lat = transformer_to_latlon.transform(center_x, center_y)

    # Now call the function with these exact lat/lon values
    result = latlon_to_dtm_pixel(dtm_path, lat=center_lat, lon_signed=center_lon)

    assert result is not None
    row, col = result
    # Should be near the center pixel; allow 1.0 pixel tolerance for rounding
    assert row == pytest.approx(height / 2, abs=1.0)
    assert col == pytest.approx(width / 2, abs=1.0)


def test_latlon_to_dtm_pixel_returns_none_outside_bounds(tmp_path):
    dtm_path = tmp_path / "test_dtm.tif"
    transform = rasterio.transform.from_origin(137.0, -4.0, 0.001, 0.001)
    data = np.zeros((100, 100), dtype=np.float32)
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=100, width=100, count=1,
        dtype=data.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)

    result = latlon_to_dtm_pixel(dtm_path, lat=20.0, lon_signed=50.0)
    assert result is None
