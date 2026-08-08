import math
from collections import namedtuple

import pytest
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
    # Build a tiny real north-up GeoTIFF in a known equirectangular-like CRS
    # so this test exercises the real rasterio/pyproj path, not a mock.
    dtm_path = tmp_path / "test_dtm.tif"
    width, height = 100, 100
    pixel_size_deg = 0.001
    top_left_lon, top_left_lat = 137.0, -4.0
    transform = rasterio.transform.from_origin(
        top_left_lon, top_left_lat, pixel_size_deg, pixel_size_deg,
    )
    data = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=data.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)

    # Center of the raster: lon = top_left_lon + width/2*pixel_size,
    #                        lat = top_left_lat - height/2*pixel_size
    center_lon = top_left_lon + (width / 2) * pixel_size_deg
    center_lat = top_left_lat - (height / 2) * pixel_size_deg

    result = latlon_to_dtm_pixel(dtm_path, lat=center_lat, lon_signed=center_lon)

    assert result is not None
    row, col = result
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
