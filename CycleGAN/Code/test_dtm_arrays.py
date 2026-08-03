import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from dtm_arrays import DtmArrays, load_dtm_arrays


def _write_geotiff(path, arr, pixel_scale_m):
    transform = from_origin(0, 0, pixel_scale_m, pixel_scale_m)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, transform=transform, crs=CRS.from_epsg(4326),
    ) as dst:
        dst.write(arr, 1)


def test_load_dtm_arrays_same_resolution(tmp_path):
    dtm_path = tmp_path / "dtm.tif"
    ortho_path = tmp_path / "ortho.tif"
    _write_geotiff(dtm_path, np.full((20, 20), 5.0, dtype=np.float32), pixel_scale_m=1.0)
    _write_geotiff(ortho_path, np.full((20, 20), 200, dtype=np.uint8), pixel_scale_m=1.0)

    result = load_dtm_arrays(dtm_path, ortho_path)

    assert isinstance(result, DtmArrays)
    assert result.heightmap.shape == (20, 20)
    assert result.albedo.shape == (20, 20)
    assert result.pixel_scale_m == pytest.approx(1.0)
    assert result.heightmap[0, 0] == pytest.approx(5.0)
    assert result.albedo[0, 0] == 200


def test_load_dtm_arrays_resamples_higher_res_ortho(tmp_path):
    # Ortho at 2x the DTM's resolution (common in practice — HiRISE orthos
    # can be finer than the derived DTM) must be resampled down to match
    # the DTM's grid so the two arrays are co-registered pixel-for-pixel.
    dtm_path = tmp_path / "dtm.tif"
    ortho_path = tmp_path / "ortho.tif"
    _write_geotiff(dtm_path, np.full((10, 10), 3.0, dtype=np.float32), pixel_scale_m=2.0)
    _write_geotiff(ortho_path, np.full((20, 20), 150, dtype=np.uint8), pixel_scale_m=1.0)

    result = load_dtm_arrays(dtm_path, ortho_path)

    assert result.heightmap.shape == (10, 10)
    assert result.albedo.shape == (10, 10)  # resampled to match the DTM grid
    assert result.pixel_scale_m == pytest.approx(2.0)


def test_load_dtm_arrays_masks_nodata_to_nan(tmp_path):
    dtm_path = tmp_path / "dtm_nodata.tif"
    ortho_path = tmp_path / "ortho_nodata.tif"
    nodata_val = -3.4028235e+38
    arr = np.full((10, 10), 5.0, dtype=np.float32)
    arr[0, 0] = nodata_val
    transform = from_origin(0, 0, 1.0, 1.0)
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype=np.float32, transform=transform, nodata=nodata_val,
    ) as dst:
        dst.write(arr, 1)
    _write_geotiff(ortho_path, np.full((10, 10), 100, dtype=np.uint8), pixel_scale_m=1.0)

    result = load_dtm_arrays(dtm_path, ortho_path)

    assert np.isnan(result.heightmap[0, 0])
    assert result.heightmap[5, 5] == pytest.approx(5.0)


def test_load_dtm_arrays_resample_path_preserves_real_variation(tmp_path):
    # Guards against a regression back to an all-zero resampled ortho.
    # Resampling no longer goes through rasterio.warp.reproject/CRS/PROJ at
    # all (see _resample_ortho_to_dtm_grid) — it's direct affine-coordinate
    # bilinear interpolation between the DTM's and ortho's own transforms.
    # This test doesn't exercise any CRS-mismatch mechanism (there's no CRS
    # involved in this code path any more); it just uses a non-constant
    # source array, so a regression back to a degenerate all-zero (or
    # otherwise flat) resample is actually caught — constant-valued
    # fixtures would not catch this, which is exactly how the original
    # reproject-based bug went undetected.
    dtm_path = tmp_path / "dtm_variation.tif"
    ortho_path = tmp_path / "ortho_variation.tif"
    dtm_arr = np.full((20, 20), 5.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    ortho_arr = rng.integers(50, 200, size=(40, 40), dtype=np.uint8)

    transform_dtm = from_origin(0, 0, 2.0, 2.0)
    transform_ortho = from_origin(0, 0, 1.0, 1.0)
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=20, width=20, count=1,
        dtype=np.float32, transform=transform_dtm,
    ) as dst:
        dst.write(dtm_arr, 1)
    with rasterio.open(
        ortho_path, "w", driver="GTiff", height=40, width=40, count=1,
        dtype=np.uint8, transform=transform_ortho,
    ) as dst:
        dst.write(ortho_arr, 1)

    result = load_dtm_arrays(dtm_path, ortho_path)

    assert result.albedo.shape == (20, 20)
    # A zero-filled (buggy) resample would have std=0; real resampled
    # variation from the non-constant source must be preserved.
    assert result.albedo.std() > 5.0
    assert result.albedo.max() > 0


def test_resample_ortho_to_dtm_grid_correct_bilinear_values():
    # A distinctive horizontal gradient (0, 10, 20, ..., 190 across 20
    # columns) at higher resolution than the DTM grid, so we can verify the
    # resampled DTM-grid values land at the geometrically correct positions
    # and interpolate correctly, not just "some non-zero value."
    from dtm_arrays import _resample_ortho_to_dtm_grid

    ortho_arr = np.tile(np.arange(0, 200, 10, dtype=np.uint8), (20, 1))  # (20, 20), gradient along columns
    ortho_transform = from_origin(0, 0, 1.0, 1.0)  # 1m/px, origin (0,0)
    dtm_transform = from_origin(0, 0, 2.0, 2.0)    # 2m/px, same origin — every DTM pixel covers a 2x2 ortho block
    dtm_shape = (10, 10)

    result = _resample_ortho_to_dtm_grid(ortho_arr, ortho_transform, dtm_transform, dtm_shape)

    assert result.shape == (10, 10)
    assert result.std() > 5.0  # real spatial variation, not a flat fill
    # DTM column 0 covers ortho columns 0-1 (gradient values 0,10) -> ~5
    # DTM column 5 covers ortho columns 10-11 (gradient values 100,110) -> ~105
    assert result[5, 0] == pytest.approx(5.0, abs=15.0)
    assert result[5, 5] == pytest.approx(105.0, abs=15.0)
    # Must be monotonically non-decreasing across columns (matches the source gradient)
    assert (np.diff(result[5, :]) >= -1.0).all()
