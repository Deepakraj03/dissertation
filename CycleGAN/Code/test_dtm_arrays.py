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


def test_load_dtm_arrays_resamples_correctly_despite_differing_datum_labels(tmp_path):
    # Reproduces the real bug: HiRISE DTM and ortho products can share the
    # same physical Mars-equirectangular projection but carry different
    # DATUM labels in their CRS WKT (e.g. "D_MARS" vs "unnamed"), which can
    # make PROJ silently fail the coordinate transform during reproject and
    # zero-fill the destination instead of raising. Uses two CRS objects
    # with matching projection parameters/bounds but different datum names,
    # plus a non-constant source array, so a regression back to all-zero
    # output is actually caught (constant-valued fixtures would not catch
    # this, which is exactly how the original bug went undetected).
    from rasterio.crs import CRS

    dtm_crs = CRS.from_wkt(
        'PROJCS["EQUIRECTANGULAR MARS",GEOGCS["GCS_MARS",'
        'DATUM["D_MARS",SPHEROID["MARS_localRadius",3396190,0]],'
        'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
        'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],UNIT["metre",1]]'
    )
    ortho_crs = CRS.from_wkt(
        'PROJCS["Equirectangular MARS",GEOGCS["GCS_MARS",'
        'DATUM["unnamed",SPHEROID["unnamed",3396190,0]],'
        'PRIMEM["Reference meridian",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Equirectangular"],PARAMETER["latitude_of_origin",0],'
        'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],UNIT["metre",1]]'
    )

    dtm_path = tmp_path / "dtm_datum.tif"
    ortho_path = tmp_path / "ortho_datum.tif"
    dtm_arr = np.full((20, 20), 5.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    ortho_arr = rng.integers(50, 200, size=(40, 40), dtype=np.uint8)

    transform_dtm = from_origin(0, 0, 2.0, 2.0)
    transform_ortho = from_origin(0, 0, 1.0, 1.0)
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=20, width=20, count=1,
        dtype=np.float32, transform=transform_dtm, crs=dtm_crs,
    ) as dst:
        dst.write(dtm_arr, 1)
    with rasterio.open(
        ortho_path, "w", driver="GTiff", height=40, width=40, count=1,
        dtype=np.uint8, transform=transform_ortho, crs=ortho_crs,
    ) as dst:
        dst.write(ortho_arr, 1)

    result = load_dtm_arrays(dtm_path, ortho_path)

    assert result.albedo.shape == (20, 20)
    # A zero-filled (buggy) resample would have std=0; real resampled
    # variation from the non-constant source must be preserved.
    assert result.albedo.std() > 5.0
    assert result.albedo.max() > 0
