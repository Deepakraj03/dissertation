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
