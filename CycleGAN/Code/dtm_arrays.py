"""Load a HiRISE DTM raster and its co-registered orthoimage into aligned
numpy arrays, resampling the ortho onto the DTM's pixel grid if their
native resolutions differ."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


@dataclass
class DtmArrays:
    heightmap: np.ndarray  # float32, meters
    albedo: np.ndarray     # uint8, grayscale
    pixel_scale_m: float


def load_dtm_arrays(dtm_path: Path, ortho_path: Path) -> DtmArrays:
    with rasterio.open(dtm_path) as dtm_src:
        heightmap = dtm_src.read(1).astype(np.float32)
        dtm_transform = dtm_src.transform
        dtm_crs = dtm_src.crs
        pixel_scale_m = abs(dtm_transform.a)

    with rasterio.open(ortho_path) as ortho_src:
        if ortho_src.shape == heightmap.shape:
            albedo = ortho_src.read(1)
        else:
            albedo = np.empty(heightmap.shape, dtype=ortho_src.dtypes[0])
            reproject(
                source=rasterio.band(ortho_src, 1),
                destination=albedo,
                src_transform=ortho_src.transform,
                src_crs=ortho_src.crs,
                dst_transform=dtm_transform,
                dst_crs=dtm_crs or ortho_src.crs,
                resampling=Resampling.bilinear,
            )

    return DtmArrays(
        heightmap=heightmap,
        albedo=albedo.astype(np.uint8),
        pixel_scale_m=pixel_scale_m,
    )
