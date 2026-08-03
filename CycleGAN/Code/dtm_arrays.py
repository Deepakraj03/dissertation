"""Load a HiRISE DTM raster and its co-registered orthoimage into aligned
numpy arrays, resampling the ortho onto the DTM's pixel grid if their
native resolutions differ."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass
class DtmArrays:
    heightmap: np.ndarray  # float32, meters
    albedo: np.ndarray     # uint8, grayscale
    pixel_scale_m: float


def _resample_ortho_to_dtm_grid(ortho_arr: np.ndarray, ortho_transform,
                                dtm_transform, dtm_shape: tuple[int, int]) -> np.ndarray:
    """Resample ortho_arr (native pixel grid described by ortho_transform)
    onto the DTM's pixel grid (described by dtm_transform/dtm_shape) via
    direct affine-coordinate bilinear interpolation — no CRS/PROJ involved.
    Both rasters are assumed to already share the same real-world projected
    coordinate space (verified for HiRISE DTM/ortho pairs: their CRS objects
    carry cosmetically different WKT — different DATUM names and differently
    -encoded projection parameters — but describe the same physical Mars
    equirectangular projection with closely overlapping bounds). This
    sidesteps a confirmed rasterio.warp.reproject bug where this exact
    DTM/ortho pairing silently resamples to all-zero despite geographically
    overlapping bounds, even when src_crs/dst_crs are forced to match.
    Processes one destination row at a time (vectorized across columns) to
    bound peak memory on the large real rasters (DTM grids can be ~120M
    pixels) while avoiding a fully scalar Python double loop.
    """
    h_dtm, w_dtm = dtm_shape
    h_ortho, w_ortho = ortho_arr.shape
    inv_ortho = ~ortho_transform

    da, db, dc, dd, de, df = dtm_transform.a, dtm_transform.b, dtm_transform.c, \
        dtm_transform.d, dtm_transform.e, dtm_transform.f
    ia, ib, ic, id_, ie, if_ = inv_ortho.a, inv_ortho.b, inv_ortho.c, \
        inv_ortho.d, inv_ortho.e, inv_ortho.f

    col_idx = np.arange(w_dtm, dtype=np.float64) + 0.5
    out = np.zeros(dtm_shape, dtype=np.float64)

    for row in range(h_dtm):
        row_center = row + 0.5
        xs = da * col_idx + db * row_center + dc
        ys = dd * col_idx + de * row_center + df

        # inv_ortho maps world coords to corner-based fractional pixel
        # coordinates (integer values fall exactly on pixel corners, e.g.
        # rasterio/Affine convention), but bilinear interpolation needs
        # center-based coordinates (pixel i's value represents the point at
        # its center, i + 0.5) — subtract 0.5 to convert, otherwise sampling
        # carries a systematic half-pixel bias and spuriously runs out of
        # bounds one pixel early at the trailing edge of the raster.
        src_cols = ia * xs + ib * ys + ic - 0.5
        src_rows = id_ * xs + ie * ys + if_ - 0.5

        r0 = np.floor(src_rows).astype(np.int64)
        c0 = np.floor(src_cols).astype(np.int64)
        r1 = r0 + 1
        c1 = c0 + 1
        fr = src_rows - r0
        fc = src_cols - c0

        valid = (r0 >= 0) & (c0 >= 0) & (r1 < h_ortho) & (c1 < w_ortho)
        r0c = np.clip(r0, 0, h_ortho - 1)
        r1c = np.clip(r1, 0, h_ortho - 1)
        c0c = np.clip(c0, 0, w_ortho - 1)
        c1c = np.clip(c1, 0, w_ortho - 1)

        top = ortho_arr[r0c, c0c] * (1 - fc) + ortho_arr[r0c, c1c] * fc
        bot = ortho_arr[r1c, c0c] * (1 - fc) + ortho_arr[r1c, c1c] * fc
        vals = top * (1 - fr) + bot * fr
        out[row, :] = np.where(valid, vals, 0.0)

    return out


def load_dtm_arrays(dtm_path: Path, ortho_path: Path) -> DtmArrays:
    with rasterio.open(dtm_path) as dtm_src:
        heightmap = dtm_src.read(1, masked=True).astype(np.float32).filled(np.nan)
        dtm_transform = dtm_src.transform
        pixel_scale_m = abs(dtm_transform.a)

    with rasterio.open(ortho_path) as ortho_src:
        ortho_arr = ortho_src.read(1)
        if ortho_src.shape == heightmap.shape and ortho_src.transform == dtm_transform:
            albedo = ortho_arr
        else:
            albedo = _resample_ortho_to_dtm_grid(
                ortho_arr, ortho_src.transform, dtm_transform, heightmap.shape,
            )

    return DtmArrays(
        heightmap=heightmap,
        albedo=albedo.astype(np.uint8),
        pixel_scale_m=pixel_scale_m,
    )
