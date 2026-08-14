import numpy as np
import pytest

from height_encoding import decode_height_from_unit_range, encode_height_to_unit_range


def test_encode_maps_min_and_max_to_unit_range_endpoints():
    height = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
    encoded, meta = encode_height_to_unit_range(height)
    assert encoded.min() == pytest.approx(-1.0)
    assert encoded.max() == pytest.approx(1.0)
    assert meta == {"min": pytest.approx(0.0), "max": pytest.approx(10.0)}


def test_decode_is_exact_inverse_of_encode():
    rng = np.random.default_rng(0)
    height = rng.uniform(-50.0, 50.0, size=(32, 32)).astype(np.float32)

    encoded, meta = encode_height_to_unit_range(height)
    recovered = decode_height_from_unit_range(encoded, meta)

    np.testing.assert_allclose(recovered, height, atol=1e-3)


def test_flat_patch_does_not_divide_by_zero():
    height = np.full((8, 8), 3.0, dtype=np.float32)
    encoded, meta = encode_height_to_unit_range(height)
    assert not np.isnan(encoded).any()
    assert not np.isinf(encoded).any()


def test_nan_cells_filled_with_local_min_not_propagated():
    height = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    encoded, meta = encode_height_to_unit_range(height)
    assert not np.isnan(encoded).any()
    # The NaN cell was filled with the patch's own real min (1.0), which
    # encodes to -1.0 (the same as the real minimum) -- distinguishable
    # from an arbitrary sentinel, and never propagates NaN downstream.
    assert encoded[1, 0] == pytest.approx(-1.0)


def test_all_nan_patch_returns_zero_encoding_not_crash():
    height = np.full((4, 4), np.nan, dtype=np.float32)
    encoded, meta = encode_height_to_unit_range(height)
    assert not np.isnan(encoded).any()
