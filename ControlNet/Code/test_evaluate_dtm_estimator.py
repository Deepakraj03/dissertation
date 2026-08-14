import numpy as np
import pytest

from evaluate_dtm_estimator import compute_rmse


def test_compute_rmse_zero_for_identical_arrays():
    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert compute_rmse(real.copy(), real) == pytest.approx(0.0)


def test_compute_rmse_known_closed_form_value():
    estimated = np.array([[0.0, 0.0]], dtype=np.float32)
    real = np.array([[3.0, 4.0]], dtype=np.float32)
    # errors are 3 and 4 -> RMSE = sqrt((9+16)/2) = sqrt(12.5)
    assert compute_rmse(estimated, real) == pytest.approx(np.sqrt(12.5))


def test_compute_rmse_ignores_nan_in_real_ground_truth():
    estimated = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    real = np.array([[3.0, np.nan, 4.0]], dtype=np.float32)
    # only the two non-NaN cells count: errors 3 and 4
    assert compute_rmse(estimated, real) == pytest.approx(np.sqrt(12.5))


def test_compute_rmse_all_nan_real_returns_nan():
    estimated = np.zeros((2, 2), dtype=np.float32)
    real = np.full((2, 2), np.nan, dtype=np.float32)
    assert np.isnan(compute_rmse(estimated, real))
