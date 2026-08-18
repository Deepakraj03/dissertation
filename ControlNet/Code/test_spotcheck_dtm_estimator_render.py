from spotcheck_dtm_estimator_render import crop_window_centered


def test_crop_window_centered_on_a_large_raster():
    row_start, col_start = crop_window_centered((1000, 1000), 500, 500, 256)
    assert row_start == 500 - 128
    assert col_start == 500 - 128


def test_crop_window_centered_clamps_to_top_left_when_near_edge():
    row_start, col_start = crop_window_centered((1000, 1000), 10, 10, 256)
    assert row_start == 0
    assert col_start == 0


def test_crop_window_centered_clamps_to_bottom_right_when_near_far_edge():
    row_start, col_start = crop_window_centered((1000, 1000), 990, 990, 256)
    assert row_start == 1000 - 256
    assert col_start == 1000 - 256


def test_crop_window_centered_result_always_stays_in_bounds():
    for center in [0, 1, 500, 999, 1000]:
        row_start, col_start = crop_window_centered((1000, 1000), center, center, 256)
        assert 0 <= row_start <= 1000 - 256
        assert 0 <= col_start <= 1000 - 256
