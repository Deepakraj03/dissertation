import pytest

from ctx_mosaic import tile_name_for, tiles_for_bbox


def test_tile_name_for_matches_known_grid_point():
    # AEB_000001_0150's footprint (from test_hirise_index.py), signed
    # lon/lat ~ (-59.80, -52.32). Verified against the real Murray Lab
    # tile listing at https://murray-lab.caltech.edu/CTX/V01/tiles/ —
    # this exact tile filename exists there.
    assert tile_name_for(-59.80, -52.32) == "MurrayLab_GlobalCTXMosaic_V01_E-060_N-56"


def test_tile_name_for_positive_coordinates():
    # 4-degree cell containing (5.0, 10.0) is (4, 8).
    assert tile_name_for(5.0, 10.0) == "MurrayLab_GlobalCTXMosaic_V01_E004_N08"


def test_tile_name_for_zero_is_not_negative_zero():
    assert tile_name_for(0.0, 0.0) == "MurrayLab_GlobalCTXMosaic_V01_E000_N00"


def test_tiles_for_bbox_within_single_tile():
    # A small box fully inside the (-60, -56) cell.
    result = tiles_for_bbox(-59.9, -59.1, -55.9, -55.1)
    assert result == ["MurrayLab_GlobalCTXMosaic_V01_E-060_N-56"]


def test_tiles_for_bbox_spanning_two_tiles():
    # Box straddles the lon=-60 cell boundary.
    result = tiles_for_bbox(-60.5, -59.5, -55.9, -55.1)
    assert len(result) == 2


def test_download_tile_retries_once_then_raises(tmp_path, monkeypatch):
    import ctx_mosaic

    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(ctx_mosaic.requests, "get", fake_get)

    with pytest.raises(ConnectionError):
        ctx_mosaic.download_tile("MurrayLab_GlobalCTXMosaic_V01_E-060_N-56", tmp_path)

    # Initial attempt + exactly one retry = 2 calls.
    assert calls["n"] == 2
