from unittest.mock import Mock, patch

from hirise_fullres import real_rdr_url_for, download_with_verify


def test_real_rdr_url_for_red_jp2():
    spec = "RDR/ESP/ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_RED.JP2"
    assert real_rdr_url_for(spec) == (
        "https://hirise.lpl.arizona.edu/PDS/RDR/ESP/ORB_039900_039999/"
        "ESP_039912_1095/ESP_039912_1095_RED.JP2"
    )


def test_real_rdr_url_for_non_red_returns_none():
    spec = "RDR/ESP/ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_COLOR.JP2"
    assert real_rdr_url_for(spec) is None


def test_real_rdr_url_for_non_jp2_returns_none():
    spec = "RDR/ESP/ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_RED.IMG"
    assert real_rdr_url_for(spec) is None


def _mock_response(content: bytes, content_length: int | None = None):
    resp = Mock()
    resp.headers = {"Content-Length": str(
        content_length if content_length is not None else len(content)
    )}
    resp.iter_content = lambda chunk_size: [content]
    resp.raise_for_status = Mock()
    return resp


def test_download_with_verify_succeeds_on_matching_size(tmp_path):
    dest = tmp_path / "test.jp2"
    content = b"x" * 1000

    with patch("hirise_fullres.requests.get", return_value=_mock_response(content)):
        result = download_with_verify("http://example.com/f.jp2", dest)

    assert result is True
    assert dest.exists()
    assert dest.stat().st_size == 1000


def test_download_with_verify_deletes_and_fails_on_size_mismatch(tmp_path):
    dest = tmp_path / "test.jp2"
    content = b"x" * 1000

    with patch("hirise_fullres.requests.get",
               return_value=_mock_response(content, content_length=2000)):
        result = download_with_verify("http://example.com/f.jp2", dest)

    assert result is False
    assert not dest.exists()


import numpy as np

from hirise_fullres import extract_qualifying_patches, sample_patch_positions


def test_sample_patch_positions_stays_in_bounds():
    positions = sample_patch_positions(width=1000, height=800, patch_size=256,
                                       n_candidates=50, seed=0)

    assert len(positions) == 50
    for y, x in positions:
        assert 0 <= y <= 800 - 256
        assert 0 <= x <= 1000 - 256


def test_sample_patch_positions_is_reproducible_with_same_seed():
    a = sample_patch_positions(width=1000, height=800, patch_size=256,
                               n_candidates=20, seed=42)
    b = sample_patch_positions(width=1000, height=800, patch_size=256,
                               n_candidates=20, seed=42)

    assert a == b


def test_extract_qualifying_patches_stops_at_target_count():
    rng = np.random.default_rng(0)
    # High-entropy (random noise) 512x512 array -> every patch qualifies.
    arr = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
    positions = [(0, 0), (0, 256), (256, 0), (256, 256)]

    patches = extract_qualifying_patches(arr, positions, patch_size=256,
                                         target_count=2)

    assert len(patches) == 2
    assert all(p.shape == (256, 256) for p in patches)


def test_extract_qualifying_patches_skips_low_entropy_patches():
    # Uniform (zero-entropy) array -> no patch should qualify.
    arr = np.zeros((512, 512), dtype=np.uint8)
    positions = [(0, 0), (0, 256), (256, 0), (256, 256)]

    patches = extract_qualifying_patches(arr, positions, patch_size=256,
                                         target_count=2)

    assert patches == []
