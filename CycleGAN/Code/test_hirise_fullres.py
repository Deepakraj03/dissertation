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
