from unittest.mock import Mock

from fetch_hirise_dtm import parse_productfiles_html, fetch_dtm_and_orthos
from dtm_coverage import DtmCoverageRecord

# Trimmed real fixture, captured 2026-08-02 from
# https://ode.rsl.wustl.edu/mars/productfiles.aspx?product_id=DTEEC_015985_2040_016262_2040_U01&product_idGeo=26731198
SAMPLE_PRODUCTFILES_HTML = """
<html><body>
<a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/DTEEC_015985_2040_016262_2040_U01.IMG">DTM</a>
<a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_015985_2040_RED_A_01_ORTHO.JP2">Ortho A</a>
<a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_015985_2040_RED_A_01_ORTHO.LBL">Ortho A label</a>
<a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_015985_2040_RED_C_01_ORTHO.JP2">Ortho C</a>
<a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_016262_2040_RED_A_01_ORTHO.JP2">Ortho A</a>
<a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_016262_2040_RED_C_01_ORTHO.JP2">Ortho C</a>
</body></html>
"""

SAMPLE_RECORD = DtmCoverageRecord(
    product_id="DTEEC_015985_2040_016262_2040_U01",
    dtm_url="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/DTEEC_015985_2040_016262_2040_U01.IMG",
    obs_id_a="ESP_015985_2040",
    obs_id_b="ESP_016262_2040",
    min_lat=23.6875, max_lat=23.9711,
    min_lon=341.035, max_lon=341.178,
    comment="test",
    files_url="https://ode.rsl.wustl.edu/mars/productfiles.aspx?product_id=DTEEC_015985_2040_016262_2040_U01&product_idGeo=26731198",
)


def test_parse_productfiles_html_groups_ortho_by_obs_id():
    result = parse_productfiles_html(SAMPLE_PRODUCTFILES_HTML)
    assert set(result.keys()) == {"ESP_015985_2040", "ESP_016262_2040"}
    assert len(result["ESP_015985_2040"]) == 2  # A and C variants
    assert all(u.endswith("_ORTHO.JP2") for u in result["ESP_015985_2040"])


def test_parse_productfiles_html_no_orthos():
    result = parse_productfiles_html("<html><body>no links here</body></html>")
    assert result == {}


def test_fetch_dtm_and_orthos_success_path_prefers_a_variant(monkeypatch, tmp_path):
    """Test the full success path: DTM downloaded, productfiles fetched, orthos
    selected (preferring _A_ variant), and status "ok" returned with both
    ortho_paths populated."""
    scratch_dir = tmp_path / "scratch"
    dest_dir = tmp_path / "dest"

    # Mock download_with_verify to always succeed
    mock_download = Mock(return_value=True)
    monkeypatch.setattr("fetch_hirise_dtm.download_with_verify", mock_download)

    # Mock requests.get to return the productfiles HTML
    mock_response = Mock()
    mock_response.text = SAMPLE_PRODUCTFILES_HTML
    mock_response.raise_for_status = Mock()
    mock_requests = Mock(return_value=mock_response)
    monkeypatch.setattr("fetch_hirise_dtm.requests.get", mock_requests)

    result = fetch_dtm_and_orthos(SAMPLE_RECORD, scratch_dir, dest_dir)

    # Verify success
    assert result["status"] == "ok"
    assert result["product_id"] == "DTEEC_015985_2040_016262_2040_U01"
    assert "dtm_path" in result
    assert "ortho_paths" in result
    assert len(result["ortho_paths"]) == 2
    assert "ESP_015985_2040" in result["ortho_paths"]
    assert "ESP_016262_2040" in result["ortho_paths"]

    # Verify that _A_ variant was chosen for both observations
    # Call sequence: DTM, obs_a ortho, obs_b ortho
    assert mock_download.call_count == 3
    calls = mock_download.call_args_list
    # Call 0: DTM
    assert calls[0][0][0] == SAMPLE_RECORD.dtm_url
    # Call 1: obs_a ortho (should be _A_ variant)
    assert "_A_" in calls[1][0][0]
    assert "ESP_015985_2040" in calls[1][0][0]
    # Call 2: obs_b ortho (should be _A_ variant)
    assert "_A_" in calls[2][0][0]
    assert "ESP_016262_2040" in calls[2][0][0]

    # Verify requests.get was called with the files_url
    mock_requests.assert_called_once()
    assert SAMPLE_RECORD.files_url in str(mock_requests.call_args)


def test_fetch_dtm_and_orthos_dtm_download_failed(monkeypatch, tmp_path):
    """Test that DTM download failure returns appropriate status."""
    scratch_dir = tmp_path / "scratch"
    dest_dir = tmp_path / "dest"

    # Mock download_with_verify to fail on first call (DTM download)
    mock_download = Mock(return_value=False)
    monkeypatch.setattr("fetch_hirise_dtm.download_with_verify", mock_download)

    result = fetch_dtm_and_orthos(SAMPLE_RECORD, scratch_dir, dest_dir)

    assert result["status"] == "dtm_download_failed"
    assert result["product_id"] == "DTEEC_015985_2040_016262_2040_U01"
    # Verify that requests.get was never called (failed before that point)
    # by checking that only one download was attempted (the DTM)
    assert mock_download.call_count == 1


def test_fetch_dtm_and_orthos_no_ortho_found_for_observation(monkeypatch, tmp_path):
    """Test that missing ortho for an observation returns appropriate error."""
    scratch_dir = tmp_path / "scratch"
    dest_dir = tmp_path / "dest"

    # Mock download_with_verify to succeed for DTM
    mock_download = Mock(return_value=True)
    monkeypatch.setattr("fetch_hirise_dtm.download_with_verify", mock_download)

    # Return HTML with only one observation's orthos (missing the second)
    html_missing_obs_b = """
    <html><body>
    <a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_015985_2040_RED_A_01_ORTHO.JP2">Ortho A</a>
    <a href="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/ESP_015985_2040_RED_C_01_ORTHO.JP2">Ortho C</a>
    </body></html>
    """

    mock_response = Mock()
    mock_response.text = html_missing_obs_b
    mock_response.raise_for_status = Mock()
    mock_requests = Mock(return_value=mock_response)
    monkeypatch.setattr("fetch_hirise_dtm.requests.get", mock_requests)

    result = fetch_dtm_and_orthos(SAMPLE_RECORD, scratch_dir, dest_dir)

    # Should fail because ESP_016262_2040 (obs_id_b) has no orthos in the HTML
    assert "no_ortho_found_for_" in result["status"]
    assert "ESP_016262_2040" in result["status"]
    assert result["product_id"] == "DTEEC_015985_2040_016262_2040_U01"
