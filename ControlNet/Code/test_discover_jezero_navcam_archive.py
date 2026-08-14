from unittest.mock import Mock

from discover_jezero_navcam_archive import classify_fetch_result


def _mock_response(status_code, text="", headers=None):
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    return resp


def test_classify_fetch_result_flags_real_pds3_label_content():
    resp = _mock_response(200, text="PDS_VERSION_ID = PDS3\nGROUP = SITE_DERIVED_GEOMETRY_PARMS")
    assert classify_fetch_result(resp) == "real_content"


def test_classify_fetch_result_flags_pds4_xml_content():
    resp = _mock_response(200, text='<?xml version="1.0"?><Product_Observational>')
    assert classify_fetch_result(resp) == "real_content"


def test_classify_fetch_result_flags_html_redirect_landing_page():
    resp = _mock_response(200, text="<html><body>Found</body></html>")
    assert classify_fetch_result(resp) == "html_not_data"


def test_classify_fetch_result_flags_error_status():
    resp = _mock_response(404, text="")
    assert classify_fetch_result(resp) == "http_error"
