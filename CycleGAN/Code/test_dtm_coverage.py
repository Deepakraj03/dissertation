import pytest
from dtm_coverage import (
    build_ode_dtm_query_url,
    DtmCoverageRecord,
    parse_ode_response,
)

SAMPLE_RESPONSE_MULTI = {
    "ODEResults": {
        "Products": {
            "Product": [
                {
                    "LabelURL": "https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/DTEEC_015985_2040_016262_2040_U01.IMG",
                    "Minimum_latitude": "23.6875",
                    "Maximum_latitude": "23.9711",
                    "Westernmost_longitude": "341.035",
                    "Easternmost_longitude": "341.178",
                    "Comment": "Possible MSL landing site in Mawrth Vallis",
                    "ODE_notes": {
                        "ODE_note": [
                            "NOTE: Product Type set by ODE",
                            "\"MROHR_0001\",\"DTM/ESP/ORB_015900_015999/ESP_015985_2040_ESP_016262_2040/DTEEC_015985_2040_016262_2040_U01.IMG\",\"MRO\",\"HIRISE\",\"DTEEC_015985_2040_016262_2040_U01\",\"1  \",\"MARS                            \",\"Possible MSL landing site in Mawrth Vallis                                 \",\"ESP_015985_2040\",\"ESP_016262_2040\",\"NA                               \",\"DTM             \", 16689,  7940",
                        ]
                    },
                },
                {
                    "LabelURL": "https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_017800_017899/ESP_017897_2045_ESP_018530_2045/DTEEC_017897_2045_018530_2045_A01.IMG",
                    "Minimum_latitude": "23.9681",
                    "Maximum_latitude": "24.2108",
                    "Westernmost_longitude": "341.396",
                    "Easternmost_longitude": "341.522",
                    "Comment": "Clay diversity on Mawrth Vallis flank",
                    "ODE_notes": {
                        "ODE_note": [
                            "\"MROHR_0001\",\"DTM/ESP/ORB_017800_017899/ESP_017897_2045_ESP_018530_2045/DTEEC_017897_2045_018530_2045_A01.IMG\",\"MRO\",\"HIRISE\",\"DTEEC_017897_2045_018530_2045_A01\",\"1  \",\"MARS                            \",\"Clay diversity on Mawrth Vallis flank                                      \",\"ESP_017897_2045\",\"ESP_018530_2045\",\"NA                               \",\"DTM             \", 14301,  6969",
                        ]
                    },
                },
            ]
        },
        "Status": "Success",
    }
}

SAMPLE_RESPONSE_SINGLE = {
    "ODEResults": {
        "Products": {
            "Product": SAMPLE_RESPONSE_MULTI["ODEResults"]["Products"]["Product"][0]
        },
        "Status": "Success",
    }
}

SAMPLE_RESPONSE_EMPTY = {
    "ODEResults": {"Count": "0", "Status": "Success"}
}


def test_build_ode_dtm_query_url():
    url = build_ode_dtm_query_url(16.0, 24.0, 335.0, 345.0)
    assert url.startswith("https://oderest.rsl.wustl.edu/live2/?")
    assert "target=mars" in url
    assert "ihid=MRO" in url
    assert "iid=HIRISE" in url
    assert "pt=DTM" in url
    assert "results=m" in url
    assert "minlat=16.0" in url
    assert "maxlat=24.0" in url
    assert "westlon=335.0" in url
    assert "eastlon=345.0" in url


def test_parse_ode_response_multiple_products():
    records = parse_ode_response(SAMPLE_RESPONSE_MULTI)
    assert len(records) == 2
    first = records[0]
    assert isinstance(first, DtmCoverageRecord)
    assert first.obs_id_a == "ESP_015985_2040"
    assert first.obs_id_b == "ESP_016262_2040"
    assert first.dtm_url == (
        "https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_015900_015999/"
        "ESP_015985_2040_ESP_016262_2040/DTEEC_015985_2040_016262_2040_U01.IMG"
    )
    assert first.min_lat == pytest.approx(23.6875)
    assert first.max_lat == pytest.approx(23.9711)
    assert first.min_lon == pytest.approx(341.035)
    assert first.max_lon == pytest.approx(341.178)
    assert "Mawrth Vallis" in first.comment


def test_parse_ode_response_single_product_not_wrapped_in_list():
    # The ODE API returns a bare dict (not a one-item list) for single
    # results — this is what breaks a naive implementation.
    records = parse_ode_response(SAMPLE_RESPONSE_SINGLE)
    assert len(records) == 1
    assert records[0].obs_id_a == "ESP_015985_2040"


def test_parse_ode_response_no_products():
    records = parse_ode_response(SAMPLE_RESPONSE_EMPTY)
    assert records == []
