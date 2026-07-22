from hirise_fullres import real_rdr_url_for


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
