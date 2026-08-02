from fetch_hirise_dtm import parse_productfiles_html

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


def test_parse_productfiles_html_groups_ortho_by_obs_id():
    result = parse_productfiles_html(SAMPLE_PRODUCTFILES_HTML)
    assert set(result.keys()) == {"ESP_015985_2040", "ESP_016262_2040"}
    assert len(result["ESP_015985_2040"]) == 2  # A and C variants
    assert all(u.endswith("_ORTHO.JP2") for u in result["ESP_015985_2040"])


def test_parse_productfiles_html_no_orthos():
    result = parse_productfiles_html("<html><body>no links here</body></html>")
    assert result == {}
