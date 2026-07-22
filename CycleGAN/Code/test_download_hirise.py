from hirise_index import Footprint
from download_hirise import REGIONS, browse_url_for, footprint_in_region


def test_oxia_planum_region_matches_real_landing_site_footprint():
    # Real ExoMars Rosalind Franklin landing site: ~18.2N, 335.45E
    # (signed: -24.55). Sourced from published coordinates during the
    # region-filter fix (see project history).
    footprint = Footprint(
        obs_id="ESP_REAL_0001",
        min_lat=17.5, max_lat=18.5,
        min_lon=-25.5, max_lon=-24.5,
        projection="EQUIRECTANGULAR",
        file_name_spec="RDR/ESP/ORB_X/ESP_REAL_0001/ESP_REAL_0001_RED.JP2",
    )
    assert footprint_in_region(footprint, REGIONS["oxia_planum"])


def test_oxia_planum_region_rejects_south_polar_footprint():
    # Reproduces the real bug found in the existing 304-image dataset:
    # ESP_039912_1095 was filed under oxia_planum but its real footprint
    # is at lat -70.34 (south polar), nowhere near Oxia Planum.
    footprint = Footprint(
        obs_id="ESP_039912_1095",
        min_lat=-70.6, max_lat=-70.1,
        min_lon=178.0, max_lon=178.4,
        projection="Polar_Stereographic MARS",
        file_name_spec="RDR/ESP/ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_RED.JP2",
    )
    assert not footprint_in_region(footprint, REGIONS["oxia_planum"])


def test_gale_crater_region_matches_real_curiosity_landing_site():
    # Real Curiosity landing site: ~-4.5S, 137.4E.
    footprint = Footprint(
        obs_id="ESP_REAL_0002",
        min_lat=-5.0, max_lat=-4.0,
        min_lon=137.0, max_lon=138.0,
        projection="EQUIRECTANGULAR",
        file_name_spec="RDR/ESP/ORB_Y/ESP_REAL_0002/ESP_REAL_0002_RED.JP2",
    )
    assert footprint_in_region(footprint, REGIONS["gale_crater"])


def test_browse_url_for_converts_jp2_path_to_browse_jpg():
    url = browse_url_for(
        "RDR/ESP/ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_RED.JP2"
    )
    assert url == (
        "https://hirise.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/"
        "ORB_039900_039999/ESP_039912_1095/ESP_039912_1095_RED.browse.jpg"
    )


def test_browse_url_for_returns_none_for_non_jp2_path():
    assert browse_url_for("RDR/ESP/ORB_X/ESP_X/ESP_X_COLOR.IMG") is None
