# test_check_pose_feasibility.py
from unittest.mock import Mock, patch

from check_pose_feasibility import list_navcam_products_for_sol, run_feasibility_check
from rover_localization import SiteDrivePose


def _mock_listing_response(html: str):
    resp = Mock()
    resp.text = html
    resp.raise_for_status = Mock()
    return resp


def test_list_navcam_products_for_sol_extracts_full_frame_ids_only():
    html = """
    <a href="NLA_401573345EDR_F0042100NCAM00307M1.LBL">...</a>
    <a href="NLA_401573345EDR_T0042100NCAM00307M1.LBL">...</a>
    <a href="NLA_401578433EDR_F0042100NCAM00308M1.LBL">...</a>
    <a href="NLB_504037552EDR_F0520000NCAM00327M1.LBL">...</a>
    <a href="NLB_504037552EDR_T0520000NCAM00327M1.LBL">...</a>
    <a href="NRB_504037552EDR_F0520000NCAM00327M1.LBL">...</a>
    """
    with patch("check_pose_feasibility.requests.get",
               return_value=_mock_listing_response(html)):
        result = list_navcam_products_for_sol(46)

    # "_F" (full frame) variants only, "_T" (thumbnail) excluded; both the
    # "A" and "B" flight-string variants are matched.
    assert result == [
        "NLA_401573345EDR_F0042100NCAM00307M1",
        "NLA_401578433EDR_F0042100NCAM00308M1",
        "NLB_504037552EDR_F0520000NCAM00327M1",
        "NRB_504037552EDR_F0520000NCAM00327M1",
    ]


def test_run_feasibility_check_tallies_success_and_failure():
    localization = {
        (4, 2100): SiteDrivePose(site=4, drive=2100, latitude=-4.59,
                                 longitude=137.44, elevation=-4501.5,
                                 yaw_deg=45.0, sol=46),
    }

    def fake_list_products(sol):
        return ["PRODUCT_OK", "PRODUCT_MISSING_SITE"]

    def fake_fetch_and_parse_pose(product_id, sol, localization):
        if product_id == "PRODUCT_OK":
            return Mock(site=4, drive=2100)
        return None

    with patch("check_pose_feasibility.list_navcam_products_for_sol",
               side_effect=fake_list_products), \
         patch("check_pose_feasibility.fetch_and_parse_pose",
               side_effect=fake_fetch_and_parse_pose):
        report = run_feasibility_check(sols=[46], per_sol=2,
                                       localization=localization)

    assert report["products_attempted"] == 2
    assert report["products_parsed"] == 1
    assert report["parse_success_rate"] == 0.5
