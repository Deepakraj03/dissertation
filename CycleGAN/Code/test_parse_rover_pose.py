from unittest.mock import Mock, patch

import pytest

from parse_rover_pose import (
    label_url_for,
    parse_navcam_label,
    fetch_and_parse_pose,
    RoverPose,
)
from rover_localization import SiteDrivePose

# Real label content (fetched 2026-08-08 from
# msl_navcam_raw/DATA/SOL00046/NLA_401573345EDR_F0042100NCAM00307M1.LBL),
# trimmed to the two groups this module actually reads.
SAMPLE_LABEL = """PDS_VERSION_ID                    = PDS3
PRODUCT_ID                        = "NLA_401573345EDR_F0042100NCAM00307M1"
ROVER_MOTION_COUNTER              = (4,2100,8,22,0,0,218,144,0,0)
ROVER_MOTION_COUNTER_NAME         = ("SITE","DRIVE","POSE","ARM","CHIMRA",
                                     "DRILL","RSM","HGA","DRT","IC")

GROUP                             = ARM_ARTICULATION_STATE_PARMS
  ARTICULATION_DEVICE_ID          = "ARM"
  ARTICULATION_DEVICE_ANGLE       = (0.315343 <rad>,-0.558888 <rad>)
END_GROUP                         = ARM_ARTICULATION_STATE_PARMS

GROUP                             = RSM_ARTICULATION_STATE_PARMS
  ARTICULATION_DEVICE_ID          = "RSM"
  ARTICULATION_DEVICE_NAME        = "REMOTE SENSING MAST"
  ARTICULATION_DEVICE_ANGLE       = (2.92119 <rad>,0.821286 <rad>,
                                     2.92742 <rad>,0.825024 <rad>,
                                     2.90017 <rad>,0.832174 <rad>,
                                     2.92738 <rad>,0.82503 <rad>)
  ARTICULATION_DEVICE_ANGLE_NAME  = ("AZIMUTH-MEASURED",
                                     "ELEVATION-MEASURED",
                                     "AZIMUTH-REQUESTED",
                                     "ELEVATION-REQUESTED","AZIMUTH-INITIAL"
                                     ,"ELEVATION-INITIAL","AZIMUTH-FINAL",
                                     "ELEVATION-FINAL")
  ARTICULATION_DEVICE_MODE        = DEPLOYED
END_GROUP                         = RSM_ARTICULATION_STATE_PARMS
"""


def test_label_url_for_matches_real_archive_path():
    assert label_url_for("NLA_401573345EDR_F0042100NCAM00307M1", sol=46) == (
        "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_navcam_raw/"
        "DATA/SOL00046/NLA_401573345EDR_F0042100NCAM00307M1.LBL"
    )


def test_parse_navcam_label_extracts_site_drive_and_mast_azimuth():
    result = parse_navcam_label(SAMPLE_LABEL)

    assert result["site"] == 4
    assert result["drive"] == 2100
    # AZIMUTH-MEASURED is the first value in RSM's ARTICULATION_DEVICE_ANGLE,
    # in radians (2.92119) -> degrees.
    import math
    assert result["mast_azimuth_deg"] == pytest.approx(math.degrees(2.92119))


def test_parse_navcam_label_ignores_arm_group_angle():
    # ARM_ARTICULATION_STATE_PARMS also has an ARTICULATION_DEVICE_ANGLE key
    # with the same name — regression test that we scope to the RSM group,
    # not just the first match in the file.
    result = parse_navcam_label(SAMPLE_LABEL)
    import math
    assert result["mast_azimuth_deg"] != pytest.approx(math.degrees(0.315343))


def _mock_response(text: str):
    resp = Mock()
    resp.text = text
    resp.raise_for_status = Mock()
    return resp


def test_fetch_and_parse_pose_combines_label_and_localization():
    localization = {
        (4, 2100): SiteDrivePose(
            site=4, drive=2100, latitude=-4.59, longitude=137.44,
            elevation=-4501.5, yaw_deg=45.0, sol=46,
        )
    }
    with patch("parse_rover_pose.requests.get",
               return_value=_mock_response(SAMPLE_LABEL)):
        pose = fetch_and_parse_pose(
            "NLA_401573345EDR_F0042100NCAM00307M1", sol=46,
            localization=localization,
        )

    assert isinstance(pose, RoverPose)
    assert pose.site == 4 and pose.drive == 2100
    assert pose.latitude == pytest.approx(-4.59)
    assert pose.longitude == pytest.approx(137.44)
    import math
    expected_heading = (45.0 + math.degrees(2.92119)) % 360
    assert pose.compass_heading_deg == pytest.approx(expected_heading)


def test_fetch_and_parse_pose_returns_none_on_localization_miss():
    with patch("parse_rover_pose.requests.get",
               return_value=_mock_response(SAMPLE_LABEL)):
        pose = fetch_and_parse_pose(
            "NLA_401573345EDR_F0042100NCAM00307M1", sol=46,
            localization={},  # (4, 2100) not present
        )
    assert pose is None


def test_fetch_and_parse_pose_returns_none_on_malformed_label():
    with patch("parse_rover_pose.requests.get",
               return_value=_mock_response("not a real label")):
        pose = fetch_and_parse_pose(
            "BAD_PRODUCT_ID", sol=1, localization={(4, 2100): Mock()},
        )
    assert pose is None


def test_fetch_and_parse_pose_returns_none_on_fetch_error():
    # Network error (connection failure) case
    import requests
    with patch("parse_rover_pose.requests.get",
               side_effect=requests.exceptions.ConnectionError("Network error")):
        pose = fetch_and_parse_pose(
            "NLA_401573345EDR_F0042100NCAM00307M1", sol=46,
            localization={(4, 2100): Mock()},
        )
    assert pose is None


def test_fetch_and_parse_pose_returns_none_on_http_error():
    # HTTP error (e.g. 404 Not Found) case
    import requests
    resp = Mock()
    resp.raise_for_status = Mock(
        side_effect=requests.exceptions.HTTPError("404 Not Found")
    )
    with patch("parse_rover_pose.requests.get", return_value=resp):
        pose = fetch_and_parse_pose(
            "NLA_401573345EDR_F0042100NCAM00307M1", sol=46,
            localization={(4, 2100): Mock()},
        )
    assert pose is None
