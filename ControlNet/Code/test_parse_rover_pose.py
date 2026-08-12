from unittest.mock import Mock, patch

import pytest

from parse_rover_pose import (
    label_url_for,
    parse_navcam_label,
    fetch_and_parse_pose,
    RoverPose,
)
from rover_localization import SiteDrivePose

# Real label content (fetched 2026-08-09 from
# msl_navcam_raw/DATA/SOL02001/NLB_575142982EDR_F0682626NCAM00207M1.LBL),
# trimmed to the two groups this module actually reads. This is a real
# example of the 2026-08-09 down-look/workspace-context content-mismatch
# finding: SITE_DERIVED_GEOMETRY_PARMS.INSTRUMENT_ELEVATION=-38.3 (steeply
# down), not the near-horizontal pointing the old RSM-joint-angle
# approximation implicitly assumed.
SAMPLE_LABEL = """PDS_VERSION_ID                    = PDS3
PRODUCT_ID                        = "NLB_575142982EDR_F0682626NCAM00207M1"
ROVER_MOTION_COUNTER              = (68,2626,25,196,0,0,290,108,12,0)
ROVER_MOTION_COUNTER_NAME         = ("SITE","DRIVE","POSE","ARM","CHIMRA",
                                     "DRILL","RSM","HGA","DRT","IC")

GROUP                             = ROVER_DERIVED_GEOMETRY_PARMS
  INSTRUMENT_AZIMUTH              = 336.478 <deg>
  INSTRUMENT_ELEVATION            = -46.9314 <deg>
  REFERENCE_COORD_SYSTEM_INDEX    = (68,2626,25,196,0,0,290,108,12,0)
  REFERENCE_COORD_SYSTEM_NAME     = "ROVER_NAV_FRAME"
  SOLAR_AZIMUTH                   = 139.811 <deg>
  SOLAR_ELEVATION                 = 26.4747 <deg>
END_GROUP                         = ROVER_DERIVED_GEOMETRY_PARMS

GROUP                             = SITE_DERIVED_GEOMETRY_PARMS
  INSTRUMENT_AZIMUTH              = 125.84 <deg>
  INSTRUMENT_ELEVATION            = -38.3218 <deg>
  POSITIVE_AZIMUTH_DIRECTION      = CLOCKWISE
  REFERENCE_COORD_SYSTEM_INDEX    = 68
  REFERENCE_COORD_SYSTEM_NAME     = "SITE_FRAME"
  SOLAR_AZIMUTH                   = 285.822 <deg>
  SOLAR_ELEVATION                 = 21.1407 <deg>
END_GROUP                         = SITE_DERIVED_GEOMETRY_PARMS
"""


def test_label_url_for_matches_real_archive_path():
    assert label_url_for("NLA_401573345EDR_F0042100NCAM00307M1", sol=46) == (
        "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_navcam_raw/"
        "DATA/SOL00046/NLA_401573345EDR_F0042100NCAM00307M1.LBL"
    )


def test_parse_navcam_label_extracts_site_drive_and_real_pointing():
    result = parse_navcam_label(SAMPLE_LABEL)
    assert result["site"] == 68
    assert result["drive"] == 2626
    assert result["azimuth_deg"] == pytest.approx(125.84)
    assert result["elevation_deg"] == pytest.approx(-38.3218)


def test_parse_navcam_label_uses_site_frame_not_rover_frame():
    # ROVER_DERIVED_GEOMETRY_PARMS also has INSTRUMENT_AZIMUTH/ELEVATION
    # keys with different real values -- regression test that we scope to
    # SITE_DERIVED_GEOMETRY_PARMS specifically, not the first match.
    result = parse_navcam_label(SAMPLE_LABEL)
    assert result["azimuth_deg"] != pytest.approx(336.478)
    assert result["elevation_deg"] != pytest.approx(-46.9314)


def _mock_response(text: str):
    resp = Mock()
    resp.text = text
    resp.raise_for_status = Mock()
    return resp


def test_fetch_and_parse_pose_combines_label_and_localization():
    localization = {
        (68, 2626): SiteDrivePose(
            site=68, drive=2626, latitude=-4.59, longitude=137.44,
            elevation=-4501.5, yaw_deg=45.0, sol=2001,
        )
    }
    with patch("parse_rover_pose.requests.get",
               return_value=_mock_response(SAMPLE_LABEL)):
        pose = fetch_and_parse_pose(
            "NLB_575142982EDR_F0682626NCAM00207M1", sol=2001,
            localization=localization,
        )

    assert isinstance(pose, RoverPose)
    assert pose.site == 68 and pose.drive == 2626
    assert pose.latitude == pytest.approx(-4.59)
    assert pose.longitude == pytest.approx(137.44)
    # Real derived azimuth/elevation pass through directly -- no rover-yaw
    # composition needed, since SITE_DERIVED_GEOMETRY_PARMS is already
    # resolved into the absolute site frame by JPL's own geometry pipeline.
    assert pose.compass_heading_deg == pytest.approx(125.84)
    assert pose.pitch_deg == pytest.approx(-38.3218)


def test_fetch_and_parse_pose_returns_none_on_localization_miss():
    with patch("parse_rover_pose.requests.get",
               return_value=_mock_response(SAMPLE_LABEL)):
        pose = fetch_and_parse_pose(
            "NLB_575142982EDR_F0682626NCAM00207M1", sol=2001,
            localization={},  # (68, 2626) not present
        )
    assert pose is None


def test_fetch_and_parse_pose_returns_none_on_malformed_label():
    with patch("parse_rover_pose.requests.get",
               return_value=_mock_response("not a real label")):
        pose = fetch_and_parse_pose(
            "BAD_PRODUCT_ID", sol=1, localization={(68, 2626): Mock()},
        )
    assert pose is None


def test_fetch_and_parse_pose_returns_none_on_fetch_error():
    # Network error (connection failure) case
    import requests
    with patch("parse_rover_pose.requests.get",
               side_effect=requests.exceptions.ConnectionError("Network error")):
        pose = fetch_and_parse_pose(
            "NLB_575142982EDR_F0682626NCAM00207M1", sol=2001,
            localization={(68, 2626): Mock()},
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
            "NLB_575142982EDR_F0682626NCAM00207M1", sol=2001,
            localization={(68, 2626): Mock()},
        )
    assert pose is None
