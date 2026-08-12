import pytest
from pathlib import Path

from rover_localization import parse_localization_csv, SiteDrivePose

# Real header + a handful of real rows from localized_interp.csv (fetched
# 2026-08-08), trimmed to columns we use plus enough to prove filtering works.
SAMPLE_CSV = """frame,site,drive,pose,landing_x,landing_y,landing_z,northing,easting,planetocentric_latitude,planetodetic_latitude,longitude,elevation,map_pixel_line,map_pixel_sample,dem_pixel_line,dem_pixel_sample,roll,pitch,yaw,quat_s,quat_v1,quat_v2,quat_v3,sclk,sol
SITE,1,-1,-1,0.000,0.000,0.000,-272039.268,8146811.223,-4.589466996,-4.643738049,137.441632997,-4501.040,2101.07,23092.89,526.19,5773.73,0.000,-0.000,0.000,1.000000000,0.000000000,0.000000000,0.000000000,0,-1
ROVER,1,0,0,0.000,0.000,0.000,-272039.268,8146811.223,-4.589466996,-4.643738049,137.441632997,-4501.040,2101.07,23092.89,526.19,5773.73,-2.414,-3.612,112.722,0.554112319,0.014571694,-0.034982380,0.831578882,397502188,0
ROVER,1,2,-1,0.100,0.000,0.000,-272040.100,8146812.000,-4.589400000,-4.643700000,137.441700000,-4501.000,2101.10,23093.00,526.20,5773.80,-2.414,-3.612,112.722,0.554112319,0.014571694,-0.034982380,0.831578882,397504952,0
ROVER,4,2100,8,1.000,0.000,0.000,-272100.000,8146900.000,-4.590000000,-4.644000000,137.442000000,-4501.500,2101.50,23095.00,526.50,5774.00,-2.400,-3.600,45.000,0.554112319,0.014571694,-0.034982380,0.831578882,401573000,46
"""


def test_parse_localization_csv_keeps_only_rover_frame(tmp_path):
    csv_path = tmp_path / "localized_interp.csv"
    csv_path.write_text(SAMPLE_CSV)

    result = parse_localization_csv(csv_path)

    # 3 ROVER rows in the fixture; the SITE row must be excluded.
    assert len(result) == 3
    assert (1, -1) not in result  # that key belongs to the excluded SITE row


def test_parse_localization_csv_keys_by_site_and_drive(tmp_path):
    csv_path = tmp_path / "localized_interp.csv"
    csv_path.write_text(SAMPLE_CSV)

    result = parse_localization_csv(csv_path)

    pose = result[(4, 2100)]
    assert isinstance(pose, SiteDrivePose)
    assert pose.site == 4
    assert pose.drive == 2100
    assert pose.latitude == pytest.approx(-4.590000000)
    assert pose.longitude == pytest.approx(137.442000000)
    assert pose.elevation == pytest.approx(-4501.500)
    assert pose.yaw_deg == pytest.approx(45.000)
    assert pose.sol == 46


def test_parse_localization_csv_prefers_pose_minus_one_on_duplicate_site_drive(tmp_path):
    # Real data sometimes has multiple ROVER rows for the same (site, drive)
    # with different `pose` sub-index values (see ROVER_MOTION_COUNTER's
    # third field). The -1 entry is the site/drive's default/summary row —
    # prefer it when present.
    csv = SAMPLE_CSV + (
        "ROVER,4,2100,-1,2.000,0.000,0.000,-272200.000,8147000.000,"
        "-4.600000000,-4.650000000,137.450000000,-4502.000,2102.00,23096.00,"
        "527.00,5775.00,-2.300,-3.500,50.000,0.55,0.01,-0.03,0.83,401573500,46\n"
    )
    csv_path = tmp_path / "localized_interp.csv"
    csv_path.write_text(csv)

    result = parse_localization_csv(csv_path)

    assert result[(4, 2100)].yaw_deg == pytest.approx(50.000)  # the pose=-1 row, not pose=8
