from render_pitch_bucket_samples import bucket_poses_by_pitch
from parse_rover_pose import RoverPose


def _pose(product_id, pitch_deg):
    return RoverPose(product_id=product_id, sol=46, site=4, drive=2100,
                     latitude=-4.5, longitude=137.5,
                     compass_heading_deg=55.0, pitch_deg=pitch_deg)


def test_bucket_poses_by_pitch_assigns_by_absolute_value():
    poses = [
        _pose("A", pitch_deg=-22.0),   # 20-25 bucket
        _pose("B", pitch_deg=-30.0),   # 25-35 bucket
        _pose("C", pitch_deg=-45.0),   # 35-50 bucket
        _pose("D", pitch_deg=-10.0),   # below all buckets -- unassigned
    ]
    buckets = [(20.0, 25.0), (25.0, 35.0), (35.0, 50.0)]

    result = bucket_poses_by_pitch(poses, buckets)

    assert {p.product_id for p in result["20.0-25.0"]} == {"A"}
    assert {p.product_id for p in result["25.0-35.0"]} == {"B"}
    assert {p.product_id for p in result["35.0-50.0"]} == {"C"}
    assert sum(len(v) for v in result.values()) == 3  # D excluded


def test_bucket_poses_by_pitch_handles_empty_bucket():
    poses = [_pose("A", pitch_deg=-22.0)]
    buckets = [(20.0, 25.0), (25.0, 35.0)]

    result = bucket_poses_by_pitch(poses, buckets)

    assert result["25.0-35.0"] == []
