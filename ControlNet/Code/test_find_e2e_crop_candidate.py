import numpy as np
import pytest

from find_e2e_crop_candidate import score_candidates, pick_best_candidate


def test_score_candidates_prefers_camera_facing_ground_structure():
    # Flat, empty (all-sky) terrain on one side of the raster vs. a bumpy,
    # textured patch on the other -- a camera near the bumpy side should
    # score higher (more non-sky ground drawn, more entropy in that ground).
    heightmap = np.zeros((400, 400), dtype=np.float32)
    albedo = np.full((400, 400), 50, dtype=np.uint8)
    # Add a textured, elevated bump in the bottom-right quadrant so a
    # nearby camera actually has ground to draw.
    rng = np.random.default_rng(0)
    heightmap[300:400, 300:400] = 5.0
    albedo[300:400, 300:400] = rng.integers(0, 255, size=(100, 100), dtype=np.uint8)

    candidates = [(50, 50), (350, 350)]  # far corner (isolated, likely all-sky) vs. near the bump
    scored = score_candidates(heightmap, albedo, pixel_scale_m=1.0,
                              candidates=candidates, pitch_deg=-15.0)

    scores_by_pos = {(c["row"], c["col"]): c["score"] for c in scored}
    assert scores_by_pos[(350, 350)] > scores_by_pos[(50, 50)]


def test_score_candidates_skips_unrenderable_camera_positions():
    heightmap = np.full((100, 100), np.nan, dtype=np.float32)
    albedo = np.zeros((100, 100), dtype=np.uint8)
    scored = score_candidates(heightmap, albedo, pixel_scale_m=1.0,
                              candidates=[(50, 50)], pitch_deg=-15.0)
    assert scored == []


def test_pick_best_candidate_returns_highest_score():
    scored = [
        {"row": 10, "col": 10, "score": 1.0, "sky_frac": 0.9},
        {"row": 20, "col": 20, "score": 5.0, "sky_frac": 0.3},
        {"row": 30, "col": 30, "score": 2.0, "sky_frac": 0.6},
    ]
    best = pick_best_candidate(scored)
    assert best["row"] == 20 and best["col"] == 20


def test_pick_best_candidate_raises_on_empty_list():
    with pytest.raises(ValueError):
        pick_best_candidate([])
