import numpy as np

import sqp_reference as ref


def test_inactive_case_matches_classic_cylinder_result():
    out = ref.reference_solution(V0=10.0, h_max=5.0)
    r_star = (10.0 / (2 * np.pi)) ** (1 / 3)
    assert out["case"] == "inactive"
    assert abs(out["r"] - r_star) < 1e-12
    assert abs(out["h"] - 2 * r_star) < 1e-12
    assert out["mu"] == 0.0


def test_active_case_has_positive_multiplier():
    out = ref.reference_solution(V0=10.0, h_max=1.5)
    assert out["case"] == "active"
    assert out["h"] == 1.5
    assert out["mu"] > 0.0
