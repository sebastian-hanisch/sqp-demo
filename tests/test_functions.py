import numpy as np

import sqp_functions as fn


def test_surface_area_matches_formula():
    x = np.array([2.0, 3.0])
    assert fn.surface_area(x) == 2 * np.pi * 4 + 2 * np.pi * 2 * 3


def test_gradient_matches_finite_differences():
    eps = 1e-6
    x = np.array([1.7, 2.3])
    analytic = fn.grad_surface_area(x)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric = (fn.surface_area(xp) - fn.surface_area(xm)) / (2 * eps)
        assert abs(analytic[i] - numeric) < 1e-4


def test_hessian_matches_finite_differences_of_gradient():
    eps = 1e-6
    x = np.array([1.7, 2.3])
    analytic = fn.hess_surface_area()
    for j in range(2):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        numeric_col = (fn.grad_surface_area(xp) - fn.grad_surface_area(xm)) / (2 * eps)
        assert np.max(np.abs(analytic[:, j] - numeric_col)) < 1e-4


def test_volume_constraint_zero_at_exact_volume():
    r, h, V0 = 1.0, 10.0 / np.pi, 10.0
    assert abs(fn.volume_constraint(np.array([r, h]), V0)) < 1e-9


def test_height_constraint_sign():
    assert fn.height_constraint(np.array([1.0, 2.0]), 3.0) < 0
    assert fn.height_constraint(np.array([1.0, 4.0]), 3.0) > 0
