import numpy as np

import sqp_reference as ref
import sqp_solver as solver


def test_sqp_converges_to_reference_inactive_case():
    reference = ref.reference_solution(V0=10.0, h_max=5.0)
    out = solver.sqp_solve(10.0, 5.0)
    assert out.converged
    assert abs(out.r - reference["r"]) < 1e-8
    assert abs(out.h - reference["h"]) < 1e-8
    assert out.n_iter <= 8


def test_sqp_converges_to_reference_active_case():
    reference = ref.reference_solution(V0=10.0, h_max=1.5)
    out = solver.sqp_solve(10.0, 1.5)
    assert out.converged
    assert abs(out.r - reference["r"]) < 1e-8
    assert abs(out.h - reference["h"]) < 1e-8
    assert out.n_iter <= 8


def test_sqp_trajectory_starts_at_informed_start_and_ends_at_solution():
    out = solver.sqp_solve(10.0, 5.0)
    cube = 10.0 ** (1 / 3)
    assert out.trajectory[0] == (cube, cube)
    assert abs(out.trajectory[-1][0] - out.r) < 1e-9
    assert abs(out.trajectory[-1][1] - out.h) < 1e-9


def test_naive_lambda_start_still_converges_but_needs_more_iterations():
    naive = solver.sqp_solve(10.0, 5.0, lam0=0.0)
    informed = solver.sqp_solve(10.0, 5.0)
    assert naive.converged
    assert naive.n_iter > informed.n_iter
