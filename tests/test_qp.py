import numpy as np

import sqp_functions as fn
import sqp_qp as qp


def test_inactive_case_selected_when_predicted_feasible():
    x = np.array([2.154435, 2.154435])
    d, lam_qp, mu_qp, case = qp.solve_qp_subproblem(x, -2 / 10.0 ** (1 / 3), 10.0, 5.0)
    assert case == "inactive"
    assert mu_qp == 0.0


def test_active_case_step_satisfies_linearized_height_constraint():
    x = np.array([2.0, 2.0])
    d, lam_qp, mu_qp, case = qp.solve_qp_subproblem(x, -1.0, 10.0, 1.5)
    if case == "active":
        g2v = fn.height_constraint(x, 1.5)
        gg2 = fn.grad_height_constraint()
        assert abs(g2v + gg2 @ d) < 1e-9
        assert mu_qp >= 0.0


def test_qp_step_solves_the_linear_kkt_system_exactly():
    x = np.array([1.3, 2.1])
    lam = 0.5
    V0, h_max = 10.0, 5.0
    d, lam_qp, _mu_qp, case = qp.solve_qp_subproblem(x, lam, V0, h_max)
    assert case == "inactive"
    HL = fn.hess_surface_area() + lam * fn.hess_volume_constraint(x)
    gg1 = fn.grad_volume_constraint(x)
    residual_stationarity = HL @ d + fn.grad_surface_area(x) + lam_qp * gg1
    residual_constraint = gg1 @ d + fn.volume_constraint(x, V0)
    assert np.max(np.abs(residual_stationarity)) < 1e-8
    assert abs(residual_constraint) < 1e-8
