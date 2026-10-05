"""Unabhängige Orakel für SQP: (1) Gesamtlösung über die Reduktion auf eine Variable r
(h = V0/(pi r^2), f(r) = 2 pi r^2 + 2 V0/r konvex), (2) QP-Teilaufgabe über die Parametrisierung der
linearisierten Gleichungsgeraden, (3) Straf-Ableitungen per finiter Differenz und
Barriere-Minimierer/-Stufenzahl per Goldener-Schnitt-Suche."""
import numpy as np
import pytest

import sqp_evaluation as ev
import sqp_functions as fn
import sqp_qp as qp
import sqp_solver as solver

PI = np.pi


def oracle(V0, h_max):
    r = max((V0 / (2 * PI)) ** (1 / 3), np.sqrt(V0 / (PI * h_max)))
    return r, V0 / (PI * r * r)


@pytest.mark.parametrize("V0,h_max", [(10.0, 5.0), (10.0, 1.5), (1.0, 0.5), (1.0, 6.0), (100.0, 0.5),
                                      (100.0, 6.0), (37.0, 3.3), (62.0, 2.2), (5.0, 2.0), (80.0, 5.9)])
def test_sqp_solution_and_multipliers_match_reduced_problem(V0, h_max):
    r, h = oracle(V0, h_max)
    out = solver.sqp_solve(V0, h_max)
    assert out.converged
    assert abs(out.r - r) < 1e-8 and abs(out.h - h) < 1e-8
    x = np.array([r, h])
    gf, g1, g2 = (fn.grad_surface_area(x), fn.grad_volume_constraint(x), fn.grad_height_constraint())
    active = abs(h - h_max) < 1e-9
    assert out.case == ("active" if active else "inactive")
    if active:
        lam, mu = np.linalg.solve(np.column_stack([g1, g2]), -gf)
    else:
        lam, mu = -gf[0] / g1[0], 0.0
    assert abs(out.lam - lam) < 1e-6 * max(1.0, abs(lam))
    assert abs(out.mu - mu) < 1e-6 * max(1.0, abs(mu))


def _qp_oracle(x, lam, V0, h_max):
    gf = fn.grad_surface_area(x)
    g1 = fn.grad_volume_constraint(x)
    HL = fn.hess_surface_area() + lam * fn.hess_volume_constraint(x)
    z = np.array([-g1[1], g1[0]])
    d0 = g1 * (-fn.volume_constraint(x, V0)) / (g1 @ g1)
    a = 0.5 * z @ HL @ z
    b = gf @ z + d0 @ HL @ z
    rhs = -fn.height_constraint(x, h_max) - d0[1]
    t = -b / (2 * a)
    if z[1] > 0:
        t = min(t, rhs / z[1])
    elif z[1] < 0:
        t = max(t, rhs / z[1])
    return d0 + t * z, a


def test_qp_subproblem_matches_line_parametrization_oracle():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(300):
        x = np.array([rng.uniform(0.2, 4.0), rng.uniform(0.2, 8.0)])
        lam = rng.uniform(-4.0, 2.0)
        V0, h_max = rng.uniform(1, 100), rng.uniform(0.5, 6.0)
        d_orc, a = _qp_oracle(x, lam, V0, h_max)
        if a <= 0:  # nicht konvex auf der Geraden: Minimum nicht eindeutig, hier nicht verglichen
            continue
        d, _lam_new, _mu, _case = qp.solve_qp_subproblem(x, lam, V0, h_max)
        assert np.max(np.abs(d - d_orc)) < 1e-7 * max(1.0, float(np.max(np.abs(d_orc))))
        checked += 1
    assert checked > 100


def test_penalty_gradient_and_hessian_match_finite_differences():
    rng = np.random.default_rng(5)

    def pen(x, V0, h_max, rho):
        return (fn.surface_area(x) + 0.5 * rho * fn.volume_constraint(x, V0) ** 2
                + 0.5 * rho * max(fn.height_constraint(x, h_max), 0.0) ** 2)

    for _ in range(40):
        V0, h_max, rho = rng.uniform(1, 100), rng.uniform(0.5, 6.0), 10 ** rng.uniform(-1, 3)
        x = np.array([rng.uniform(0.3, 4.0), rng.uniform(0.3, 8.0)])
        e = np.eye(2) * 1e-6
        g_num = np.array([(pen(x + e[i], V0, h_max, rho) - pen(x - e[i], V0, h_max, rho)) / 2e-6
                          for i in range(2)])
        g = ev._penalty_grad(x, V0, h_max, rho)
        assert np.max(np.abs(g - g_num) / np.maximum(1.0, np.abs(g_num))) < 1e-5
        H_num = np.column_stack([(ev._penalty_grad(x + e[i], V0, h_max, rho)
                                  - ev._penalty_grad(x - e[i], V0, h_max, rho)) / 2e-6
                                 for i in range(2)])
        H = ev._penalty_hess(x, V0, h_max, rho)
        assert np.max(np.abs(H - H_num) / np.maximum(1.0, np.abs(H_num))) < 1e-4


def _golden_min(f, lo, hi, iters=200):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(iters):
        c, d = b - g * (b - a), a + g * (b - a)
        if f(c) < f(d):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


def _barrier_stages_oracle(V0, h_max, target=1e-6, max_outer=40):
    r_ref, h_ref = oracle(V0, h_max)
    rb = np.sqrt(V0 / (PI * h_max))
    mu = 1.0
    for k in range(max_outer):
        def psi(r):
            c = h_max - V0 / (PI * r * r)
            return 2 * PI * r * r + 2 * V0 / r - mu * np.log(c) if c > 0 else 1e300
        r = _golden_min(psi, rb * (1 + 1e-13), 50 * rb + 50)
        if np.hypot(r - r_ref, V0 / (PI * r * r) - h_ref) < target:
            return k + 1
        mu /= 3.0
    return None


@pytest.mark.parametrize("V0,h_max,expected", [(10.0, 5.0, 12), (10.0, 1.5, 13), (50.0, 2.0, 12),
                                               (100.0, 6.0, 13), (20.0, 5.0, 13)])
def test_barrier_stage_count_matches_golden_section_oracle(V0, h_max, expected):
    assert _barrier_stages_oracle(V0, h_max) == expected
    assert ev.iteration_count_comparison(V0, h_max)["barrier_iter"] == expected
