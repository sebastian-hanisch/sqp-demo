"""Loest die kleine SQP-QP-Teilaufgabe je Iteration per aktive-Menge-Enumeration (dieselbe Idee
wie kkt_solver.py in Stueck 4, hier je Iteration neu geprueft, weil sich die Linearisierung mit x
aendert): da nur eine Ungleichung vorliegt, gibt es genau zwei Faelle (Hoehenlimit in der QP
inaktiv/aktiv)."""
import numpy as np

import sqp_functions as fn


def solve_qp_subproblem(x: np.ndarray, lam: float, V0: float, h_max: float):
    """Minimiere grad_f^T d + 0.5 d^T H_L d unter grad_g1^T d = -g1(x) und
    grad_g2^T d <= -g2(x), mit H_L = Hess(f) + lam*Hess(g1) (g2 ist linear, Hess(g2)=0). Gibt
    (d, lam_neu, mu_neu, fall) zurueck."""
    r, h = x
    gf = fn.grad_surface_area(x)
    gg1 = fn.grad_volume_constraint(x)
    gg2 = fn.grad_height_constraint()
    HL = fn.hess_surface_area() + lam * fn.hess_volume_constraint(x)
    g1_val = fn.volume_constraint(x, V0)
    g2_val = fn.height_constraint(x, h_max)

    # Fall "inaktiv": nur die Gleichung linearisiert.
    K = np.zeros((3, 3))
    K[:2, :2] = HL
    K[:2, 2] = gg1
    K[2, :2] = gg1
    rhs = np.concatenate([-gf, [-g1_val]])
    sol = np.linalg.solve(K, rhs)
    d_inactive = sol[:2]
    lam_qp = float(sol[2])
    predicted_g2 = g2_val + gg2 @ d_inactive
    if predicted_g2 <= 1e-12:
        return d_inactive, lam_qp, 0.0, "inactive"

    # Fall "aktiv": zusaetzlich grad_g2^T d = -g2(x) erzwingen (d_h = -g2_val, da grad_g2=(0,1)).
    d_h = -g2_val
    A = np.array([[HL[0, 0], gg1[0]], [gg1[0], 0.0]])
    b = np.array([-gf[0] - HL[0, 1] * d_h, -g1_val - gg1[1] * d_h])
    sol2 = np.linalg.solve(A, b)
    d_r, lam_qp2 = float(sol2[0]), float(sol2[1])
    d = np.array([d_r, d_h])
    mu_qp = float(-(gf[1] + HL[1, 0] * d_r + HL[1, 1] * d_h + lam_qp2 * gg1[1]))
    return d, lam_qp2, mu_qp, "active"
