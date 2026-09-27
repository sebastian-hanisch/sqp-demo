"""Referenzloesung (Kopie der Hand-Formel aus Stueck 4 / kkt_solver.py, kein Import) - der Punkt,
gegen den SQP konvergieren muss."""
import numpy as np

import sqp_functions as fn


def reference_solution(V0: float, h_max: float) -> dict:
    r_free = (V0 / (2 * np.pi)) ** (1 / 3)
    h_free = 2 * r_free
    if h_free <= h_max:
        return {"r": r_free, "h": h_free, "mu": 0.0, "case": "inactive"}
    r_b = float(np.sqrt(V0 / (np.pi * h_max)))
    h_b = h_max
    x = np.array([r_b, h_b])
    gf = fn.grad_surface_area(x)
    gg1 = fn.grad_volume_constraint(x)
    lam = float(-gf[0] / gg1[0])
    mu = float(-(gf[1] + lam * gg1[1]))
    return {"r": r_b, "h": h_b, "mu": mu, "case": "active"}
