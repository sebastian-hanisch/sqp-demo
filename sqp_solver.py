"""Aeussere SQP-Iteration: ruft je Schritt sqp_qp.solve_qp_subproblem() auf und macht den vollen
Schritt (kein Line-Search/Merit-Function noetig - Vehikel B ist klein genug, dass der volle
Newton-Schritt konvergiert, wie schon in Stueck 2/4 beobachtet). Startwert standardmaessig aus
Dimensionsanalyse hergeleitet ($x_0=(V_0^{1/3},V_0^{1/3})$, $\\lambda_0=-2/V_0^{1/3}$, identisch zu
Stueck 4) - ein neutraler Startwert wie $\\lambda_0=0$ kann zu einem voruebergehenden
Fehlgriff bei der aktiven Menge und/oder einem physikalisch unsinnigen Ergebnis mit negativem
Radius fuehren (siehe README, dieselbe Klasse Fund wie Stueck 4s SciPy-Fallstrick)."""
from dataclasses import dataclass

import numpy as np

import sqp_qp as qp


@dataclass
class SQPSolution:
    r: float
    h: float
    lam: float
    mu: float
    case: str
    n_iter: int
    converged: bool
    cases: list
    trajectory: list


def sqp_solve(V0: float, h_max: float, x0=None, lam0: float = None, max_iter: int = 50,
             tol: float = 1e-10) -> SQPSolution:
    cube = V0 ** (1 / 3)
    x = np.array(x0, dtype=float) if x0 is not None else np.array([cube, cube])
    lam = lam0 if lam0 is not None else -2 / cube
    mu = 0.0
    case = "inactive"
    cases = []
    trajectory = [(float(x[0]), float(x[1]))]
    for k in range(max_iter):
        d, lam_new, mu_new, case = qp.solve_qp_subproblem(x, lam, V0, h_max)
        cases.append(case)
        if np.linalg.norm(d) < tol:
            return SQPSolution(r=float(x[0]), h=float(x[1]), lam=lam, mu=mu, case=case,
                               n_iter=k, converged=True, cases=cases, trajectory=trajectory)
        x = x + d
        lam = lam_new
        mu = mu_new
        trajectory.append((float(x[0]), float(x[1])))
    return SQPSolution(r=float(x[0]), h=float(x[1]), lam=lam, mu=mu, case=case, n_iter=max_iter,
                       converged=False, cases=cases, trajectory=trajectory)
