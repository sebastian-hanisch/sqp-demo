"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, exakte Newton-KKT-Reduktion (Kern-
Korrektheitsargument), Referenz-Konvergenz + Robustheit, exakte 1-Schritt-Konvergenz an einem
rein linear restringierten Hilfsproblem, Iterationszahl-Vergleich gegen Straf-/Barriere-Verfahren
(Stueck 5, eigene schlanke Kopien), Nebenbefund-Check (naiver Start), Gradienten-Check."""
from dataclasses import dataclass

import numpy as np

import sqp_functions as fn
import sqp_qp as qp
import sqp_reference as ref
import sqp_solver as solver


@dataclass(frozen=True)
class Settings:
    V0: float
    h_max: float


def analyse(settings: Settings) -> dict:
    solution = solver.sqp_solve(settings.V0, settings.h_max)
    f_val = fn.surface_area(np.array([solution.r, solution.h]))
    return {"solution": solution, "f_val": f_val}


def newton_kkt_reduction_check(V0: float = 10.0, h_max: float = 5.0) -> dict:
    """Bei fixierter aktiver Menge (Gleichungsfall) liefert der SQP-QP-Schritt exakt denselben
    Schritt wie ein ungedaempfter Newton-Schritt auf dem KKT-Stationaritaetssystem (Stueck 4) -
    keine Naeherung, eine exakte algebraische Identitaet."""
    x = np.array([1.3, 2.1])
    lam = 0.5
    d_qp, lam_qp, _mu_qp, case = qp.solve_qp_subproblem(x, lam, V0, h_max)
    assert case == "inactive", "Testpunkt muss den Gleichungsfall auslösen"

    gf = fn.grad_surface_area(x)
    gg1 = fn.grad_volume_constraint(x)
    HL = fn.hess_surface_area() + lam * fn.hess_volume_constraint(x)
    F = np.concatenate([gf + lam * gg1, [fn.volume_constraint(x, V0)]])
    J = np.zeros((3, 3))
    J[:2, :2] = HL
    J[:2, 2] = gg1
    J[2, :2] = gg1
    newton_step = np.linalg.solve(J, -F)
    # Newton auf F(v)=0 liefert ein INKREMENT (dr,dh,dlam); die QP-Loesung liefert lam_qp direkt
    # als NEUEN Multiplikator-Wert - beide Formen sind aequivalent, lam_qp = lam + dlam.
    newton_lam_new = lam + newton_step[2]

    max_abs_err = float(max(np.max(np.abs(d_qp - newton_step[:2])),
                           abs(lam_qp - newton_lam_new)))
    return {"max_abs_err": max_abs_err}


def reference_convergence_check(h_max_values=(5.0, 1.5)) -> list:
    """SQP konvergiert fuer beide Presets exakt gegen dieselbe Referenzloesung wie Stueck 4/5."""
    rows = []
    for h_max in h_max_values:
        reference = ref.reference_solution(10.0, h_max)
        solution = solver.sqp_solve(10.0, h_max)
        err = float(np.hypot(solution.r - reference["r"], solution.h - reference["h"]))
        rows.append({"h_max": h_max, "case": reference["case"], "n_iter": solution.n_iter,
                    "converged": solution.converged, "err": err})
    return rows


def robustness_across_volumes(V0_values=(0.5, 1.0, 5.0, 10.0, 50.0, 100.0)) -> list:
    """Der informierte Startwert (x0=(V0^(1/3),V0^(1/3)), lam0=-2/V0^(1/3), identisch zu Stueck 4)
    konvergiert robust und mit konstant wenigen Iterationen ueber sechs Groessenordnungen."""
    rows = []
    for V0 in V0_values:
        reference = ref.reference_solution(V0, 100.0)  # h_max grosszuegig, immer inaktiv
        solution = solver.sqp_solve(V0, 100.0)
        err = float(np.hypot(solution.r - reference["r"], solution.h - reference["h"]))
        rows.append({"V0": V0, "n_iter": solution.n_iter, "converged": solution.converged,
                    "matches_analytic": err < 1e-6 and solution.r > 0})
    return rows


# ---------- Hilfsproblem fuer Hook 3: rein linear restringiert ----------
def _aux_grad_f(x):
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])


def _aux_hess_f():
    return 2 * np.eye(2)


def _aux_g(x):
    return x[0] + x[1] - 4.0


def _aux_grad_g():
    return np.array([1.0, 1.0])


def linear_constraint_exactness_check(x0_values=((0.0, 0.0), (10.0, -5.0), (-3.0, 8.0),
                                                 (100.0, 100.0))) -> list:
    """Eigenes, kleines Hilfsproblem (KEIN Vehikel B): minimiere (x-2)^2+(y-3)^2 unter x+y=4 -
    Ziel quadratisch, Nebenbedingung bereits linear, also macht die Linearisierung KEINEN Fehler.
    SQP muss von JEDEM Startpunkt aus in genau einer Iteration konvergieren."""
    rows = []
    for x0 in x0_values:
        x = np.array(x0, dtype=float)
        for k in range(10):
            gf = _aux_grad_f(x)
            gg = _aux_grad_g()
            HL = _aux_hess_f()
            K = np.zeros((3, 3))
            K[:2, :2] = HL
            K[:2, 2] = gg
            K[2, :2] = gg
            rhs = np.concatenate([-gf, [-_aux_g(x)]])
            sol = np.linalg.solve(K, rhs)
            d = sol[:2]
            if np.linalg.norm(d) < 1e-12:
                rows.append({"x0": x0, "n_iter": k, "x_star": x.tolist()})
                break
            x = x + d
        else:
            rows.append({"x0": x0, "n_iter": 10, "x_star": x.tolist()})
    return rows


# ---------- Hilfs-Kopien der Straf-/Barriere-Folgen aus Stueck 5 (nur fuer den Vergleich) ----------
def _penalty_grad(x, V0, h_max, rho):
    gv1 = fn.volume_constraint(x, V0)
    gv2 = fn.height_constraint(x, h_max)
    grad = fn.grad_surface_area(x) + rho * gv1 * fn.grad_volume_constraint(x)
    if gv2 > 0:
        grad = grad + rho * gv2 * fn.grad_height_constraint()
    return grad


def _penalty_hess(x, V0, h_max, rho):
    gv1 = fn.volume_constraint(x, V0)
    gv2 = fn.height_constraint(x, h_max)
    gg1 = fn.grad_volume_constraint(x)
    hess = fn.hess_surface_area() + rho * (np.outer(gg1, gg1) + gv1 * fn.hess_volume_constraint(x))
    if gv2 > 0:
        gg2 = fn.grad_height_constraint()
        hess = hess + rho * np.outer(gg2, gg2)
    return hess


def _penalty_step(x0, V0, h_max, rho, tol=1e-8, max_iter=200):
    x = np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        grad = _penalty_grad(x, V0, h_max, rho)
        base_norm = float(np.linalg.norm(grad))
        if base_norm < tol:
            return x
        hess = _penalty_hess(x, V0, h_max, rho)
        delta = np.linalg.solve(hess, -grad)
        step = 1.0
        for _ in range(30):
            new_norm = float(np.linalg.norm(_penalty_grad(x + step * delta, V0, h_max, rho)))
            if new_norm < base_norm or step < 1e-8:
                break
            step *= 0.5
        x = x + step * delta
    return x


def _barrier_step(r0, V0, h_max, mu, tol=1e-8, max_iter=200):
    def c(r):
        return h_max - V0 / (np.pi * r ** 2)

    def dpsi(r):
        cc = c(r)
        return 4 * np.pi * r - 2 * V0 / r ** 2 - mu * (2 * V0 / (np.pi * r ** 3)) / cc

    def d2psi(r):
        cc = c(r)
        return (4 * np.pi + 4 * V0 / r ** 3 - mu * (-6 * V0 / (np.pi * r ** 4)) / cc
                + mu * (2 * V0 / (np.pi * r ** 3)) ** 2 / cc ** 2)

    r = r0
    for _ in range(max_iter):
        d = dpsi(r)
        if abs(d) < tol:
            return r
        delta = -d / d2psi(r)
        step = 1.0
        for _ in range(30):
            r_new = r + step * delta
            if c(r_new) > 1e-12 and abs(dpsi(r_new)) < abs(d):
                break
            step *= 0.5
        r = r + step * delta
    return r


def iteration_count_comparison(V0: float = 10.0, h_max: float = 5.0, target_err: float = 1e-6,
                               max_outer: int = 40) -> dict:
    """Wie viele aeussere Iterationen/Stufen braucht jedes Verfahren, um denselben Fehler
    (target_err) gegenueber der Referenzloesung zu unterschreiten? SQP direkt gegen die Straf-/
    Barriere-Folgen aus Stueck 5 (eigene, schlanke Kopien hier, kein Import)."""
    reference = ref.reference_solution(V0, h_max)

    solution = solver.sqp_solve(V0, h_max)
    sqp_iter = solution.n_iter

    cube = V0 ** (1 / 3)
    x = np.array([cube, cube])
    rho = 1.0
    penalty_iter = None
    for k in range(max_outer):
        x = _penalty_step(x, V0, h_max, rho)
        err = np.hypot(x[0] - reference["r"], x[1] - reference["h"])
        if err < target_err:
            penalty_iter = k + 1
            break
        rho *= 3.0
    # Nicht erreicht: das Strafverfahren kann an dem trivialen Stationaerpunkt x = (0, 0) haengen
    # bleiben (dort verschwinden Gradient von A und von g, fuer jedes rho).
    penalty_stuck_at_origin = penalty_iter is None and bool(np.linalg.norm(x) < 1e-3)

    r_boundary = float(np.sqrt(V0 / (np.pi * h_max)))
    r = 1.5 * r_boundary
    mu = 1.0
    barrier_iter = None
    for k in range(max_outer):
        r = _barrier_step(r, V0, h_max, mu)
        h = V0 / (np.pi * r ** 2)
        err = np.hypot(r - reference["r"], h - reference["h"])
        if err < target_err:
            barrier_iter = k + 1
            break
        mu /= 3.0

    return {"sqp_iter": sqp_iter, "penalty_iter": penalty_iter, "barrier_iter": barrier_iter,
            "target_err": target_err, "max_outer": max_outer,
            "penalty_stuck_at_origin": penalty_stuck_at_origin}


def describe_iteration_comparison(comp: dict) -> str:
    """Ehrlicher Satz zur Iterationszahl-Messung. Erreicht ein Verfahren die Toleranz in den
    erlaubten Stufen nicht (iter = None), wird das gesagt, statt eine Zahl vorzutäuschen."""
    max_outer = comp.get("max_outer", 40)
    tol = f"{comp['target_err']:.0e}"
    head = (f"SQP braucht **{comp['sqp_iter']}** Iterationen, um auf Maschinengenauigkeit zu "
            "konvergieren. ")
    pen, bar = comp["penalty_iter"], comp["barrier_iter"]

    def _failure(name, stuck_at_origin=False):
        txt = (f"das {name} erreicht den Fehler {tol} innerhalb von {max_outer} äußeren Stufen "
               "gar nicht")
        if stuck_at_origin:
            txt += (" (es bleibt am trivialen Stationärpunkt r = h = 0 hängen, wo das Volumen "
                    "null ist)")
        return txt

    if pen is not None and bar is not None:
        body = (f"Straf- und Barriere-Verfahren (Stück 5) brauchen **{pen}** bzw. **{bar}** "
                f"äußere Stufen, nur um den viel gröberen Fehler {tol} zu unterschreiten")
    elif pen is None and bar is None:
        body = (f"Straf- und Barriere-Verfahren (Stück 5) erreichen den viel gröberen Fehler {tol} "
                f"innerhalb von {max_outer} äußeren Stufen beide nicht")
    elif pen is None:
        body = (f"Das Barriere-Verfahren (Stück 5) braucht **{bar}** äußere Stufen, nur um den "
                f"viel gröberen Fehler {tol} zu unterschreiten; "
                + _failure("Straf-Verfahren", comp.get("penalty_stuck_at_origin", False)))
    else:
        body = (f"Das Straf-Verfahren (Stück 5) braucht **{pen}** äußere Stufen, nur um den "
                f"viel gröberen Fehler {tol} zu unterschreiten; " + _failure("Barriere-Verfahren"))
    return (head + body + " — SQP greift das restringierte Problem direkt an, statt über eine "
            "Parameterfolge zu iterieren.")


def naive_start_pitfall_check() -> dict:
    """Ehrlicher Nebenbefund: ein neutraler Startwert lam0=0 (statt des informierten
    lam0=-2/V0^(1/3)) fuehrt zu einem voruebergehenden Fehlgriff bei der aktiven Menge und
    braucht viel mehr Iterationen; von manchen Startpunkten aus konvergiert SQP sogar zu einem
    mathematisch gueltigen, aber physikalisch unsinnigen KKT-Punkt mit negativem Radius -
    dieselbe Klasse Fund wie Stueck 4s SciPy-Fallstrick, hier ueber eine andere Methode
    (Newton auf der Linearisierung statt SLSQP) reproduziert."""
    V0, h_max = 10.0, 5.0
    good = solver.sqp_solve(V0, h_max)
    naive = solver.sqp_solve(V0, h_max, lam0=0.0)

    bad_start = solver.sqp_solve(V0, h_max, x0=(0.5, 0.5))
    return {"good_n_iter": good.n_iter, "naive_n_iter": naive.n_iter,
            "naive_active_set_switches": len(set(naive.cases)) > 1,
            "bad_start_r": bad_start.r, "bad_start_is_negative": bad_start.r < 0,
            "bad_start_case": bad_start.case}


def gradient_check(eps: float = 1e-6) -> dict:
    x = np.array([1.3, 0.7])
    V0 = 10.0
    results = {}
    for name, func, gradf in (
        ("f", fn.surface_area, fn.grad_surface_area),
        ("g1", lambda xx: fn.volume_constraint(xx, V0), fn.grad_volume_constraint),
    ):
        analytic = gradf(x)
        numeric = np.zeros(2)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            numeric[i] = (func(xp) - func(xm)) / (2 * eps)
        results[f"{name}_max_rel_err"] = float(
            np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), 1e-8)))
    return results
