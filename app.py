"""SQP — Sequential Quadratic Programming

Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Stück 5 loeste das restringierte Problem ueber eine FOLGE unrestringierter Probleme. SQP loest es
DIREKT: an jedem Punkt eine kleine quadratische Naeherung (QP) des Originalproblems, deren Loesung
der naechste Schritt ist. Mit der exakten Hesse-Matrix der Lagrange-Funktion ist das mathematisch
identisch zu Newton's Verfahren auf dem KKT-System (Stück 4) - keine Naeherung, eine exakte
Aequivalenz.

Lauffähig mit: streamlit run app.py
"""
import streamlit as st

import sqp_constants as C
import sqp_evaluation as ev
import sqp_presets as pr
import sqp_reference as ref
import sqp_visualization as viz

st.set_page_config(page_title="SQP", layout="wide")


@st.cache_data(show_spinner=False)
def _run(V0, h_max):
    out = ev.analyse(ev.Settings(V0=V0, h_max=h_max))
    s = out["solution"]
    return {"r": s.r, "h": s.h, "lam": s.lam, "mu": s.mu, "case": s.case, "n_iter": s.n_iter,
            "converged": s.converged, "trajectory": s.trajectory, "f_val": out["f_val"]}


@st.cache_data(show_spinner=False)
def _reference(V0, h_max):
    return ref.reference_solution(V0, h_max)


@st.cache_data(show_spinner=False)
def _newton_kkt_reduction():
    return ev.newton_kkt_reduction_check()


@st.cache_data(show_spinner=False)
def _reference_convergence():
    return ev.reference_convergence_check()


@st.cache_data(show_spinner=False)
def _robustness():
    return ev.robustness_across_volumes()


@st.cache_data(show_spinner=False)
def _linear_exactness():
    return ev.linear_constraint_exactness_check()


@st.cache_data(show_spinner=False)
def _iteration_comparison(V0, h_max):
    return ev.iteration_count_comparison(V0=V0, h_max=h_max)


@st.cache_data(show_spinner=False)
def _naive_pitfall():
    return ev.naive_start_pitfall_check()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


st.title("🧮 SQP — Sequential Quadratic Programming")
st.markdown(
    "Stück 5 löste die restringierte Zylinder-Aufgabe über eine **Folge unrestringierter "
    "Probleme**. SQP löst sie **direkt**: an jedem Punkt eine kleine quadratische Näherung (QP) "
    "des Originalproblems, deren Lösung der nächste Schritt ist. Mit der exakten Hesse-Matrix der "
    "Lagrange-Funktion ist das mathematisch **identisch** zu Newtons Verfahren auf dem "
    "KKT-System (Stück 4) — keine Näherung, eine exakte Äquivalenz."
)
st.caption(
    "Stück 6 der 'Nichtlineare Optimierung'-Reihe. Folgestücke: "
    "Innere-Punkte-Verfahren (Stück 7), Stochastische Gradientenverfahren (Stück 8)."
)

with st.expander("So funktioniert SQP", expanded=True):
    st.markdown(
        "1. An Punkt $x_k$ werden die Nebenbedingungen linearisiert und eine QP gelöst: "
        "$\\min_d \\nabla f^\\top d+\\frac12 d^\\top H_L d$ unter "
        "$\\nabla g_1^\\top d=-g_1$, $\\nabla g_2^\\top d\\le-g_2$, mit $H_L$ der Hesse-Matrix "
        "der Lagrange-Funktion.\n"
        "2. Bei nur einer Ungleichung hat auch die QP nur zwei Fälle (inaktiv/aktiv) — dieselbe "
        "Enumerations-Idee wie Stück 4, hier je Iteration neu geprüft, weil sich die "
        "Linearisierung mit $x_k$ ändert.\n"
        "3. $x_{k+1}=x_k+d$. Mit der exakten Hesse-Matrix und einer bereits korrekten aktiven "
        "Menge ist das exakt ein Newton-Schritt auf dem KKT-System aus Stück 4 (siehe 📐)."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                  use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    V0 = st.slider("Volumen V₀", C.V0_MIN, C.V0_MAX, ss["V0"], step=1.0, key="widget_V0",
                   on_change=pr.store_from_widget, args=("V0",))
    ss["V0"] = V0
    h_max = st.slider("Höhenlimit h_max", C.H_MAX_MIN, C.H_MAX_MAX, ss["h_max"], step=0.05,
                      key="widget_h_max", on_change=pr.store_from_widget, args=("h_max",))
    ss["h_max"] = h_max

pr.sync_query_params(dict(V0=V0, h_max=h_max))

out = _run(V0, h_max)
reference = _reference(V0, h_max)

st.markdown("---")
st.subheader("🎯 Direkte Konvergenz, wenige Iterationen")
r_star_now = (V0 / (2 * 3.141592653589793)) ** (1 / 3)
pad_r = max(2.5, r_star_now * 2.2)
pad_h = max(6.0, 2 * r_star_now * 2.2, h_max * 1.3)
col_left, col_right = st.columns([3, 2])
with col_left:
    fig = viz.build_trajectory_figure(V0, h_max, out["trajectory"], reference,
                                      r_range=(0.05, pad_r), h_range=(0.05, pad_h),
                                      title=f"Volumen={V0:.0f}, Höhenlimit={h_max:.2f}")
    st.plotly_chart(fig, key=f"traj_{V0}_{h_max}", use_container_width=True)
with col_right:
    st.metric("Fall", "Höhenlimit inaktiv" if out["case"] == "inactive"
              else "Höhenlimit bindet")
    m1, m2 = st.columns(2)
    m1.metric("Radius r*", f"{out['r']:.4f}")
    m2.metric("Höhe h*", f"{out['h']:.4f}")
    m3, m4 = st.columns(2)
    m3.metric("λ (Volumen)", f"{out['lam']:.4f}")
    m4.metric("μ (Höhenlimit)", f"{out['mu']:.4f}")
    st.metric("SQP-Iterationen", out["n_iter"])
    st.caption(
        "Zum Vergleich: Straf-/Barriere-Verfahren (Stück 5) brauchen für dieselbe Genauigkeit "
        "im gemessenen Reglerraster 10 bis 25 (Straf) bzw. 10 bis 17 (Barriere) äußere Stufen — oder erreichen sie gar nicht (siehe 🎯 unten)."
    )

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: Iterationszahl vs. Straf-/Barriere-Verfahren")
comp = _iteration_comparison(V0, h_max)
col_a, col_b = st.columns([2, 3])
with col_a:
    st.plotly_chart(
        viz.build_iteration_comparison_figure(comp["sqp_iter"], comp["penalty_iter"],
                                              comp["barrier_iter"], comp["target_err"],
                                              max_outer=comp["max_outer"]),
        key=f"comp_{V0}_{h_max}", use_container_width=True)
with col_b:
    st.markdown(ev.describe_iteration_comparison(comp))

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Voller Newton-Schritt konvergiert ohne Sicherung | Bei größeren/steiferen Problemen kann "
    "der volle Schritt divergieren — ein Line-Search oder eine Merit-Funktion wird nötig | "
    "Innere-Punkte-Verfahren (Stück 7) haben eine eingebaute Sicherung |\n"
    "| Startwert für λ ist informiert (aus Dimensionsanalyse) | Ein neutraler Start (λ₀=0) "
    "verursacht einen vorübergehenden Fehlgriff bei der aktiven Menge und viel mehr Iterationen "
    "(siehe 📐) | Informierte Startwerte (hier bereits verwendet) |\n"
    "| Nur eine Ungleichung | Mehr Ungleichungen brauchen mehr als zwei Fälle in der "
    "Enumeration | Innere-Punkte-Verfahren skalieren systematischer |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**QP-Teilaufgabe:** $\min_d \nabla f(x_k)^\top d+\frac12 d^\top H_L d$ unter
$\nabla g_1(x_k)^\top d=-g_1(x_k)$, $\nabla g_2(x_k)^\top d\le-g_2(x_k)$,
$H_L=\nabla^2f+\lambda\nabla^2g_1$.
"""
    )
    nk = _newton_kkt_reduction()
    st.metric("Max. Abweichung SQP-Schritt vs. Newton-KKT-Schritt", f"{nk['max_abs_err']:.2e}")
    st.caption("Bei fixierter aktiver Menge ist der SQP-Schritt exakt identisch mit einem "
              "ungedämpften Newton-Schritt auf dem KKT-System aus Stück 4.")

    st.markdown("**Referenz-Konvergenz** (beide Presets, informierter Start):")
    rc = _reference_convergence()
    for row in rc:
        st.caption(f"h_max={row['h_max']} ({row['case']}): {row['n_iter']} Iterationen, "
                  f"Fehler={row['err']:.2e}, konvergiert: {row['converged']}")

    st.markdown("**Robustheit über verschiedene Volumina V₀** (informierter Startwert):")
    rob = _robustness()
    rob_ok = sum(1 for r in rob if r["matches_analytic"])
    st.metric("Trifft die analytische Lösung", f"{rob_ok}/{len(rob)} getestete V₀-Werte")

    st.markdown(
        "**Rein linear restringiertes Hilfsproblem** (kein Vehikel B): "
        "$\\min (x-2)^2+(y-3)^2$ unter $x+y=4$ — Ziel quadratisch, Nebenbedingung bereits "
        "linear, die Linearisierung macht KEINEN Fehler:"
    )
    lc = _linear_exactness()
    lc_ok = sum(1 for r in lc if r["n_iter"] == 1)
    st.metric("Konvergiert in genau 1 Iteration", f"{lc_ok}/{len(lc)} getestete Startpunkte")

    st.markdown(
        "**Ehrlicher Nebenbefund** — ein neutraler Startwert kann fehlschlagen: von "
        "$\\lambda_0=0$ statt dem informierten $\\lambda_0=-2/V_0^{1/3}$ braucht SQP "
        "vorübergehend einen falschen Fall der aktiven Menge und viel mehr Iterationen; von "
        "manchen Startpunkten aus konvergiert SQP sogar zu einem mathematisch gültigen, aber "
        "physikalisch unsinnigen KKT-Punkt mit negativem Radius — dieselbe Klasse Fund wie "
        "Stück 4s SciPy-Fallstrick, hier über eine andere Methode reproduziert:"
    )
    pit = _naive_pitfall()
    p1, p2 = st.columns(2)
    p1.metric("Informierter Start", f"{pit['good_n_iter']} Iterationen")
    p2.metric("Neutraler Start (λ₀=0)", f"{pit['naive_n_iter']} Iterationen")
    st.caption(f"Startpunkt (0,5, 0,5) konvergiert zu r={pit['bad_start_r']:.4f} "
              f"(negativ: {pit['bad_start_is_negative']}), ebenfalls ein gültiger KKT-Punkt "
              f"(Fall: {pit['bad_start_case']}).")

    grad_err = _gradient_check()
    g1, g2 = st.columns(2)
    g1.metric("Gradienten-Check f", f"{grad_err['f_max_rel_err']:.1e}")
    g2.metric("Gradienten-Check g₁", f"{grad_err['g1_max_rel_err']:.1e}")

    st.markdown(
        "**Literatur:** Wilson, R. B. (1963). *A Simplicial Algorithm for Concave Programming.* "
        "PhD-Thesis, Harvard University. Nocedal, J. & Wright, S. J. (2006). "
        "*Numerical Optimization* (2. Aufl.), Kap. 18."
    )
    st.caption(
        "Implementiert in `sqp_functions.py` (Zielfunktion, Nebenbedingungen), `sqp_qp.py` "
        "(QP-Teilaufgabe), `sqp_solver.py` (äußere Iteration), `sqp_reference.py` "
        "(Referenzlösung), `sqp_evaluation.py` (Korrektheits-Kette), `sqp_visualization.py` "
        "(Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html)."
)
