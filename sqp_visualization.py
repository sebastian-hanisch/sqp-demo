"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range). Balken-x-Achse
nutzt echte Kategorienamen (Methode), keine numerisch aussehenden Strings - kein
Plotly-Achsentyp-Risiko (feedback_plotly_bar_numeric_labels_vline_mismatch)."""
import numpy as np
import plotly.graph_objects as go

import sqp_functions as fn

COLOR_CONSTRAINT = "#d62728"
COLOR_LIMIT = "#9467bd"
COLOR_OPT = "#2ca02c"
COLOR_SQP = "#1f77b4"


def build_trajectory_figure(V0, h_max, trajectory, reference, r_range=(0.1, 2.5),
                            h_range=(0.1, 6.0), title=""):
    rs = np.linspace(r_range[0], r_range[1], 150)
    hs = np.linspace(h_range[0], h_range[1], 150)
    Z = np.zeros((len(hs), len(rs)))
    for i, hv in enumerate(hs):
        for j, rv in enumerate(rs):
            Z[i, j] = fn.surface_area(np.array([rv, hv]))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=rs, y=hs, z=Z, showscale=False, colorscale="Blues", contours=dict(coloring="fill"),
        opacity=0.5,
    ))
    h_curve = V0 / (np.pi * rs ** 2)
    mask = (h_curve >= h_range[0]) & (h_curve <= h_range[1])
    fig.add_trace(go.Scatter(x=rs[mask], y=h_curve[mask], mode="lines", name="Volumen V=V₀",
                             line=dict(color=COLOR_CONSTRAINT, width=2)))
    fig.add_trace(go.Scatter(x=list(r_range), y=[h_max, h_max], mode="lines",
                             name=f"Höhenlimit h≤{h_max:.2f}",
                             line=dict(color=COLOR_LIMIT, width=2, dash="dash")))
    traj_r = [p[0] for p in trajectory]
    traj_h = [p[1] for p in trajectory]
    fig.add_trace(go.Scatter(x=traj_r, y=traj_h, mode="lines+markers", name="SQP-Trajektorie",
                             line=dict(color=COLOR_SQP, width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[reference["r"]], y=[reference["h"]], mode="markers",
                             name="Referenzlösung", marker=dict(color=COLOR_OPT, size=14,
                                                                symbol="star")))
    fig.update_layout(
        title=title, xaxis=dict(range=list(r_range), fixedrange=True, title="Radius r"),
        yaxis=dict(range=list(h_range), fixedrange=True, title="Höhe h"),
        showlegend=True, height=440, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_iteration_comparison_figure(sqp_iter, penalty_iter, barrier_iter, target_err,
                                      title=None):
    methods = ["SQP", "Straf-Verfahren", "Barriere-Verfahren"]
    values = [sqp_iter, penalty_iter, barrier_iter]
    if title is None:
        title = f"Äußere Iterationen bis Fehler < {target_err:.0e}"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=methods, y=values, marker_color=[COLOR_SQP, "#d62728", "#ff7f0e"],
                         text=values, textposition="outside"))
    fig.update_layout(
        title=title, xaxis=dict(title="Verfahren", fixedrange=True),
        yaxis=dict(title="Äußere Iterationen/Stufen", fixedrange=True),
        showlegend=False, height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
