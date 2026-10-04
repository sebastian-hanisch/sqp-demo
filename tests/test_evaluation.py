import sqp_evaluation as ev


def test_newton_kkt_reduction_is_exact():
    out = ev.newton_kkt_reduction_check()
    assert out["max_abs_err"] < 1e-10


def test_reference_convergence_check_both_presets():
    rows = ev.reference_convergence_check()
    for row in rows:
        assert row["converged"]
        assert row["err"] < 1e-6
        assert row["n_iter"] <= 8


def test_robustness_across_volumes():
    rows = ev.robustness_across_volumes()
    assert all(r["matches_analytic"] for r in rows)
    assert all(r["converged"] for r in rows)


def test_linear_constraint_exactness_is_always_one_iteration():
    rows = ev.linear_constraint_exactness_check()
    assert all(r["n_iter"] == 1 for r in rows)


def test_iteration_count_comparison_sqp_is_much_faster():
    out = ev.iteration_count_comparison()
    assert out["sqp_iter"] < out["penalty_iter"]
    assert out["sqp_iter"] < out["barrier_iter"]


def test_iteration_comparison_small_volume_penalty_not_reached_is_reported_honestly():
    """Regression: für kleines V0 erreicht das Strafverfahren (Start r=h=V0^(1/3)) den Fehler
    nicht, penalty_iter ist None. Unabhängige Referenz: der Ursprung ist für jedes rho ein
    Stationärpunkt der Straffunktion (zentrale Differenzen, eigene Implementierung)."""
    import numpy as np
    V0, h_max = 1.0, 5.0

    def pen(x, rho):
        r, h = x
        return (2 * np.pi * r * r + 2 * np.pi * r * h
                + 0.5 * rho * ((np.pi * r * r * h - V0) ** 2 + max(h - h_max, 0.0) ** 2))

    for rho in (1.0, 1e3, 1e9):
        eps = 1e-6
        g = [(pen(np.eye(2)[i] * eps, rho) - pen(-np.eye(2)[i] * eps, rho)) / (2 * eps)
             for i in range(2)]
        assert np.allclose(g, 0.0, atol=1e-6)

    out = ev.iteration_count_comparison(V0=V0, h_max=h_max)
    assert out["penalty_iter"] is None and out["penalty_stuck_at_origin"]
    assert out["sqp_iter"] <= 5
    text = ev.describe_iteration_comparison(out)
    assert "None" not in text
    assert "nicht" in text and str(out["max_outer"]) in text
    assert f"**{out['barrier_iter']}**" in text


def test_describe_iteration_comparison_all_cases():
    base = dict(sqp_iter=5, target_err=1e-6, max_outer=40, penalty_stuck_at_origin=False)
    both = ev.describe_iteration_comparison(dict(base, penalty_iter=12, barrier_iter=13))
    assert "**12**" in both and "**13**" in both and "None" not in both
    neither = ev.describe_iteration_comparison(dict(base, penalty_iter=None, barrier_iter=None))
    assert "beide nicht" in neither and "None" not in neither
    only_bar = ev.describe_iteration_comparison(dict(base, penalty_iter=12, barrier_iter=None))
    assert "**12**" in only_bar and "Barriere-Verfahren erreicht" in only_bar


def test_naive_start_pitfall_check():
    out = ev.naive_start_pitfall_check()
    assert out["naive_n_iter"] > out["good_n_iter"]
    assert out["bad_start_is_negative"]


def test_gradient_check_below_threshold():
    out = ev.gradient_check()
    assert out["f_max_rel_err"] < 1e-6
    assert out["g1_max_rel_err"] < 1e-6
