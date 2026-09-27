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


def test_naive_start_pitfall_check():
    out = ev.naive_start_pitfall_check()
    assert out["naive_n_iter"] > out["good_n_iter"]
    assert out["bad_start_is_negative"]


def test_gradient_check_below_threshold():
    out = ev.gradient_check()
    assert out["f_max_rel_err"] < 1e-6
    assert out["g1_max_rel_err"] < 1e-6
