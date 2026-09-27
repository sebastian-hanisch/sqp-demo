"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet."""
import sqp_evaluation as ev


def test_claim_sqp_step_equals_newton_kkt_step_exactly():
    out = ev.newton_kkt_reduction_check()
    assert out["max_abs_err"] < 1e-10


def test_claim_sqp_converges_in_at_most_5_iterations_for_both_presets():
    rows = ev.reference_convergence_check()
    for row in rows:
        assert row["n_iter"] <= 5
        assert row["converged"]


def test_claim_linear_constraint_problem_converges_in_exactly_one_iteration():
    rows = ev.linear_constraint_exactness_check()
    assert all(r["n_iter"] == 1 for r in rows)


def test_claim_sqp_needs_far_fewer_iterations_than_penalty_or_barrier():
    out = ev.iteration_count_comparison()
    assert out["sqp_iter"] <= 5
    assert out["penalty_iter"] >= 10
    assert out["barrier_iter"] >= 10


def test_claim_naive_lambda_start_causes_active_set_misfire():
    out = ev.naive_start_pitfall_check()
    assert out["naive_active_set_switches"]
    assert out["naive_n_iter"] > out["good_n_iter"]


def test_claim_bad_start_converges_to_unphysical_negative_radius():
    out = ev.naive_start_pitfall_check()
    assert out["bad_start_is_negative"]


def test_claim_gradient_check_below_1e_minus_6():
    out = ev.gradient_check()
    assert out["f_max_rel_err"] < 1e-6
    assert out["g1_max_rel_err"] < 1e-6
