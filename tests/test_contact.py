import numpy as np

from lastmetre.contact import ContactParams, draw_trials, gamma, plan_offset, simulate


def test_boussinesq_transfer_decays_with_offset():
    p = ContactParams()
    g = gamma(np.array([0.05, 0.10, 0.20, 0.40]), 0.05, p)
    assert np.all(np.diff(g) < 0) and g.max() <= 1.0


def test_planner_quantile_is_more_conservative_than_nominal():
    p = ContactParams()
    r_q, g_q = plan_offset(0.05, p, np.random.default_rng(0))
    r_n, _ = plan_offset(0.05, p, np.random.default_rng(0), nominal=True)
    assert r_q > r_n
    assert g_q * p.W * p.headroom <= p.theta + 1e-12


def test_barrier_never_exceeds_its_force_limit():
    """Whatever the (bounded) soil stiffness and surface error, the stopping-distance
    barrier keeps the force below the limit the planner gave it."""
    p = ContactParams()
    trials, _ = draw_trials(60, p, seed=3)
    r, g = zip(*[plan_offset(z, p, np.random.default_rng(i)) for i, z in enumerate(trials["z_hat"])])
    for ctrl in ("robust", "adaptive"):
        res = simulate(ctrl, trials, np.array(r), np.array(g), p, v_max=0.15)
        assert res["completed"].all()
        assert res["F_ratio"].max() <= 1.0


def test_fast_guarded_move_does_exceed_the_limit():
    p = ContactParams()
    trials, _ = draw_trials(60, p, seed=3)
    r, g = zip(*[plan_offset(z, p, np.random.default_rng(i)) for i, z in enumerate(trials["z_hat"])])
    res = simulate("guarded", trials, np.array(r), np.array(g), p, v_max=0.15)
    assert res["F_ratio"].max() > 1.0
