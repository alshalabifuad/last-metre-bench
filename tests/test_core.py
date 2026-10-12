import numpy as np
import pytest

from lastmetre import HazardBelief, Params, Sim, generate_field
from lastmetre.grid import Grid, dilate_disk, dilate_max, disk_offsets
from lastmetre.planning import boustrophedon, densify, grid_path


def test_field_is_deterministic_and_separated():
    a, b = generate_field(7), generate_field(7)
    assert [(i.x, i.y, i.kind) for i in a.items] == [(i.x, i.y, i.kind) for i in b.items]
    pts = np.array([[i.x, i.y] for i in a.items])
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(len(pts), 1)].min() >= 2.0


def test_fast_dilation_matches_reference():
    rng = np.random.default_rng(0)
    a = rng.random((40, 50))
    for r in (1.0, 2.5, 5.0, 8.0):
        assert np.array_equal(dilate_max(a, disk_offsets(r), 0.0), dilate_disk(a, r, 0.0))


def test_belief_clears_empty_ground_and_flags_items():
    p = Params()
    f = generate_field(3)
    bel = HazardBelief(p, f.w, f.h)
    it = f.hazards()[0]
    # sweep a dense set of points over an item and over empty ground
    xs = np.linspace(it.x - 0.6, it.x + 0.6, 25)
    rng = np.random.default_rng(1)
    for _ in range(4):
        pts = np.stack([xs, np.full_like(xs, it.y)], 1)
        bel.update(pts, f.signal_at(pts, p.sensor_height, rng, p.sensor_noise))
    bel.refresh()
    i, j = bel.grid.index(it.x, it.y)
    assert bel.P_obj[i, j] > 0.5
    # unsensed ground keeps the prior and is NOT safe for the robot
    assert bel.cell_class()[bel.grid.index(f.w / 2 + 0.05, f.h - 0.5)] != 0


def test_unsensed_ground_is_forbidden_and_perimeter_is_safe():
    p = Params()
    bel = HazardBelief(p, 12.0, 12.0)
    h_out, _ = bel.barrier(-1.5, 6.0)      # cleared perimeter
    h_in, _ = bel.barrier(6.0, 6.0)        # never sensed
    assert h_out > 0 > h_in


def test_grid_path_avoids_blocked_cells():
    g = Grid(0, 0, 5, 5, 0.1)
    cost = np.ones(g.shape)
    cost[:, 24:26] = np.inf
    cost[45:, 24:26] = 1.0                 # a gap at the top
    path = grid_path(g, cost, (1.0, 1.0), (4.0, 1.0))
    assert path is not None
    i, j = g.index(path[:, 0], path[:, 1])
    assert np.isfinite(cost[i, j]).all()


@pytest.mark.slow
def test_belief_policy_never_triggers_on_a_seeded_field():
    from lastmetre import survey
    p = Params()
    p.survey_time_limit = 400
    sim = Sim(generate_field(2), p, seed=2, policy="belief", record_every=0)
    survey(sim)
    assert len(sim.triggers) == 0
    assert sim.min_h > -0.2               # discrete-time CBF: tiny violations only


def test_boustrophedon_lanes_cover_field():
    w = boustrophedon(12, 12, 1.4)
    ys = sorted(set(np.round(w[:, 1], 3)))
    assert ys[0] - 0.9 <= 0 and ys[-1] + 0.9 >= 12
    assert len(densify(w, 0.1)) > 100
