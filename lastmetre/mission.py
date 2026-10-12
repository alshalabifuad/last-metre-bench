"""Mission logic: survey, then the per-item "last metre" chain.

Survey rule (belief policy): new ground is only ever entered head-on, with
the boom sensor leading. If a lane is blocked by a suspect, the robot returns
over verified-safe ground and works the same lane from the other end -- the
same practice manual deminers use.

Chain per suspect (belief policy only; everything inert and simulated):
  APPROACH  -> drive on verified-safe ground to a pose within placer reach
  IDENTIFY  -> (abstracted) contactless uncovering + RGB-D identification
  PLACE     -> place the unit next to the item, never touching it
  WITHDRAW  -> move over verified-safe ground to the stand-off distance
  INITIATE  -> a certified human operator decides and initiates (simulated)
  VERIFY    -> old evidence is invalidated; the robot re-senses the spot
"""
import numpy as np

from .grid import dilate_disk
from .planning import (PurePursuit, boustrophedon, densify, dijkstra,
                       extract_path, grid_path, wrap)


# ---------------------------------------------------------------- primitives
def follow(sim, path, timeout=400.0, stop_when=None):
    p = sim.p
    pp = PurePursuit(path, p.lookahead, p.v_max, p.w_max)
    t0, blocked = sim.t, 0.0
    while True:
        if pp.done(sim.pose):
            return "reached"
        v_nom, w_nom = pp.command(sim.pose)
        v = sim.step(v_nom, w_nom)
        if stop_when is not None and stop_when():
            return "stopped"
        blocked = blocked + p.dt if (v_nom > 0.1 and v < 0.03) else 0.0
        if blocked > 1.5:
            return "blocked"
        if sim.t - t0 > timeout:
            return "timeout"


def dwell(sim, seconds, w=0.0):
    for _ in range(int(round(seconds / sim.p.dt))):
        sim.step(0.0, w)


def face(sim, target, tol=0.05, timeout=20.0):
    t0 = sim.t
    while sim.t - t0 < timeout:
        a = wrap(np.arctan2(target[1] - sim.pose[1], target[0] - sim.pose[0]) - sim.pose[2])
        if abs(a) < tol:
            return
        sim.step(0.0, float(np.clip(2.0 * a, -sim.p.w_max, sim.p.w_max)))


def travel_cost(sim):
    """Belief policy: verified-safe ground only (a little slack near the robot).
    Reactive policy: everything except the fixed discs around marks.
    Naive policy: everything."""
    if sim.policy == "belief":
        cls = sim.bel.cell_class()
        # keep ~0.25 m away from the edge of the safe set where possible
        near = dilate_disk((cls > 0).astype(float), 2.5, 0.0) > 0.5
        return np.select([(cls == 0) & ~near, cls == 0, cls == 1],
                         [1.0, 8.0, 40.0], default=np.inf)
    c = np.ones(sim.bel.grid.shape)
    if sim.policy == "reactive":
        X, Y = sim.bel.grid.centers()
        r = sim.p.react_radius + 0.1    # plan with a little clearance ...
        for m in sim.marks:
            c[(X - m[0]) ** 2 + (Y - m[1]) ** 2 < r ** 2] = np.inf
        # ... but let the robot leave the spot where it stopped
        c[(X - sim.pose[0]) ** 2 + (Y - sim.pose[1]) ** 2 < 0.35 ** 2] = 1.0
    return c


def back_off(sim, dist=0.15, margin=None):
    """Recovery: if the robot sits on its safety boundary, turn along the
    barrier gradient and move `dist` into the interior of the safe set."""
    b = sim.barrier()
    margin = margin if margin is not None else (1.0 if sim.policy == "belief" else 0.1)
    if b is None or b[0] > margin or np.linalg.norm(b[1]) < 1e-6:
        return
    g = b[1] / np.linalg.norm(b[1])
    face(sim, sim.pose[:2] + g)
    x0 = sim.pose[:2].copy()
    for _ in range(int(3 * dist / (0.5 * sim.p.v_max * sim.p.dt))):
        if np.hypot(*(sim.pose[:2] - x0)) >= dist:
            break
        sim.step(0.5 * sim.p.v_max, 0.0)


def go_to(sim, goal, stop_when=None):
    if np.hypot(*(sim.pose[:2] - goal)) < 0.2:
        return "reached"
    back_off(sim)
    path = grid_path(sim.bel.grid, travel_cost(sim), sim.pose[:2], goal)
    if path is None:
        return "no path"
    return follow(sim, path, stop_when=stop_when)


# ---------------------------------------------------------------- survey
def _work_lane(sim, a, b, label):
    """Drive one lane head-on; if blocked, work it from the other end."""
    if np.hypot(*(sim.pose[:2] - b)) < np.hypot(*(sim.pose[:2] - a)):
        a, b = b, a
    sim.note = label
    if go_to(sim, a) != "reached":
        sim.event(f"{label}: start not reachable on safe ground; skipped")
        return
    face(sim, b)
    if follow(sim, densify([sim.pose[:2], b])) == "reached":
        return
    stop = sim.pose[:2].copy()
    sim.note = f"{label} blocked -> working it from the other end"
    sim.event(sim.note)
    if go_to(sim, b) == "reached":
        face(sim, stop)
        follow(sim, densify([sim.pose[:2], stop]))


def survey(sim):
    p = sim.p
    sim.phase = "SURVEY"
    wps = boustrophedon(sim.f.w, sim.f.h, p.lane_spacing)
    for k in range(0, len(wps), 2):
        if sim.t > p.survey_time_limit:
            sim.event("survey time limit reached")
            break
        _work_lane(sim, wps[k], wps[k + 1], f"lane {k // 2 + 1}")
    sim.note = ""


def mop_up(sim, min_cells=5, time_limit=900.0):
    """After the chain: revisit only the lanes whose swath still contains
    unsensed ground (pockets that were enclosed by suspects before those were
    neutralised, removed or verified)."""
    p, bel = sim.p, sim.bel
    sim.phase = "MOP-UP SURVEY"
    t0 = sim.t
    _, Y = bel.grid.centers()
    wps = boustrophedon(sim.f.w, sim.f.h, p.lane_spacing)
    for k in range(0, len(wps), 2):
        if sim.t - t0 > time_limit:
            sim.event("mop-up time limit reached")
            break
        yl = wps[k][1]
        gap = bel.in_field & ~bel.observed & (np.abs(Y - yl) <= p.lane_spacing / 2)
        if gap.sum() >= min_cells:
            _work_lane(sim, wps[k], wps[k + 1], f"mop-up lane {k // 2 + 1}")
    sim.note = ""


# ---------------------------------------------------------------- chain
def _nearest_item(field, xy, radius=0.5):
    best = min(field.items, key=lambda it: np.hypot(it.x - xy[0], it.y - xy[1]))
    return best if np.hypot(best.x - xy[0], best.y - xy[1]) < radius else None


def _safe_tree(sim):
    g = sim.bel.grid
    cls = sim.bel.cell_class()
    near = dilate_disk((cls > 0).astype(float), 2.5, 0.0) > 0.5
    cost = np.select([(cls == 0) & ~near, cls == 0], [1.0, 8.0], default=np.inf)
    start = tuple(int(v) for v in g.index(*sim.pose[:2]))
    cost[start] = 1.0
    dist, parent = dijkstra(cost, start)
    return dist, parent


def _path_to_best(sim, score):
    """Path over verified-safe ground to the cell minimising `score(dist, X, Y)`."""
    g = sim.bel.grid
    dist, parent = _safe_tree(sim)
    X, Y = g.centers()
    sc = np.where(np.isfinite(dist), score(dist * g.res, X, Y), np.inf)
    if not np.isfinite(sc).any():
        return None
    goal = np.unravel_index(np.argmin(sc), sc.shape)
    cells = extract_path(parent, goal)
    xs, ys = g.center(*np.array(cells).T)
    pts = np.stack([xs, ys], 1)
    pts[0] = sim.pose[:2]
    return pts


def _resense_spot(sim, xh, approach_pose):
    """Invalidate stale evidence around xh (the world changed there) and
    re-sense it head-on from the approach side. Returns (P_max, fraction seen)
    within 0.25 m of xh."""
    p = sim.p
    sim.bel.reset_region(xh[0], xh[1], p.verify_reset_radius)
    sim.bel.refresh()
    d0 = approach_pose - xh
    d0 /= np.linalg.norm(d0)
    stand = xh + d0 * (p.sensor_offset + p.verify_reset_radius + 0.2)
    go_to(sim, stand)
    face(sim, xh)
    creep_to = xh + d0 * (p.sensor_offset - 0.25)
    follow(sim, densify([sim.pose[:2], creep_to]), timeout=60.0)
    face(sim, xh)
    X, Y = sim.bel.grid.centers()
    near = (X - xh[0]) ** 2 + (Y - xh[1]) ** 2 <= 0.25 ** 2
    return float(sim.bel.P_obj[near].max()), float(sim.bel.observed[near].mean())


def process_suspect(sim, s, rng, force_device=False):
    p = sim.p
    xh = s["mean"]
    rec = dict(estimate=np.round(xh, 2).tolist(), t_start=round(sim.t, 1))
    truth = _nearest_item(sim.f, xh)          # evaluation metadata only; never used for control
    rec["truth_kind"] = truth.kind if truth is not None else "none"
    if force_device:
        rec["second_look"] = True

    # 1) approach on verified-safe ground, within placer reach
    sim.phase, sim.note = "APPROACH", f"suspect at ({xh[0]:.1f}, {xh[1]:.1f})"
    reach = p.boom_reach - 0.4
    path = _path_to_best(sim, lambda d, X, Y: np.where(
        np.hypot(X - xh[0], Y - xh[1]) <= reach, d + 3.0 * np.hypot(X - xh[0], Y - xh[1]), np.inf))
    if path is None:
        rec["result"] = "unreachable on verified-safe ground"
        sim.event(rec["result"])
        return rec
    follow(sim, path)
    face(sim, xh)
    approach_pose = sim.pose[:2].copy()

    # 2) uncover + identify (abstracted in Stage A)
    sim.phase, sim.note = "UNCOVER + IDENTIFY", "contactless uncovering, RGB-D identification"
    dwell(sim, 3.0)
    item = _nearest_item(sim.f, xh)
    if item is None:
        rec["result"] = "false alarm (nothing found)"
        sim.event(rec["result"])
        return rec
    if force_device:
        # something is still there after the operator handled the spot: treat as a device
        call = item.kind if item.kind != "clutter" else "ied_ap"
    else:
        call = item.kind if rng.random() < p.p_identify else (
            "clutter" if item.kind != "clutter" else "ied_ap")
    rec.update(item=item.id, kind=item.kind, call=call)
    if call == "clutter":
        rec["result"] = ("DEVICE MISSED AT IDENTIFICATION" if item.kind != "clutter"
                         else "clutter -> handed to operator, no unit placed")
        sim.event(rec["result"])
        # The operator removes what was called clutter. A device wrongly called
        # clutter is NOT removed (the human check is the backstop). Either way the
        # old evidence is stale: re-sense the spot now (this keeps the ground
        # around it usable and gives an immediate second look).
        if item.kind == "clutter":
            item.neutralised = True
        sim.phase, sim.note = "RE-SENSE AFTER REMOVAL", "old evidence invalidated"
        pmax, _ = _resense_spot(sim, xh, approach_pose)
        rec["after_removal_pmax"] = round(pmax, 3)
        if pmax > 0.5:
            rec["redetected"] = True
            sim.event("still detected after removal -> will be treated as a device")
        return rec
    x_cam = np.array([item.x, item.y]) + rng.normal(0, p.camera_sigma, 2)
    body_hat = item.body_r * (1 + rng.normal(0, p.body_rel_sigma))

    # 3) place the unit next to the item without touching it
    sim.phase, sim.note = "PLACE", f"no-touch placement, margin k={p.k_sigma:g}"
    u = sim.pose[:2] - x_cam
    u /= np.linalg.norm(u)
    target = x_cam + u * (body_hat + p.k_sigma * p.camera_sigma + p.unit_radius)
    dist_t = np.linalg.norm(target - sim.pose[:2])
    if dist_t > p.boom_reach:
        rec["result"] = f"target out of reach ({dist_t:.2f} m)"
        sim.event(rec["result"])
        return rec
    base = sim.pose[:2] + 0.35 * (target - sim.pose[:2]) / dist_t
    contact = False
    for a in np.linspace(0, 1, 30):
        sim.ee = base + a * (target - base)
        if np.hypot(*(sim.ee - (item.x, item.y))) < item.body_r + p.unit_radius:
            contact = True
        sim.step(0.0, 0.0)
    gap = np.hypot(*(target - (item.x, item.y))) - item.body_r - p.unit_radius
    sim.units.append(target.copy())
    for a in np.linspace(1, 0, 10):
        sim.ee = base + a * (target - base)
        sim.step(0.0, 0.0)
    sim.ee = None
    rec.update(contact=contact, gap=round(float(gap), 3),
               placed_ok=bool(not contact and 0.0 <= gap <= p.gap_max))
    if contact:
        sim.contacts.append(item.id)
        sim.event(f"CONTACT with item {item.id} during placement")

    # 4) withdraw over verified-safe ground to the stand-off distance
    sim.phase, sim.note = "WITHDRAW", f"to stand-off {p.withdraw_distance:g} m (sim value)"
    R = p.withdraw_distance
    path = _path_to_best(sim, lambda d, X, Y: np.where(
        np.hypot(X - xh[0], Y - xh[1]) >= R, d, np.inf))
    if path is not None:
        follow(sim, path)
    rec["withdraw_dist"] = round(float(np.hypot(*(sim.pose[:2] - xh))), 2)

    # 5) human decision and initiation (simulated outcome)
    sim.phase, sim.note = "OPERATOR DECIDES + INITIATES", "certified operator, remote (simulated)"
    dwell(sim, 3.0)
    success = rng.random() < p.p_neutralise
    if success:
        item.neutralised = True
    sim.units = [q for q in sim.units if np.hypot(*(q - target)) > 1e-6]
    rec["neutralised_truth"] = success

    # 6) verify: invalidate stale evidence, re-sense the spot head-on
    sim.phase, sim.note = "VERIFY", "old evidence invalidated, re-sensing"
    pmax, seen = _resense_spot(sim, xh, approach_pose)
    verdict = ("residual hazard -> re-task" if pmax > 0.5 else
               "verified neutralised" if seen > 0.9 and pmax < 0.05 else "inconclusive")
    rec.update(verdict=verdict, verify_pmax=round(pmax, 3),
               verdict_correct=((verdict == "verified neutralised") == success)
               if verdict != "inconclusive" else None,
               t_end=round(sim.t, 1))
    sim.event(f"item {item.id}: {verdict}")
    rec["result"] = "processed"
    return rec


def neutralisation_chain(sim, seed=0, second_pass=False):
    """Process every current suspect. On the second pass (after the mop-up),
    spots already handled are skipped, except where something was called
    clutter and is still detected after the operator handled it -- that is
    re-processed as a device (safety net for identification errors)."""
    rng = np.random.default_rng(20_000 + seed)
    records = []
    if not hasattr(sim, "processed"):
        sim.processed = []
    pending = []
    for s in sim.bel.suspects():
        near = [r for xy, r in sim.processed if np.hypot(*(s["mean"] - xy)) < 0.5]
        if not near:
            pending.append((s, False))
        elif second_pass and any(r.startswith(("clutter", "DEVICE MISSED")) for r in near):
            pending.append((s, True))
    while pending:
        pending.sort(key=lambda sf: np.hypot(*(sf[0]["mean"] - sim.pose[:2])))
        s, forced = pending.pop(0)
        rec = process_suspect(sim, s, rng, force_device=forced)
        rec["pass"] = 2 if second_pass else 1
        sim.processed.append((s["mean"].copy(), rec.get("result", "")))
        records.append(rec)
    sim.phase, sim.note = "DONE", ""
    sim._record(force=True)
    return records
