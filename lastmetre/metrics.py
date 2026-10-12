"""Evaluation against ground truth (never visible to the robot)."""
import numpy as np


def survey_metrics(sim, match_radius=0.5):
    bel, f = sim.bel, sim.f
    sus = bel.suspects()
    hz = f.hazards()
    detected = {it.id for it in hz
                if any(np.hypot(*(s["mean"] - (it.x, it.y))) < match_radius for s in sus)}
    clutter_alarms = sum(1 for s in sus if any(
        it.kind == "clutter" and np.hypot(*(s["mean"] - (it.x, it.y))) < match_radius
        for it in f.items))
    false_alarms = sum(1 for s in sus if all(
        np.hypot(*(s["mean"] - (it.x, it.y))) >= match_radius for it in f.items))
    area_ha = f.w * f.h / 1e4
    return dict(
        policy=sim.policy, seed=f.seed,
        coverage=round(float(bel.observed[bel.in_field].mean()), 3),
        hazards=len(hz), detected=len(detected),
        pod=round(len(detected) / max(1, len(hz)), 3),
        clutter_alarms=clutter_alarms,
        false_alarms_per_ha=round(false_alarms / area_ha, 1),
        virtual_triggers=len(sim.triggers),
        time_s=round(sim.t, 1), distance_m=round(sim.dist, 1),
        min_barrier=None if not np.isfinite(sim.min_h) else round(float(sim.min_h), 3),
    )


def chain_metrics(records):
    proc = [r for r in records if r.get("result") == "processed"]
    return dict(
        suspects=len(records),
        processed=len(proc),
        placed_ok=sum(bool(r.get("placed_ok")) for r in proc),
        contacts=sum(bool(r.get("contact")) for r in proc),
        verified_correct=sum(r.get("verdict_correct") is True for r in proc),
        verified_wrong=sum(r.get("verdict_correct") is False for r in proc),
        inconclusive=sum(r.get("verdict") == "inconclusive" for r in proc),
        missed_at_identification=sum(r.get("result") == "DEVICE MISSED AT IDENTIFICATION"
                                     for r in records),
        false_alarms_resolved=sum(r.get("result", "").startswith(("false alarm", "clutter"))
                                  for r in records),
        # a device first called clutter, re-detected after the spot was handled and
        # re-flagged as a hazard (processed, or left marked as an unreachable hazard)
        recovered_on_second_look=sum(bool(r.get("second_look"))
                                     and r.get("truth_kind") not in ("clutter", "none")
                                     for r in records),
    )
