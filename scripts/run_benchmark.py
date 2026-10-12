"""Run the Stage-A benchmark: three survey policies on several seeded fields,
then the last-metre chain for the belief policy.

    python scripts/run_benchmark.py --seeds 1 2 3 --gif 1

Writes results/survey_metrics.csv, results/chain_metrics.csv,
results/summary_seed<N>.png and (optionally) results/mission_seed<N>.gif.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lastmetre import (Params, Sim, chain_metrics, generate_field,  # noqa: E402
                       neutralisation_chain, survey, survey_metrics)
from lastmetre.mission import mop_up  # noqa: E402
from lastmetre.viz import animate, summary_figure  # noqa: E402

POLICIES = ["naive", "reactive", "belief"]
TITLES = {"naive": "Naive lawn-mower (no safety layer)",
          "reactive": "Reactive detect-and-avoid",
          "belief": "Probabilistic barrier (proposed)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--gif", type=int, default=None, help="seed to animate (belief policy)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = Params()
    srows, crows, logs = [], [], {}
    for seed in a.seeds:
        field = generate_field(seed, p.field_w, p.field_h)
        field.to_json(os.path.join(a.out, f"field_seed{seed}.json"))
        sims = []
        for pol in POLICIES:
            t0 = time.time()
            rec_every = 10 if (pol == "belief" and a.gif == seed) else 0
            sim = Sim(generate_field(seed, p.field_w, p.field_h), p, seed=seed,
                      policy=pol, record_every=rec_every)
            survey(sim)
            m = survey_metrics(sim)
            srows.append(m)
            if pol == "belief":
                records = neutralisation_chain(sim, seed=seed)
                t_chain = sim.t
                mop_up(sim)
                records += neutralisation_chain(sim, seed=seed + 100, second_pass=True)
                sim.phase, sim.note = "DONE", ""
                sim._record(force=True)
                c = chain_metrics(records)
                c.update(seed=seed, survey_coverage=m["coverage"],
                         final_coverage=round(float(sim.bel.observed[sim.bel.in_field].mean()), 3),
                         mop_up_time_s=round(sim.t - t_chain, 1), mission_time_s=round(sim.t, 1),
                         virtual_triggers_total=len(sim.triggers))
                crows.append(c)
                logs[seed] = dict(records=records, events=sim.log)
            sims.append(sim)
            print(f"seed {seed} {pol:9s} {m}  [{time.time() - t0:.0f}s wall]", flush=True)
        summary_figure(sims, os.path.join(a.out, f"summary_seed{seed}.png"),
                       [TITLES[s.policy] for s in sims])
        if a.gif == seed:
            t0 = time.time()
            animate(sims[-1], os.path.join(a.out, f"mission_seed{seed}.gif"),
                    title="last-metre-bench — survey + no-touch placement (all inert, simulated)")
            print(f"gif written [{time.time() - t0:.0f}s]", flush=True)
    for name, rows in (("survey_metrics.csv", srows), ("chain_metrics.csv", crows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(a.out, "chain_log.json"), "w") as f:
        json.dump(logs, f, indent=1, default=str)


if __name__ == "__main__":
    main()
