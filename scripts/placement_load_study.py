"""C2 prototype: keep the load reaching a buried object below a threshold while
lowering a unit onto the soil next to it (see lastmetre/contact.py).

    python scripts/placement_load_study.py --n 400

Writes results/placement_load.csv and results/placement_load.png.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lastmetre.contact import ContactParams, draw_trials, gamma, plan_offset, simulate  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [  # (label, controller, planner, top speed m/s)
    ("position control, 15 cm/s", "position", "quantile", 0.15),
    ("guarded move, 15 cm/s", "guarded", "quantile", 0.15),
    ("guarded move, 3 cm/s (cautious)", "guarded", "quantile", 0.03),
    ("barrier, worst-case stiffness, 15 cm/s", "robust", "quantile", 0.15),
    ("barrier, adaptive stiffness, 15 cm/s", "adaptive", "quantile", 0.15),
    ("adaptive barrier + nominal planner", "adaptive", "nominal", 0.15),
    ("adaptive barrier + 99.9 % planner", "adaptive", "q999", 0.15),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    p = ContactParams()
    trials, rng = draw_trials(a.n, p, seed=0)
    plans = {}
    for kind in ("quantile", "nominal", "q999"):
        r, g = zip(*[plan_offset(z, p, np.random.default_rng(i), nominal=(kind == "nominal"),
                                 quantile=0.999 if kind == "q999" else None)
                     for i, z in enumerate(trials["z_hat"])])
        plans[kind] = (np.array(r), np.array(g))
    stiff = int(np.argmax(trials["k"] * (trials["e_s"] < 0)))   # a stiff-soil trial to plot
    rows, traces = [], {}
    for label, ctrl, plan, vmax in RUNS:
        t0 = time.time()
        r_cmd, g_bar = plans[plan]
        res = simulate(ctrl, trials, r_cmd, g_bar, p, record=stiff, v_max=vmax)
        traces[label] = res["trace"]
        viol = res["L_ratio"] > 1.0
        ctrl_viol = res["F_ratio"] > 1.0          # controller exceeded its own force limit
        plan_miss = res["g_true"] > res["g_bar"]  # true transfer above the planner's bound
        rows.append(dict(
            method=label, trials=a.n,
            violations=int(viol.sum()), violation_rate=round(float(viol.mean()), 3),
            controller_exceeded_force_limit=int(ctrl_viol.sum()),
            planner_bound_exceeded=int(plan_miss.sum()),
            worst_load_over_threshold=round(float(res["L_ratio"].max()), 2),
            p95_load_over_threshold=round(float(np.quantile(res["L_ratio"], 0.95)), 2),
            completed=int(res["completed"].sum()),
            median_time_s=round(float(np.nanmedian(res["t_release"])), 2),
            p95_time_s=round(float(np.nanquantile(res["t_release"], 0.95)), 2),
            mean_offset_cm=round(100 * float(r_cmd.mean()), 1)))
        print(rows[-1], f"[{time.time() - t0:.0f}s]", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "placement_load.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), dpi=110)
    for label, tr in traces.items():
        if len(tr):
            ax[0].plot(tr[:, 0], tr[:, 3], lw=1.2, label=label)
    ax[0].axhline(1.0, color="red", ls="--", lw=1)
    ax[0].set(xlabel="time (s)", ylabel="load on object / threshold",
              title=f"One stiff-soil trial (k = {trials['k'][stiff]:.0f})", xlim=(0, 4))
    ax[0].legend(fontsize=6.5)
    names = [r["method"] for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(x, [100 * r["violation_rate"] for r in rows], color="#e63946")
    ax[1].set_xticks(x, names, rotation=25, ha="right", fontsize=7)
    ax[1].set(ylabel="% trials with load > threshold", title=f"Safety ({a.n} trials)")
    ax[2].bar(x, [r["median_time_s"] for r in rows], color="#2a9d8f")
    ax[2].set_xticks(x, names, rotation=25, ha="right", fontsize=7)
    ax[2].set(ylabel="median time to release (s)", title="Speed")
    for axx in ax:
        axx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "placement_load.png"))


if __name__ == "__main__":
    main()
