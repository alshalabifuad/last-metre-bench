"""Monte Carlo study of the no-touch placement margin.

The unit is placed at  r_body_hat + margin + r_unit  from the *estimated*
item centre, on the line towards the robot. A larger margin avoids contact
but can leave the unit too far away to be effective (gap > gap_max).

Two margin rules are compared:
  position-only     margin = k * sigma_pos
  full-uncertainty  margin = k * sqrt(sigma_pos^2 + (sigma_b * r_hat)^2)
                    (a chance constraint that also accounts for the error
                    in the estimated body size)

    python scripts/placement_study.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lastmetre import Params  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASES = {
    "position-only margin, sigma = 5 cm (belief map only)": (0.05, False),
    "position-only margin, sigma = 2 cm (after identification)": (0.02, False),
    "full-uncertainty margin, sigma = 2 cm (chance constraint)": (0.02, True),
}


def trial(rng, n, sigma, k, p, body=(0.06, 0.22), full=False):
    body_r = rng.uniform(*body, n)
    est_err = rng.normal(0, sigma, (n, 2))                  # estimate = truth + error
    body_hat = body_r * (1 + rng.normal(0, p.body_rel_sigma, n))
    u = rng.normal(size=(n, 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)           # approach direction
    s_tot = np.sqrt(sigma ** 2 + (p.body_rel_sigma * body_hat) ** 2) if full else sigma
    target = est_err + u * (body_hat + k * s_tot + p.unit_radius)[:, None]
    gap = np.linalg.norm(target, axis=1) - body_r - p.unit_radius
    contact = gap < 0
    ok = (~contact) & (gap <= p.gap_max)
    return float(contact.mean()), float(ok.mean())


def main(out="results"):
    p = Params()
    rng = np.random.default_rng(0)
    ks = np.linspace(0, 4, 17)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=110)
    for label, (s, full) in CASES.items():
        c, o = zip(*[trial(rng, 20000, s, k, p, full=full) for k in ks])
        rows += [dict(case=label, sigma=s, full=full, k=round(float(k), 2),
                      contact=round(ci, 4), within_spec=round(oi, 4))
                 for k, ci, oi in zip(ks, c, o)]
        axes[0].plot(ks, np.maximum(c, 1e-5), marker="o", ms=3, label=label)
        axes[1].plot(ks, o, marker="o", ms=3, label=label)
    axes[0].set(xlabel="margin factor k", ylabel="P(contact with item)", yscale="log",
                title="Touching the item (must be ~0)")
    axes[1].set(xlabel="margin factor k",
                ylabel=f"P(no contact and gap <= {p.gap_max * 100:.0f} cm)",
                title="Placement within spec")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "placement_margin.png"))
    with open(os.path.join(out, "placement_margin.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["k"] in (2.0, 3.0):
            print(r)


if __name__ == "__main__":
    main()
