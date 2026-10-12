"""Ground truth: a seeded field of inert surrogate items.

The robot never reads this module directly. Only the simulated sensor
(`signal_at`) and the evaluator (`footprint_hits`, metrics) use it.
"""
from dataclasses import dataclass, asdict
import json

import numpy as np

# Generic surrogate classes. Numbers describe sensor contrast and geometry
# of laboratory surrogates, not any real device.
KINDS = {
    # victim-operated improvised device, low metal content
    "ied_ap": dict(strength=(0.15, 0.60), depth=(0.00, 0.15),
                   body_r=(0.06, 0.12), hazard_r=(0.20, 0.40)),
    # larger improvised device
    "ied_at": dict(strength=(0.50, 1.00), depth=(0.05, 0.15),
                   body_r=(0.12, 0.22), hazard_r=(0.30, 0.50)),
    # harmless metal clutter (fragments, cans)
    "clutter": dict(strength=(0.15, 1.00), depth=(0.00, 0.05),
                    body_r=(0.02, 0.08), hazard_r=(0.0, 0.0)),
}


@dataclass
class Item:
    id: int
    kind: str
    x: float
    y: float
    depth: float
    strength: float
    body_r: float
    hazard_r: float
    neutralised: bool = False

    @property
    def hazardous(self):
        return self.kind != "clutter" and not self.neutralised


class Field:
    def __init__(self, w, h, items, seed=None):
        self.w, self.h, self.items, self.seed = w, h, items, seed

    # ---- simulated physics -------------------------------------------
    def signal_at(self, pts, head_height, rng=None, noise=0.0):
        """Near-surface sensor response at points `pts` (n, 2).

        s = sum_i a_i * (z_i^2 / (r_i^2 + z_i^2))^(3/2),  z_i = h + depth_i
        (normalised dipole-like fall-off) plus white noise.
        """
        pts = np.atleast_2d(pts)
        s = np.zeros(len(pts))
        for it in self.items:
            if it.neutralised:
                continue
            z2 = (head_height + it.depth) ** 2
            r2 = (pts[:, 0] - it.x) ** 2 + (pts[:, 1] - it.y) ** 2
            s += it.strength * (z2 / (r2 + z2)) ** 1.5
        if rng is not None and noise > 0:
            s += rng.normal(0.0, noise, len(s))
        return s

    def footprint_hits(self, x, y, radius):
        """Ids of hazardous items whose trigger footprint overlaps a disc."""
        return [it.id for it in self.items if it.hazardous
                and np.hypot(x - it.x, y - it.y) < radius + it.hazard_r]

    def hazards(self):
        return [it for it in self.items if it.kind != "clutter"]

    # ---- io -------------------------------------------------------------
    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"seed": self.seed, "field": {"w": self.w, "h": self.h},
                       "items": [asdict(i) for i in self.items]}, f, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["field"]["w"], d["field"]["h"],
                   [Item(**i) for i in d["items"]], d.get("seed"))


def generate_field(seed, w=12.0, h=12.0, n_ap=4, n_at=2, n_clutter=4,
                   min_sep=2.0, margin=0.8):
    """Random field with a minimum separation between items."""
    rng = np.random.default_rng(seed)
    kinds = ["ied_ap"] * n_ap + ["ied_at"] * n_at + ["clutter"] * n_clutter
    pos = []
    for _ in kinds:
        for _attempt in range(1000):
            p = rng.uniform([margin, margin], [w - margin, h - margin])
            if all(np.hypot(*(p - q)) >= min_sep for q in pos):
                pos.append(p)
                break
        else:
            raise RuntimeError("could not place items; lower min_sep")
    items = []
    for k, (kind, p) in enumerate(zip(kinds, pos)):
        spec = KINDS[kind]
        u = {key: rng.uniform(*rng_) for key, rng_ in spec.items()}
        items.append(Item(id=k, kind=kind, x=float(p[0]), y=float(p[1]),
                          depth=float(u["depth"]), strength=float(u["strength"]),
                          body_r=float(u["body_r"]), hazard_r=float(u["hazard_r"])))
    return Field(w, h, items, seed)
