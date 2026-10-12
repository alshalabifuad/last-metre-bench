"""Hazard belief map and the probabilistic barrier derived from it.

Two layers:
  P_obj(c)  P(a sensed anomaly is centred in cell c)     -- Bayesian log-odds
  F(c)      max P_obj within (R_h + r_robot) of c       -- risk of putting the
            robot centre at c, because a device's trigger footprint is only
            known to be bounded by R_h = hazard_radius_max.

The barrier for the robot centre is  h(p) = log(eps) - blur(log F)(p).
h >= 0  <=>  (smoothed) P(hazard under the robot footprint) <= eps.
Unsensed ground keeps the prior (> eps), so the robot may only drive where
its own sensor has already looked -- far enough ahead to cover a footprint.
"""
from collections import deque

import numpy as np

from .grid import Grid, disk_offsets, dilate_disk, gaussian_blur


def logit(p):
    return np.log(p / (1.0 - p))


def sigmoid(l):
    return 1.0 / (1.0 + np.exp(-l))


class HazardBelief:
    def __init__(self, p, field_w, field_h):
        self.p = p
        g = self.grid = Grid(-p.perimeter, -p.perimeter,
                             field_w + p.perimeter, field_h + p.perimeter, p.res)
        X, Y = g.centers()
        self.in_field = (X >= 0) & (X <= field_w) & (Y >= 0) & (Y <= field_h)
        self.L = np.where(self.in_field, logit(p.prior), p.l_min)
        # the perimeter is a cleared access strip: known, and known clear
        self.observed = ~self.in_field
        self._r_robot = (p.hazard_radius_max + p.robot_radius) / p.res
        self._r_haz = p.hazard_radius_max / p.res
        self._d_sample = disk_offsets(p.sample_radius / p.res)
        self.refresh()

    # ---- measurement update ----------------------------------------------
    def log_lr(self, s):
        p = self.p
        l1 = -0.5 * ((s - p.mu_obj) / p.sigma_obj) ** 2 - np.log(p.sigma_obj)
        l0 = -0.5 * (s / p.sigma_bg) ** 2 - np.log(p.sigma_bg)
        return l1 - l0

    def update(self, pts, s):
        g, p = self.grid, self.p
        i0, j0 = g.index(pts[:, 0], pts[:, 1])
        llr = p.update_weight * self.log_lr(np.asarray(s))
        for di, dj in self._d_sample:
            i, j = i0 + di, j0 + dj
            ok = g.inside(i, j)
            np.add.at(self.L, (i[ok], j[ok]), llr[ok])
            self.observed[i[ok], j[ok]] = True
        # the perimeter never changes
        np.clip(self.L, p.l_min, p.l_max, out=self.L)
        self.L[~self.in_field] = p.l_min

    def reset_region(self, x, y, r):
        """Forget evidence near (x, y), e.g. after an intervention changed the world."""
        X, Y = self.grid.centers()
        m = ((X - x) ** 2 + (Y - y) ** 2 <= r * r) & self.in_field
        self.L[m] = logit(self.p.prior)
        self.observed[m] = False

    # ---- derived layers ---------------------------------------------------
    def refresh(self):
        P = sigmoid(self.L)
        self.P_obj = P
        low = sigmoid(self.p.l_min)
        self.F = dilate_disk(P, self._r_robot, low)
        self.P_haz = dilate_disk(P, self._r_haz, low)
        self.G = gaussian_blur(np.log(self.F), self.p.blur_sigma_cells)

    def barrier(self, x, y):
        """h and grad h at the robot centre (finite differences)."""
        g, d = self.grid, 0.5 * self.grid.res
        h = lambda a, b: np.log(self.p.eps) - g.interp(self.G, a, b)
        val = h(x, y)
        grad = np.array([(h(x + d, y) - h(x - d, y)) / (2 * d),
                         (h(x, y + d) - h(x, y - d)) / (2 * d)])
        return val, grad

    def cell_class(self, margin=0.5):
        """0 = safe for the robot centre (barrier h >= margin, i.e. the same
        smoothed field the safety filter uses), 1 = unknown/uncertain,
        2 = evidence of hazard (F > blocked_p)."""
        c = np.ones(self.grid.shape, dtype=np.int8)
        c[self.G <= np.log(self.p.eps) - margin] = 0
        c[self.F > self.p.blocked_p] = 2
        return c

    # ---- suspect extraction -----------------------------------------------
    def suspects(self, threshold=0.5, min_cells=2):
        """Connected blobs of P_obj > threshold -> (mean, cov, peak)."""
        m = (self.P_obj > threshold) & self.in_field
        seen = np.zeros_like(m)
        out = []
        X, Y = self.grid.centers()
        ny, nx = m.shape
        for i0, j0 in zip(*np.nonzero(m)):
            if seen[i0, j0]:
                continue
            comp, q = [], deque([(i0, j0)])
            seen[i0, j0] = True
            while q:
                i, j = q.popleft()
                comp.append((i, j))
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < ny and 0 <= b < nx and m[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        q.append((a, b))
            if len(comp) < min_cells:
                continue
            ii, jj = np.array(comp).T
            w = self.P_obj[ii, jj]
            pts = np.stack([X[ii, jj], Y[ii, jj]], 1)
            mu = (w[:, None] * pts).sum(0) / w.sum()
            d = pts - mu
            cov = (w[:, None, None] * np.einsum("ni,nj->nij", d, d)).sum(0) / w.sum()
            cov += np.eye(2) * (self.grid.res ** 2) / 12.0
            out.append(dict(mean=mu, cov=cov, peak=float(w.max()), cells=len(comp)))
        return out
