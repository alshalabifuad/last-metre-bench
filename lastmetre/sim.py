"""Closed-loop simulation: robot + boom sensor + belief + safety filter.

Policies
  naive     follows the survey lanes, ignores the sensor (shows the hazard)
  reactive  marks every sample whose raw signal exceeds a threshold and keeps
            a fixed disc around each mark, enforced by the SAME CBF filter --
            so the only difference to `belief` is the map representation
            (points vs. a calibrated belief that also knows unsensed ground)
  belief    the proposed probabilistic barrier (see belief.py)
"""
import numpy as np

from .belief import HazardBelief


class Sim:
    def __init__(self, field, p, seed=0, policy="belief", record_every=5,
                 start=None):
        self.f, self.p, self.policy = field, p, policy
        self.rng = np.random.default_rng(10_000 + seed)
        self.bel = HazardBelief(p, field.w, field.h)
        self.pose = np.array(start if start is not None
                             else [-p.perimeter + 0.25, p.lane_spacing / 2.0, 0.0], float)
        self.t = self.dist = 0.0
        self.triggers = {}          # item id -> sim time of first "virtual trigger"
        self.contacts = []          # placer touched a device body (sim)
        self.marks = []             # reactive policy marks
        self.units = []             # placed (inert) units
        self.ee = None              # placer end-effector position, if deployed
        self.min_h = np.inf
        self.phase, self.note = "SURVEY", ""
        self.log = []
        self.trace = [self.pose[:2].copy()]
        self.frames, self.record_every, self._k = [], record_every, 0
        self.last_pts = self.sensor_points()

    # ---- sensing ------------------------------------------------------------
    def sensor_points(self):
        x, y, th = self.pose
        p = self.p
        c = np.array([x + p.sensor_offset * np.cos(th), y + p.sensor_offset * np.sin(th)])
        n = np.array([-np.sin(th), np.cos(th)])
        o = np.linspace(-p.sensor_width / 2, p.sensor_width / 2, p.sensor_samples)
        return c + o[:, None] * n

    def sense(self):
        pts = self.sensor_points()
        s = self.f.signal_at(pts, self.p.sensor_height, self.rng, self.p.sensor_noise)
        self.bel.update(pts, s)
        self.bel.refresh()
        if self.policy == "reactive":
            for q in pts[s > self.p.react_threshold]:
                if all(np.hypot(*(q - m)) > 0.3 for m in self.marks):
                    self.marks.append(q.copy())
        self.last_pts = pts
        return pts, s

    # ---- safety filter --------------------------------------------------------
    def barrier(self):
        """(h, grad h) at the robot centre for the active policy, or None.
        belief:   probabilistic barrier from the hazard belief
        reactive: same CBF, but on a binary map (distance to nearest mark)"""
        x, y, _ = self.pose
        if self.policy == "belief":
            return self.bel.barrier(x, y)
        if self.policy == "reactive" and self.marks:
            M = np.array(self.marks)
            dist = np.hypot(M[:, 0] - x, M[:, 1] - y)
            k = int(np.argmin(dist))
            return dist[k] - self.p.react_radius, (np.array([x, y]) - M[k]) / max(dist[k], 1e-6)
        return None

    def filter(self, v):
        b = self.barrier()
        if b is not None:
            h, g = b
            if self.policy == "belief":
                self.min_h = min(self.min_h, h)
            th = self.pose[2]
            a = float(g @ np.array([np.cos(th), np.sin(th)]))
            # CBF condition  dh/dt = a * v >= -alpha * h
            if a < -1e-9:
                v = min(v, self.p.cbf_alpha * h / -a)
            elif a > 1e-9:
                v = max(v, -self.p.cbf_alpha * h / a)
        return float(np.clip(v, self.p.v_min, self.p.v_max))

    # ---- one control step -------------------------------------------------
    def step(self, v, w):
        self.sense()
        v = self.filter(v) if self.policy != "naive" else float(np.clip(v, 0, self.p.v_max))
        w = float(np.clip(w, -self.p.w_max, self.p.w_max))
        x, y, th = self.pose
        dt = self.p.dt
        self.pose = np.array([x + v * np.cos(th) * dt, y + v * np.sin(th) * dt,
                              (th + w * dt + np.pi) % (2 * np.pi) - np.pi])
        self.t += dt
        self.dist += abs(v) * dt
        for iid in self.f.footprint_hits(self.pose[0], self.pose[1], self.p.robot_radius):
            if iid not in self.triggers:
                self.triggers[iid] = self.t
                self.event(f"VIRTUAL TRIGGER on item {iid}")
        self.trace.append(self.pose[:2].copy())
        self._record()
        return v

    def event(self, msg):
        self.log.append((round(self.t, 1), self.phase, msg))

    def _record(self, force=False):
        self._k += 1
        if self.record_every and (force or self._k % self.record_every == 0):
            self.frames.append(dict(
                t=self.t, pose=self.pose.copy(), P=self.bel.P_haz.astype(np.float16),
                F=self.bel.F.astype(np.float16), pts=self.last_pts.copy(),
                ntrace=len(self.trace), phase=self.phase, note=self.note,
                trig=len(self.triggers), ee=None if self.ee is None else self.ee.copy(),
                units=[u.copy() for u in self.units], marks=[m.copy() for m in self.marks],
                neut=[it.id for it in self.f.items if it.neutralised]))
