"""Stage-A prototype of contribution C2: load-bounded placement.

The robot lowers a (dummy) unit onto the soil next to a buried, force-sensitive
object and must keep the load that reaches the object below a threshold at
every instant -- including the overshoot caused by sensing delay, braking and
soil damping -- although the soil stiffness is unknown.

1-D vertical, vectorised over Monte Carlo trials, everything normalised by the
unit's weight W. First-order, illustrative physics:

  soil reaction       F = k * d^1.5 * (1 + beta * d_dot)      (Hunt-Crossley;
                      d = penetration, k unknown per site: soft sand .. compacted)
  load on the object  L = gamma(r, z) * F                     (Boussinesq point load:
                      sigma_z = 3 P z^3 / (2 pi (r^2 + z^2)^(5/2)),  L ~ sigma_z * A)
  requirement         L(t) <= theta for all t

theta is a configurable, normalised "virtual activation" load. In a real
programme it would come from the certified partner / test standard -- it is
never chosen here, and nothing in this module describes a real device.

Controllers compared (all share actuator limits and a sensing delay):
  position   lower to the estimated surface - 1 cm, then release
  guarded    lower at constant speed until the measured force reaches W, stop, release
  robust     stopping-distance barrier with the worst-case stiffness k_max
  adaptive   same barrier, stiffness bound learned online from force/penetration
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ContactParams:
    W: float = 1.0                    # unit weight (normalisation)
    g: float = 9.81
    theta: float = 0.05               # max load on the object, in units of W
    headroom: float = 1.25            # planner: gamma_bar * W * headroom <= theta
    quantile: float = 0.975           # planner: upper quantile of gamma
    obj_radius: float = 0.06          # surrogate top radius (m) -> area A
    beta: float = 1.0                 # Hunt-Crossley damping (s/m)
    k_range: tuple = (350.0, 2800.0)  # W / m^1.5 ; W carried at 20 mm .. 5 mm sinkage
    surf_sigma: float = 0.01          # surface height uncertainty (m)
    z_range: tuple = (0.03, 0.10)     # true burial depth of the object top (m)
    z_sigma: float = 0.015            # depth estimate error (m)
    r_sigma: float = 0.02             # lateral position error after identification (m)
    v_max: float = 0.15               # lowering speed limit (m/s)
    a_max: float = 1.0                # braking / acceleration limit (m/s^2)
    tau: float = 0.010                # sensing + actuation delay (s)
    f_noise: float = 0.005            # force-sensor noise (W)
    dt: float = 0.0005
    t_max: float = 20.0
    start_height: float = 0.05        # above the *estimated* surface (m)


def gamma(r, z, p):
    """Fraction of a surface point load reaching the object top (Boussinesq)."""
    A = np.pi * p.obj_radius ** 2
    r2z2 = r ** 2 + z ** 2
    return np.minimum(1.0, 3.0 * A * z ** 3 / (2 * np.pi * r2z2 ** 2.5))


def plan_offset(z_hat, p, rng, nominal=False, quantile=None,
                r_grid=np.arange(0.05, 0.80, 0.005), ns=4000):
    """Smallest lateral offset whose (upper-quantile) transfer keeps the static
    load below theta / headroom. Returns (r_cmd, gamma_bar)."""
    if nominal:
        g = gamma(r_grid, z_hat, p)
    else:
        er = rng.normal(0, p.r_sigma, ns)
        z = np.clip(z_hat + rng.normal(0, p.z_sigma, ns), 0.01, None)
        q = p.quantile if quantile is None else quantile
        g = np.quantile(gamma(np.abs(r_grid[:, None] + er[None, :]), z[None, :], p), q, axis=1)
    ok = g * p.W * p.headroom <= p.theta
    i = int(np.argmax(ok)) if ok.any() else len(r_grid) - 1
    return float(r_grid[i]), float(g[i])


def draw_trials(n, p, seed=0):
    rng = np.random.default_rng(seed)
    t = dict(k=np.exp(rng.uniform(*np.log(p.k_range), n)),
             e_s=rng.normal(0, p.surf_sigma, n),
             z=rng.uniform(*p.z_range, n),
             e_r=rng.normal(0, p.r_sigma, n))
    t["z_hat"] = np.clip(t["z"] + rng.normal(0, p.z_sigma, n), 0.01, None)
    return t, rng


def _f_pred(Fm, speed, K, p):
    """Worst-case force after the delay and a full stop from `speed`, for stiffness <= K.
    (F^(2/3) + K^(2/3) * Delta)^(3/2) is increasing in K, so K = upper bound is safe."""
    delta = speed * p.tau + speed ** 2 / (2 * p.a_max)
    return (np.maximum(Fm, 0) ** (2 / 3) + K ** (2 / 3) * delta) ** 1.5 * (1 + p.beta * speed)


def _v_allow(Fm, K, F_lim, p, vmax, iters=24):
    lo, hi = np.zeros_like(Fm), np.full_like(Fm, vmax)
    ok_hi = _f_pred(Fm, hi, K, p) <= F_lim
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = _f_pred(Fm, mid, K, p) <= F_lim
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.where(ok_hi, vmax, lo)


def simulate(controller, trials, r_cmd, gamma_bar, p, seed=1, record=None, v_max=None):
    """Run one controller on all trials. Returns per-trial dict of results.
    record: index of a trial whose time series is returned for plotting."""
    rng = np.random.default_rng(seed)
    vmax = p.v_max if v_max is None else v_max
    n = len(trials["k"])
    k, s_hat = trials["k"], trials["e_s"]              # true surface is y = 0
    g_true = gamma(np.abs(r_cmd + trials["e_r"]), trials["z"], p)
    F_lim = p.theta / gamma_bar
    m = p.W / p.g
    y = s_hat + p.start_height
    v = np.zeros(n)
    phase = np.zeros(n, int)                           # 0 lowering, 1 stopping, 2 released
    t_rel = np.full(n, np.nan)
    L_max = np.zeros(n)
    F_max = np.zeros(n)
    d = int(round(p.tau / p.dt))
    F_buf, y_buf = np.zeros((d + 1, n)), np.tile(y, (d + 1, 1))
    y_contact = np.full(n, np.nan)
    k_hat = np.zeros(n)
    y_target = s_hat - 0.01
    trace = []
    steps = int(p.t_max / p.dt)
    for it in range(steps):
        t = it * p.dt
        pen = np.maximum(0.0, -y)
        F = np.maximum(0.0, k * pen ** 1.5 * (1 + p.beta * (-v)))
        L_max = np.maximum(L_max, g_true * F)
        F_max = np.maximum(F_max, F)
        F_buf = np.roll(F_buf, 1, axis=0); F_buf[0] = F
        y_buf = np.roll(y_buf, 1, axis=0); y_buf[0] = y
        Fm = F_buf[d] + rng.normal(0, p.f_noise, n)    # delayed, noisy measurement
        y_d = y_buf[d]
        speed = np.maximum(0.0, -v)

        if controller == "position":
            v_cmd = -np.minimum(vmax, 2.0 * np.maximum(0.0, y - y_target))
            done = (np.abs(y - y_target) < 1e-3) & (speed < 1e-3)
        elif controller == "guarded":
            phase = np.where((phase == 0) & (Fm >= 0.98 * p.W), 1, phase)
            v_cmd = np.where(phase == 0, -vmax, 0.0)
            done = (phase == 1) & (speed < 1e-4)
        else:
            if controller == "adaptive":
                newc = np.isnan(y_contact) & (Fm > 0.03 * p.W)
                y_contact = np.where(newc, y_d, y_contact)
                dhat = y_contact - y_d
                est = np.where((Fm > 0.1 * p.W) & (dhat > 1e-3),
                               Fm / np.maximum(dhat, 1e-6) ** 1.5, 0.0)
                k_hat = np.maximum(k_hat, est)
                K = np.where(k_hat > 0, np.minimum(p.k_range[1], 1.3 * k_hat), p.k_range[1])
            else:
                K = np.full(n, p.k_range[1])
            phase = np.where((phase == 0) & (Fm >= 0.98 * p.W), 1, phase)
            va = _v_allow(Fm + 3 * p.f_noise, K, F_lim, p, vmax)
            v_cmd = np.where(phase == 0, -va, 0.0)
            done = (phase == 1) & (speed < 1e-4)

        newly = done & (phase != 2)
        t_rel = np.where(newly, t, t_rel)
        phase = np.where(newly, 2, phase)
        held = phase != 2
        # gripper-held: velocity-controlled with acceleration limit
        dv = np.clip(v_cmd - v, -p.a_max * p.dt, p.a_max * p.dt)
        # released: free unit on Hunt-Crossley soil
        a_free = (-p.W + F) / m
        v = np.where(held, v + dv, v + a_free * p.dt)
        y = y + v * p.dt
        if record is not None:
            trace.append((t, y[record], F[record], g_true[record] * F[record] / p.theta, phase[record]))
        if (phase == 2).all() and np.all(t - t_rel > 0.5):
            break
    out = dict(L_ratio=L_max / p.theta, F_ratio=F_max / F_lim, t_release=t_rel, F_lim=F_lim,
               g_true=g_true, g_bar=gamma_bar,
               k=k, completed=~np.isnan(t_rel))
    if record is not None:
        out["trace"] = np.array(trace)
    return out
