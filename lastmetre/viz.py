"""Rendering: static summary figures and an animated GIF of a mission."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

COL = dict(ied_ap="#e63946", ied_at="#ff7f11", clutter="#8d99ae")
NORM = LogNorm(vmin=1e-4, vmax=1.0)


def _base(ax, sim, P, show_truth=True, neutralised=()):
    g = sim.bel.grid
    ax.imshow(P.astype(float), origin="lower", extent=g.extent, cmap="magma", norm=NORM,
              interpolation="nearest")
    ax.add_patch(Rectangle((0, 0), sim.f.w, sim.f.h, fill=False, ec="white", lw=1.0, ls="--"))
    if show_truth:
        for it in sim.f.items:
            c = COL[it.kind]
            if it.kind != "clutter":
                done = it.id in neutralised
                ax.add_patch(Circle((it.x, it.y), it.hazard_r, fill=False, ec=c,
                                    lw=0.8, ls=":" if not done else "-", alpha=0.9))
                ax.plot(it.x, it.y, "x" if done else "o", ms=4, color=c)
            else:
                ax.plot(it.x, it.y, "s", ms=3, color=c)
    ax.set_xlim(g.extent[0], g.extent[1])
    ax.set_ylim(g.extent[2], g.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _robot(ax, sim, pose, pts, ee=None):
    x, y, th = pose
    ax.add_patch(Circle((x, y), sim.p.robot_radius, fc="#2ec4b6", ec="white", lw=1.0, zorder=5))
    ax.plot([x, x + 0.35 * np.cos(th)], [y, y + 0.35 * np.sin(th)], color="white", lw=1.5, zorder=6)
    c = pts.mean(0)
    ax.plot([x, c[0]], [y, c[1]], color="#cbf3f0", lw=0.8, zorder=4)
    ax.plot(pts[:, 0], pts[:, 1], color="#cbf3f0", lw=2.5, zorder=4)
    if ee is not None:
        ax.plot([x, ee[0]], [y, ee[1]], color="#ffd166", lw=2.0, zorder=6)
        ax.plot(*ee, "o", color="#ffd166", ms=5, zorder=7)


def summary_figure(sims, path, titles=None):
    """Side-by-side final state of several runs (same field)."""
    n = len(sims)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), dpi=110)
    axes = np.atleast_1d(axes)
    for ax, sim, title in zip(axes, sims, titles or [s.policy for s in sims]):
        _base(ax, sim, sim.bel.P_haz)
        tr = np.array(sim.trace)
        ax.plot(tr[:, 0], tr[:, 1], color="#2ec4b6", lw=0.7, alpha=0.9)
        for iid in sim.triggers:
            it = sim.f.items[iid]
            ax.plot(it.x, it.y, "*", ms=14, color="yellow", mec="black", zorder=8)
        cov = sim.bel.observed[sim.bel.in_field].mean()
        ax.set_title(f"{title}\ntriggers={len(sim.triggers)}  coverage={cov:.0%}  t={sim.t:.0f}s",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def animate(sim, path, every=1, fps=12, title="", dpi=72, max_frames=420):
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = sim.frames[::every]
    if len(frames) > max_frames:
        frames = frames[:: int(np.ceil(len(frames) / max_frames))] + [sim.frames[-1]]
    trace = np.array(sim.trace)
    fig, ax = plt.subplots(figsize=(6.0, 6.6), dpi=dpi)

    def draw(k):
        fr = frames[k]
        ax.clear()
        _base(ax, sim, fr["P"], neutralised=fr["neut"])
        ax.contour(*sim.bel.grid.centers(), fr["F"].astype(float), levels=[sim.p.eps],
                   colors=["#2ec4b6"], linewidths=0.6, alpha=0.8)
        tr = trace[: fr["ntrace"]]
        ax.plot(tr[:, 0], tr[:, 1], color="#2ec4b6", lw=0.6, alpha=0.7)
        for u in fr["units"]:
            ax.plot(*u, "D", color="#ffd166", ms=5, zorder=7)
        _robot(ax, sim, fr["pose"], fr["pts"], fr["ee"])
        ax.set_title(f"{title}\n{fr['phase']}" + (f" — {fr['note']}" if fr["note"] else ""),
                     fontsize=9, loc="left")
        ax.text(0.01, 0.01, f"t = {fr['t']:6.0f} s   virtual triggers = {fr['trig']}",
                transform=ax.transAxes, color="white", fontsize=8, family="monospace")

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
