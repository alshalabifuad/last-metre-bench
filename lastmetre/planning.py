"""Coverage lanes, grid search and a pure-pursuit tracker."""
import heapq

import numpy as np


def boustrophedon(w, h, spacing, overhang=1.6):
    """Lawn-mower corner waypoints; lane ends sit in the cleared perimeter."""
    ys = np.arange(spacing / 2.0, h + spacing / 2.0, spacing)
    wps, left = [], True
    for y in ys:
        xa, xb = (-overhang, w + overhang) if left else (w + overhang, -overhang)
        wps += [(xa, y), (xb, y)]
        left = not left
    return np.array(wps)


def densify(wps, step=0.1):
    wps = np.asarray(wps, float)
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


_NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


def dijkstra(cost, start, goals=None):
    """Grid Dijkstra. cost[i, j] = np.inf means blocked.

    Returns (dist, parent). Stops early once any cell of `goals` (set) is popped.
    """
    ny, nx = cost.shape
    dist = np.full(cost.shape, np.inf)
    parent = -np.ones(cost.shape + (2,), dtype=int)
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if goals is not None and (i, j) in goals:
            break
        for di, dj, w in _NB:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and np.isfinite(cost[a, b]):
                nd = d + w * cost[a, b]
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    parent[a, b] = (i, j)
                    heapq.heappush(pq, (nd, (a, b)))
    return dist, parent


def extract_path(parent, goal):
    path = [goal]
    while tuple(parent[path[-1]]) != (-1, -1):
        path.append(tuple(parent[path[-1]]))
    return path[::-1]


def grid_path(grid, cost, start_xy, goal_xy):
    """Shortest path between two points on a cost grid -> (n, 2) array or None."""
    s = tuple(int(v) for v in grid.index(*start_xy))
    g = tuple(int(v) for v in grid.index(*goal_xy))
    if not (grid.inside(*s) and grid.inside(*g)) or not np.isfinite(cost[g]):
        return None
    c = cost.copy()
    c[s] = 1.0  # the robot is where it is
    dist, parent = dijkstra(c, s, goals={g})
    if not np.isfinite(dist[g]):
        return None
    cells = extract_path(parent, g)
    xs, ys = grid.center(*np.array(cells).T)
    pts = np.stack([xs, ys], 1)
    pts[0], pts[-1] = start_xy, goal_xy
    return pts  # cell-centre path (8-connected); no thinning, so no corner cutting


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PurePursuit:
    """Carrot-following on a dense path; rotates in place for large errors."""

    def __init__(self, path, lookahead, v_max, w_max):
        self.path, self.L, self.v_max, self.w_max = np.asarray(path), lookahead, v_max, w_max
        self.k = 0

    def done(self, pose, tol=0.15):
        return self.k >= len(self.path) - 1 and np.hypot(*(self.path[-1] - pose[:2])) < tol

    def command(self, pose):
        x, y, th = pose
        # advance progress index along the path
        while self.k < len(self.path) - 1 and np.hypot(*(self.path[self.k] - (x, y))) < self.L:
            self.k += 1
        tx, ty = self.path[self.k]
        a = wrap(np.arctan2(ty - y, tx - x) - th)
        dist = np.hypot(tx - x, ty - y)
        if dist < 0.05:
            return 0.0, 0.0
        if abs(a) > 0.8:
            return 0.0, float(np.clip(2.0 * a, -self.w_max, self.w_max))
        v = self.v_max * np.cos(a) ** 2 * min(1.0, dist / self.L + 0.3)
        return float(v), float(np.clip(2.0 * a, -self.w_max, self.w_max))
