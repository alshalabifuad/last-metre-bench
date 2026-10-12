"""Small numpy helpers for a 2-D grid (no SciPy dependency)."""
import numpy as np


class Grid:
    """Axis-aligned grid; array index [i, j] = [row (y), column (x)]."""

    def __init__(self, xmin, ymin, xmax, ymax, res):
        self.xmin, self.ymin, self.res = xmin, ymin, res
        self.nx = int(round((xmax - xmin) / res))
        self.ny = int(round((ymax - ymin) / res))

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def extent(self):
        return (self.xmin, self.xmin + self.nx * self.res,
                self.ymin, self.ymin + self.ny * self.res)

    def index(self, x, y):
        j = np.floor((np.asarray(x) - self.xmin) / self.res).astype(int)
        i = np.floor((np.asarray(y) - self.ymin) / self.res).astype(int)
        return i, j

    def inside(self, i, j):
        return (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)

    def center(self, i, j):
        return (self.xmin + (np.asarray(j) + 0.5) * self.res,
                self.ymin + (np.asarray(i) + 0.5) * self.res)

    def centers(self):
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.res
        ys = self.ymin + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def interp(self, arr, x, y):
        """Bilinear interpolation on cell centres (clamped at the border)."""
        fx = (x - self.xmin) / self.res - 0.5
        fy = (y - self.ymin) / self.res - 0.5
        j0 = int(np.clip(np.floor(fx), 0, self.nx - 2))
        i0 = int(np.clip(np.floor(fy), 0, self.ny - 2))
        tx = float(np.clip(fx - j0, 0.0, 1.0))
        ty = float(np.clip(fy - i0, 0.0, 1.0))
        a = arr[i0, j0] * (1 - tx) + arr[i0, j0 + 1] * tx
        b = arr[i0 + 1, j0] * (1 - tx) + arr[i0 + 1, j0 + 1] * tx
        return a * (1 - ty) + b * ty


def disk_offsets(r_cells):
    r = int(np.ceil(r_cells))
    return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
            if di * di + dj * dj <= r_cells * r_cells]


def dilate_max(arr, offsets, pad_value):
    """Grey-scale dilation (max filter) with an arbitrary structuring set."""
    r = max(max(abs(a), abs(b)) for a, b in offsets)
    ny, nx = arr.shape
    padded = np.pad(arr, r, constant_values=pad_value)
    out = np.full_like(arr, -np.inf)
    for di, dj in offsets:
        np.maximum(out, padded[r + di:r + di + ny, r + dj:r + dj + nx], out=out)
    return out


def dilate_disk(arr, r_cells, pad_value):
    """Disc max filter in O(r) array operations (same result as dilate_max
    with disk_offsets(r_cells)): grow horizontal running maxima once, then
    take one shifted row per vertical offset."""
    r = int(np.ceil(r_cells))
    ny, nx = arr.shape
    p = np.pad(arr, r, constant_values=pad_value)
    H = [p]
    for w in range(1, r + 1):
        prev = H[-1]
        cur = prev.copy()
        np.maximum(cur[:, :-w], p[:, w:], out=cur[:, :-w])
        np.maximum(cur[:, w:], p[:, :-w], out=cur[:, w:])
        H.append(cur)
    out = np.full_like(arr, -np.inf)
    for di in range(-r, r + 1):
        rem = r_cells * r_cells - di * di
        if rem < 0:
            continue
        w = int(np.floor(np.sqrt(rem) + 1e-9))
        np.maximum(out, H[w][r + di:r + di + ny, r:r + nx], out=out)
    return out


def gaussian_blur(arr, sigma_cells):
    if sigma_cells <= 0:
        return arr.copy()
    r = int(np.ceil(2.5 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    ny, nx = arr.shape
    p = np.pad(arr, ((0, 0), (r, r)), mode="edge")
    tmp = sum(k[m] * p[:, m:m + nx] for m in range(2 * r + 1))
    p = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    return sum(k[m] * p[m:m + ny, :] for m in range(2 * r + 1))
