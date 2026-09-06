"""
Recover the unknown parameters (theta, M, X) of the parametric curve

    x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)

given only a cloud of (x, y) points sampled from the curve for
t in [6, 60] (the corresponding t of each point is NOT given).

Method
------
Because each data point's t-value is unknown, this cannot be solved with a
standard curve_fit (which needs paired (t, x, y) samples). Instead we treat
each point's t as a latent/nuisance variable and use alternating
optimization, similar in spirit to Iterative Closest Point (ICP):

  1. Freeze (theta, M, X). For every data point, find the t in [6, 60]
     whose curve position is closest to that point (grid search + local
     refinement with a bounded 1-D minimizer).
  2. Freeze those per-point t's. Refine (theta, M, X) with nonlinear
     least squares (Levenberg-Marquardt) to minimize the sum of squared
     (x, y) residuals against all points simultaneously.
  3. Repeat 1-2 until the parameters stop changing.

This converges to a very small residual (well below data spacing), which is
a strong indication the recovered parameters are exact/close to exact.

Usage
-----
    python fit_curve.py data/xy_data.csv
"""

import sys
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def curve_xy(t, theta, M, X):
    """Evaluate the parametric curve at parameter value(s) t."""
    envelope = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - envelope * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + envelope * np.cos(theta)
    return x, y


def find_best_t(xs, ys, theta, M, X, t_grid, prev_t=None, window=2.0):
    """For each data point, find the t in [6, 60] whose curve point is
    closest (nearest neighbor on a grid, then local 1-D refinement)."""
    cx, cy = curve_xy(t_grid, theta, M, X)
    step = t_grid[1] - t_grid[0]
    n = len(xs)
    best_t = np.empty(n)

    for i in range(n):
        if prev_t is not None:
            lo, hi = max(6.0, prev_t[i] - window), min(60.0, prev_t[i] + window)
            mask = (t_grid >= lo) & (t_grid <= hi)
            idxs = np.where(mask)[0]
        else:
            idxs = np.arange(len(t_grid))

        d2 = (cx[idxs] - xs[i]) ** 2 + (cy[idxs] - ys[i]) ** 2
        t0 = t_grid[idxs[np.argmin(d2)]]

        def dist(t):
            env = np.exp(M * abs(t)) * np.sin(0.3 * t)
            xp = t * np.cos(theta) - env * np.sin(theta) + X
            yp = 42 + t * np.sin(theta) + env * np.cos(theta)
            return (xp - xs[i]) ** 2 + (yp - ys[i]) ** 2

        lo2, hi2 = max(6.0, t0 - 2 * step), min(60.0, t0 + 2 * step)
        res = minimize_scalar(dist, bounds=(lo2, hi2), method="bounded",
                               options={"xatol": 1e-8})
        best_t[i] = res.x

    return best_t


def residuals(params, xs, ys, t_fixed):
    theta, M, X = params
    cx, cy = curve_xy(t_fixed, theta, M, X)
    return np.concatenate([cx - xs, cy - ys])


def fit(xs, ys, init=(np.radians(25), 0.0, 50.0),
        coarse_grid_n=2000, fine_grid_n=6000,
        coarse_iters=15, fine_iters=8, verbose=True):
    params = np.array(init, dtype=float)

    # Pass 1: coarse grid, global nearest-t search each iteration.
    t_grid = np.linspace(6, 60, coarse_grid_n)
    for it in range(coarse_iters):
        t_fixed = find_best_t(xs, ys, *params, t_grid)
        result = least_squares(residuals, params, args=(xs, ys, t_fixed), method="lm")
        params = result.x
        if verbose:
            print(f"[coarse {it}] theta={np.degrees(params[0]):.4f} deg  "
                  f"M={params[1]:.5f}  X={params[2]:.4f}  cost={result.cost:.4f}")

    # Pass 2: fine grid, warm-started local search around previous t's.
    t_grid = np.linspace(6, 60, fine_grid_n)
    prev_t = None
    for it in range(fine_iters):
        prev_t = find_best_t(xs, ys, *params, t_grid, prev_t=prev_t)
        result = least_squares(residuals, params, args=(xs, ys, prev_t), method="lm")
        params = result.x
        if verbose:
            print(f"[fine {it}] theta={np.degrees(params[0]):.6f} deg  "
                  f"M={params[1]:.6f}  X={params[2]:.6f}  cost={result.cost:.6f}")

    cx, cy = curve_xy(prev_t, *params)
    l1 = np.mean(np.abs(cx - xs) + np.abs(cy - ys))
    rmse = np.sqrt(np.mean((cx - xs) ** 2 + (cy - ys) ** 2))
    return params, prev_t, l1, rmse


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/xy_data.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    xs, ys = data[:, 0], data[:, 1]

    params, t_fixed, l1, rmse = fit(xs, ys)
    theta, M, X = params

    print("\n=== Recovered parameters ===")
    print(f"theta = {theta:.6f} rad  = {np.degrees(theta):.4f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"\nMean L1 residual : {l1:.6f}")
    print(f"RMSE             : {rmse:.6f}")

    print("\n=== Desmos / LaTeX form ===")
    print(
        r"\left(t\cos({0:.4f})-e^{{{1:.4f}\left|t\right|}}\cdot\sin(0.3t)"
        r"\sin({0:.4f})+{2:.4f},\ 42+t\sin({0:.4f})+e^{{{1:.4f}\left|t\right|}}"
        r"\cdot\sin(0.3t)\cos({0:.4f})\right)".format(theta, M, X)
    )


if __name__ == "__main__":
    main()
