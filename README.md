# R&D / AI Assignment — Parametric Curve Fitting

## Problem

Points are given on the curve

```
x(t) = t·cos(θ) − e^(M·|t|) · sin(0.3t) · sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M·|t|) · sin(0.3t) · cos(θ)
```

for `t ∈ [6, 60]`, with unknowns constrained to:

```
0°   < θ < 50°
-0.05 < M < 0.05
0    < X < 100
```

The data (`data/xy_data.csv`) gives only the resulting `(x, y)` points — **not**
the `t` value each point came from.

## Why this isn't a standard curve fit

`scipy.optimize.curve_fit` (and similar tools) need paired `(t, x)` / `(t, y)`
samples to fit `θ, M, X` directly. Here `t` is missing for every point, so the
fitting problem is really: *find the curve, and the point-to-curve
correspondence, that jointly explain the data* — a total-least-squares /
point-cloud-to-curve problem, not a simple regression.

## Method: alternating optimization (ICP-style)

`t` is treated as a latent nuisance parameter per data point and the fit
alternates between two sub-problems until convergence:

1. **Correspondence step** — freeze `(θ, M, X)`. For each data point, search
   over `t ∈ [6, 60]` for the point on the curve closest to it (coarse grid
   search, then a bounded 1-D local refinement with
   `scipy.optimize.minimize_scalar`).
2. **Parameter step** — freeze the per-point `t`'s found above. Refine
   `(θ, M, X)` with nonlinear least squares
   (`scipy.optimize.least_squares`, Levenberg–Marquardt) minimizing the sum
   of squared `(x, y)` residuals across all 1,500 points at once.

Repeating these two steps is analogous to Iterative Closest Point (ICP) used
in point-cloud registration: step 1 is the "find correspondences" step, step
2 is the "fit the transform" step.

The implementation runs this in two passes for speed/accuracy:
- **Coarse pass**: 2,000-point grid over `t`, correspondence recomputed from
  scratch each iteration (15 iterations) — gets close quickly.
- **Fine pass**: 6,000-point grid, with each point's search window warm-started
  around its previous `t` (8 iterations) — squeezes out the remaining error.

## Result

The fit converges to very clean, round numbers, with residual error shrinking
to near the level of floating-point/sampling noise:

| Parameter | Value |
|---|---|
| θ | 30° (π/6 rad ≈ 0.5236) |
| M | 0.03 |
| X | 55 |

**Mean L1 residual ≈ 0.0045**, **RMSE ≈ 0.0036** — both tiny relative to the
data's scale (x, y range roughly 46–109), which is strong evidence these are
the exact (or extremely close to exact) generating parameters rather than a
local-minimum approximation.

### Desmos / LaTeX form

```
\left(t\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,\ 42+t\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

## Running it

```bash
pip install numpy scipy
python fit_curve.py data/xy_data.csv
```

This prints the per-iteration convergence trace, the final `(θ, M, X)`
(in radians/degrees), the residual metrics, and the Desmos/LaTeX string.

## Files

- `fit_curve.py` — fitting script (alternating optimization as described above).
- `data/xy_data.csv` — the provided data points.
- `README.md` — this write-up.
