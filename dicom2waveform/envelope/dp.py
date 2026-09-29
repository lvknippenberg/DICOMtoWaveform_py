"""Primary estimator: likelihood image + globally optimal boundary (dynamic programming).

Port of `CalculateEnvelopeDP` from the MATLAB implementation, which was the
best method against vendor ground truth (MAE 2.3 vs 18.1 cm/s for the legacy
per-column method). For each side of the baseline the boundary path e(x)
minimises

    sum_x [ signal density beyond e(x)  +  non-signal between baseline and e(x) ]
        + lambda * sum_x |e(x) - e(x-1)|

i.e. the two-region split of every column, solved jointly over all columns.
The L1 term is a physiological acceleration prior: one row per column is
|dy|/dt cm/s^2, so lambda = (|dy|/dt) / a_ref makes it the same physical
constraint on every image.

Differences from the MATLAB version:
- No two-state direction Viterbi: the requested outputs are the positive and
  negative outer envelopes separately, both defined in every column (a side
  without flow has its edge at the baseline, which the cost produces
  naturally: with no signal density, e = 0 is cheapest).
- The path is decoupled across the sweep (no transition penalty from the
  columns before the sweep to those after it: they are not adjacent in time).
- The distance transform uses a vectorised cumulative-minimum formulation.
- Optional scaling of a_ref with the display velocity range (MATLAB open issue:
  a fixed a_ref clipped systolic upstrokes on high-velocity-scale records).

Plan.txt section 20 also proposed a second-order (curvature) term. That makes
the state space quadratic in the number of rows (~400^2 states x ~900 columns)
and was not adopted; curvature is controlled by optional post-smoothing.
"""

from __future__ import annotations

import numpy as np

from .base import Context, EnvelopeEstimate, EnvelopeEstimator, SIDES, side_view
from .likelihood import density_window, local_density, membership


def _l1_min_convolution(f: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """h(i) = min_k f(k) + lam*|i-k| and the minimising k, in O(n)."""
    n = len(f)
    idx = np.arange(n)
    # forward: min over k <= i of f(k) - lam*k, plus lam*i
    a = f - lam * idx
    cm = np.minimum.accumulate(a)
    arg_f = np.maximum.accumulate(np.where(a <= cm, idx, 0))
    h_f = cm + lam * idx
    # backward: min over k >= i of f(k) + lam*k, minus lam*i
    b = (f + lam * idx)[::-1]
    cmb = np.minimum.accumulate(b)
    ridx = idx[::-1]
    # position (in reversed array) of the running minimum
    pos = np.maximum.accumulate(np.where(b <= cmb, np.arange(n), 0))
    arg_b = ridx[pos][::-1]
    h_b = cmb[::-1] - lam * idx
    use_b = h_b < h_f
    return np.where(use_b, h_b, h_f), np.where(use_b, arg_b, arg_f)


def dp_solve(cost: np.ndarray, lam: float, free: np.ndarray | None = None) -> np.ndarray:
    """Minimise sum_x cost[e_x, x] + lam*|e_x - e_{x-1}| exactly.

    free: (W,) bool; the transition INTO column x is unpenalised where free[x]
    (used to decouple the path across the sweep).
    """
    D, W = cost.shape
    F = np.empty((D, W))
    P = np.empty((D, W), dtype=np.int32)
    F[:, 0] = cost[:, 0]
    P[:, 0] = np.arange(D)
    for x in range(1, W):
        if free is not None and free[x]:
            k = int(np.argmin(F[:, x - 1]))
            F[:, x] = cost[:, x] + F[k, x - 1]
            P[:, x] = k
        else:
            h, arg = _l1_min_convolution(F[:, x - 1], lam)
            F[:, x] = cost[:, x] + h
            P[:, x] = arg
    path = np.empty(W, dtype=np.int64)
    path[-1] = int(np.argmin(F[:, -1]))
    for x in range(W - 1, 0, -1):
        path[x - 1] = P[path[x], x]
    return path


def split_cost(density: np.ndarray, wall_rows: int = 0, wall_weight: float = 0.01) -> np.ndarray:
    """MATLAB cost ('split'): two-region split of each column.

        cost(k) = sum_{j > k} d_j + sum_{j <= k} w_j (1 - d_j)

    Assumes the flow region is filled from the baseline outward (true for CW,
    not for the PW spectral window or TDI bands). Rows j < wall_rows are nearly
    free to include (wall-filter band).
    """
    D = density.shape[0]
    w = np.ones((D, 1))
    w[:min(wall_rows, D)] = wall_weight
    beyond = density.sum(axis=0, keepdims=True) - np.cumsum(density, axis=0)
    return beyond + np.cumsum(w * (1.0 - density), axis=0)


def edge_cost(density: np.ndarray, theta: float = 0.2, inner_rows: int = 10,
              height_prior: float = 0.01, wall_rows: int = 0, wall_weight: float = 0.01) -> np.ndarray:
    """Outer-edge cost ('edge', default).

        cost(k) = sum_{j > k}           max(d_j - theta, 0)      signal left outside
                + sum_{j in (k-h, k]}   w_j max(theta - d_j, 0)  no signal just inside
                + height_prior * k                               weak prior toward no flow

    d = background-corrected local signal density, theta = density at which the
    edge is placed, h = inner_rows. The interior deeper than h is free, so hollow
    spectra (PW spectral window, TDI bands) are handled; pure noise (d < theta
    everywhere) gives k = 0 exactly; moving the edge outward through noise never
    lowers the cost. Rows in the wall band are nearly free to include.
    """
    D = density.shape[0]
    w = np.ones((D, 1))
    w[:min(wall_rows, D)] = wall_weight
    excess = np.clip(density - theta, 0.0, None)
    short = w * np.clip(theta - density, 0.0, None)
    beyond = excess.sum(axis=0, keepdims=True) - np.cumsum(excess, axis=0)
    sc = np.cumsum(short, axis=0)
    h = max(1, int(inner_rows))
    prev = np.zeros_like(sc)
    prev[h:] = sc[:-h]
    k = np.arange(D)[:, None]
    return beyond + (sc - prev) + height_prior * k


def column_margin(cost: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Per-column confidence: how much better the chosen edge is than average."""
    best = cost[path, np.arange(cost.shape[1])]
    spread = cost.max(axis=0) - cost.min(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (cost.mean(axis=0) - best) / spread
    m[~np.isfinite(m)] = 0.0
    return np.clip(m, 0.0, 1.0)


class DPEstimator(EnvelopeEstimator):
    """Soft noise-referenced membership -> local density -> DP boundary per side."""

    name = "dp"
    klass = "image"
    defaults = {
        "reference_accel_cm_s2": None,  # None -> DPConfig
        "accel_scale_with_range": None,  # None -> DPConfig
        "hard_threshold": False,  # ablation: binary membership instead of soft
        "use_density": True,  # ablation: raw membership instead of local density
    }

    def lambda_rows(self, ctx: Context) -> float:
        spec = ctx.spec
        a_ref = self.params["reference_accel_cm_s2"] or ctx.dp.reference_accel_cm_s2
        scale = self.params["accel_scale_with_range"]
        scale = ctx.dp.accel_scale_with_range if scale is None else scale
        if scale:
            lo, hi = spec.v_limits
            a_ref = a_ref * max(abs(lo), abs(hi)) / ctx.dp.reference_range_cm_s
        return (abs(spec.dy) / spec.dt) / a_ref

    def likelihood(self, ctx: Context) -> np.ndarray:
        spec = ctx.spec
        ns = ctx.noise
        if self.params["hard_threshold"]:
            m = (spec.level >= ns.threshold).astype(float)
        else:
            m = membership(spec.level, ns)
        m[:, spec.sweep] = 0.0
        if self.params["use_density"]:
            m = local_density(m, density_window(spec, ctx.likelihood))
        q = ctx.dp.background_percentile
        if q is not None and q > 0:
            # remove the density that pure noise produces, so that noise is
            # cost-neutral rather than an attraction toward the display limit
            valid = spec.valid & ~spec.sweep[None, :]
            bg = float(np.percentile(m[valid], q)) if valid.any() else 0.0
            if bg < 0.9:
                m = np.clip((m - bg) / (1.0 - bg), 0.0, 1.0)
        qn = ctx.dp.normalise_percentile
        if qn is not None and qn > 0:
            # express density relative to this image's in-flow density, so the
            # edge level theta means "fraction of typical flow density" whether
            # the flow renders dense (vascular PW/CW) or sparse (speckled TDI)
            valid = spec.valid & ~spec.sweep[None, :]
            top = float(np.percentile(m[valid], qn)) if valid.any() else 1.0
            m = np.clip(m / max(top, ctx.dp.normalise_floor), 0.0, 1.0)
        return m

    def estimate(self, ctx: Context) -> EnvelopeEstimate:
        spec = ctx.spec
        L = self.likelihood(ctx)
        lam = self.lambda_rows(ctx)
        free = spec.sweep.copy()
        free[1:] |= spec.sweep[:-1]
        out = {}
        for side in SIDES:
            sv = side_view(spec, side, image=L)
            H = spec.level.shape[0]
            wall = int(round(ctx.dp.wall_band_rel * H))
            if ctx.dp.cost == "split":
                cost = split_cost(sv.level, wall_rows=wall)
            else:
                cost = edge_cost(sv.level, theta=ctx.dp.edge_density,
                                 inner_rows=max(1, int(round(ctx.dp.inner_band_rel * H))),
                                 height_prior=ctx.dp.height_prior, wall_rows=wall)
            cost[:, spec.sweep] = 0.0
            path = dp_solve(cost, lam, free).astype(float)
            # an edge inside the wall band means no detectable flow on this side
            path[path <= wall] = 0.0
            conf = column_margin(cost, path.astype(int))
            path[spec.sweep] = np.nan
            conf[spec.sweep] = np.nan
            out[side] = (path, conf)
        return self._result(out["pos"][0], out["neg"][0], out["pos"][1], out["neg"][1],
                            likelihood=L, lambda_rows=lam)


