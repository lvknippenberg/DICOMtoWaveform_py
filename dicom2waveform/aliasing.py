"""Aliasing correction for PW / TDI displays (automatic per-column baseline shift).

A pulsed-wave spectrum is periodic in velocity with the period of the display
span (the Nyquist range): flow faster than one display limit re-enters at the
other. The scanner's display window is therefore arbitrary, and a sonographer
fixes aliasing by shifting the baseline. This module does the same, per
column:

1. The display is treated as circular (row r == row r + H).
2. A *cut path* is found through the circular display, away from the
   zero-velocity zone, that crosses the least signal, with a smoothness prior
   (dynamic programming over the cut position). Without evidence the cut
   stays at the display edge (no change). Between columns the cut may not
   jump through signal present in both: a flow band stays on one side of the
   cut, so the cut follows a wrapped band as it grows instead of tearing
   through it to the other side.
3. Rows beyond the cut are moved by one period (+-H rows): rows beyond a cut
   in the top part become velocities past the bottom limit and vice versa.
   The result is a taller, unwrapped Spectrogram on which every estimator,
   the mean envelopes and the flow mask run unchanged.

The correction is only attempted when the wrap-around test fires
(`quality.aliasing_suspected`) and only applied when the cut path carries
clearly less signal than the display edge. It cannot resolve flows whose
combined spread exceeds the Nyquist range in the same column; those remain
(and stay flagged).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import AliasingConfig
from .envelope.dp import dp_solve
from .render import Spectrogram


@dataclass
class UnwrapResult:
    spec: Spectrogram
    applied: bool
    cut_rows: np.ndarray  # (W,) cut row per column in display coordinates (-1 = none)
    edge_cost: float
    path_cost: float
    n_columns: int  # columns in which rows were moved
    shift_up: int  # rows added above the display
    shift_down: int  # rows added below the display


def _arc_rows(H: int, z: int, guard: int) -> tuple[np.ndarray, int]:
    """Allowed cut rows as one linear coordinate running from the zero-row side of
    the bottom part, through the display edge (row H-1 -> row 0), to the zero-row
    side of the top part. Returns (rows, index of the display edge)."""
    bottom = np.arange(min(H - 1, z + guard), H)  # rows below the zero row
    top = np.arange(0, max(0, z - guard) + 1)  # rows above the zero row
    rows = np.concatenate([bottom, top])
    return rows, len(bottom) - 1  # index of row H-1; the next index is row 0


def find_cut(density: np.ndarray, zero_row: float, sweep: np.ndarray, cfg: AliasingConfig
             ) -> tuple[np.ndarray, float, float]:
    """Cut row per column (display coordinates) and the costs of the cut path and of
    the display edge (mean signal density crossed per column)."""
    H, W = density.shape
    z = int(round(zero_row))
    guard = max(1, int(round(cfg.guard_rel * H)))
    rows, i_edge = _arc_rows(H, z, guard)
    d = ndimage.uniform_filter1d(density, max(1, cfg.cut_smooth_rows), axis=0, mode="wrap")
    cost = d[rows].copy()
    u = np.arange(len(rows))
    # without evidence stay at the display edge (between rows H-1 and 0)
    cost += cfg.edge_preference * np.minimum(np.abs(u - i_edge), np.abs(u - (i_edge + 1)))[:, None]
    cost[:, sweep] = 0.0
    free = sweep.copy()
    free[1:] |= sweep[:-1]
    # transition x-1 -> x: cut_lambda per row + cut_cross_weight per unit of density
    # passed over (present in both columns, so following a growing band is free)
    da = np.clip(d[rows], 0.0, None)
    both = np.minimum(da, np.roll(da, 1, axis=1))
    coords = cfg.cut_lambda * u[:, None] + cfg.cut_cross_weight * np.cumsum(both, axis=0)
    path = dp_solve(cost, cfg.cut_lambda, free, coords)
    cut = rows[path]
    data = ~sweep
    path_cost = float(d[cut, np.arange(W)][data].mean()) if data.any() else 0.0
    edge_cost = float(0.5 * (d[0] + d[H - 1])[data].mean()) if data.any() else 0.0
    return cut, path_cost, edge_cost


def unwrap(spec: Spectrogram, density: np.ndarray, cfg: AliasingConfig) -> UnwrapResult:
    """Unwrap the spectrogram along the least-signal cut path (see module docstring)."""
    H, W = spec.level.shape
    z = spec.zero_row
    cut, path_cost, edge_cost = find_cut(density, z, spec.sweep, cfg)
    zi = int(round(z))
    # rows to move per column: below-zero part -> cut c > z: rows c+1..H-1 move up (r - H)
    #                          above-zero part -> cut c < z: rows 0..c-1 move down (r + H)
    n_up = np.where(cut > zi, H - 1 - cut, 0)
    n_down = np.where(cut < zi, cut, 0)
    n_up[spec.sweep] = 0
    n_down[spec.sweep] = 0
    moved = (n_up > 0) | (n_down > 0)
    improves = path_cost < cfg.max_cost_ratio * edge_cost and edge_cost > cfg.min_edge_density
    if not moved.any() or not improves:
        return UnwrapResult(spec, False, np.full(W, -1), edge_cost, path_cost, 0, 0, 0)

    T = int(n_up.max())
    B = int(n_down.max())
    Hn = H + T + B
    bg = float(np.median(spec.level[spec.valid])) if spec.valid.any() else 0.0

    def remap(arr: np.ndarray, fill) -> np.ndarray:
        out = np.full((Hn,) + arr.shape[1:], fill, dtype=arr.dtype)
        out[T:T + H] = arr
        for x in np.flatnonzero(n_up > 0):
            k = n_up[x]
            out[T - k:T, x] = arr[H - k:H, x]  # row r -> r - H
            out[T + H - k:T + H, x] = fill
        for x in np.flatnonzero(n_down > 0):
            k = n_down[x]
            out[T + H:T + H + k, x] = arr[0:k, x]  # row r -> r + H
            out[T:T + k, x] = fill
        return out

    new = copy.copy(spec)
    new.level = remap(spec.level.astype(np.float32), np.float32(bg))
    new.valid = remap(spec.valid, True)
    new.valid[:, spec.sweep] = False
    if spec.overlay is not None:
        new.overlay = remap(spec.overlay, False)
    if spec.rgb is not None:
        new.rgb = remap(spec.rgb, np.uint8(0))
    new.zero_row = z + T
    new.baseline_rows = spec.baseline_rows + T
    new.info = dict(spec.info)
    new.info["aliasing_unwrapped"] = {"columns": int(moved.sum()), "rows_up": T, "rows_down": B}
    return UnwrapResult(new, True, np.where(moved, cut, -1), edge_cost, path_cost, int(moved.sum()), T, B)
