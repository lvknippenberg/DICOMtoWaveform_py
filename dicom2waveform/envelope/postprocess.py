"""Envelope post-processing (Plan section 21): median, Savitzky-Golay, Kalman.

Applied to a velocity series in time order. NaN runs (the sweep) split the
series into independent segments; nothing is smoothed across them.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from ..config import PostConfig


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], valid, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def kalman_smooth(z: np.ndarray, dt: float, q: float, r: float, order: int = 2, rts: bool = True) -> np.ndarray:
    """Constant-velocity (order=2) or constant-acceleration (order=3) Kalman
    filter with Rauch-Tung-Striebel smoothing.

    q: SD of the white acceleration (order 2) or jerk (order 3) driving noise.
    r: measurement noise SD (same units as z).
    """
    n = len(z)
    if n == 0:
        return z
    if order == 2:
        F = np.array([[1, dt], [0, 1]])
        G = np.array([[dt ** 2 / 2], [dt]])
    else:
        F = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
        G = np.array([[dt ** 3 / 6], [dt ** 2 / 2], [dt]])
    Q = G @ G.T * q ** 2
    Hm = np.zeros((1, order))
    Hm[0, 0] = 1.0
    x = np.zeros(order)
    x[0] = z[0]
    P = np.eye(order) * (r ** 2 * 10)
    xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
    for k in range(n):
        if k:
            x = F @ x
            P = F @ P @ F.T + Q
        xs_p.append(x.copy())
        Ps_p.append(P.copy())
        S = Hm @ P @ Hm.T + r ** 2
        K = P @ Hm.T / S
        x = x + (K * (z[k] - Hm @ x)).ravel()
        P = (np.eye(order) - K @ Hm) @ P
        xs_f.append(x.copy())
        Ps_f.append(P.copy())
    if not rts:
        return np.array([xf[0] for xf in xs_f])
    xs = xs_f[-1].copy()
    out = np.empty(n)
    out[-1] = xs[0]
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs = xs_f[k] + C @ (xs - xs_p[k + 1])
        out[k] = xs[0]
    return out


def smooth_series(v: np.ndarray, dt: float, cfg: PostConfig, pixel_cm_s: float = 1.0) -> np.ndarray:
    """Smooth a velocity series (time order), segment by segment."""
    if cfg.method == "none":
        return v
    out = v.copy()
    for a, b in _segments(~np.isnan(v)):
        seg = v[a:b]
        if cfg.method == "median":
            n = max(1, int(round(cfg.median_window_s / dt))) | 1
            out[a:b] = ndimage.median_filter(seg, size=n, mode="nearest")
        elif cfg.method == "savgol":
            n = max(cfg.savgol_order + 2, int(round(cfg.savgol_window_s / dt)) | 1)
            if len(seg) > n:
                out[a:b] = signal.savgol_filter(seg, n, cfg.savgol_order, mode="interp")
        elif cfg.method in ("kalman_cv", "kalman_ca"):
            r = cfg.kalman_r_cm_s or 2.0 * pixel_cm_s
            order = 2 if cfg.method == "kalman_cv" else 3
            out[a:b] = kalman_smooth(seg, dt, cfg.kalman_q, r, order)
        else:
            raise ValueError(f"unknown post-processing '{cfg.method}'")
    return out
