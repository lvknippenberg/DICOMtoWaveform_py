"""Class A: spectrum-derived (integrated power spectrum, IPS) estimators.

Each image column on one side of the baseline is treated as a one-sided power
spectrum: index 0 = zero velocity, last index = the display limit (f_H).

IMPORTANT: these are image-domain adaptations. The published methods operate
on the Doppler power spectrum. Here the "spectrum" is the rendered,
log-compressed display level; by default it is de-compressed to relative
linear power assuming `dynamic_range_db` (see render.level_to_power). The true
display mapping is unknown, so results are not reproductions of the
signal-domain performance.

Definitions follow:
- GM, MGM, SNSI: as summarised by Kathpalia et al. 2016 (IEEE TUFFC 63(11)),
  section II-A (GM: Marasek & Nowicki 1994; MGM: Moraes et al. 1995;
  SNSI: Steinman et al. 2000).
- Adaptive SNSI ('asnsi'): Kathpalia et al. 2016, sections II-A (proposed
  method) and II-B (one-third condition, adaptive power threshold,
  gap zeroing, interpolation, 7.5 ms moving average).
- QIA: Li et al. 2019, Information 10(6):199, section 2.1.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..render import level_to_power
from .base import ColumnwiseEstimator, Context, SideImage, side_view


def _smooth1d(s: np.ndarray, n: int) -> np.ndarray:
    return ndimage.uniform_filter1d(s, n, mode="nearest") if n and n > 1 else s


def _max_distance_above_line(P: np.ndarray, a: int, b: int) -> int:
    """Index in [a, b] of the IPS point farthest above the line P(a)->P(b).

    The normal distance is the vertical distance times cos(theta), so the
    argmax of the vertical distance is the same point.
    """
    if b <= a:
        return a
    x = np.arange(a, b + 1)
    line = P[a] + (x - a) * (P[b] - P[a]) / (b - a)
    return int(a + np.argmax(P[a:b + 1] - line))


def _strong_signal_run(s: np.ndarray, frac: float = 0.7) -> tuple[int, int]:
    """Contiguous region around the spectral peak with s >= frac * peak."""
    p = int(np.argmax(s))
    thr = frac * s[p]
    lo = p
    while lo > 0 and s[lo - 1] >= thr:
        lo -= 1
    hi = p
    while hi < len(s) - 1 and s[hi + 1] >= thr:
        hi += 1
    return lo, hi


def _ls_slope(P: np.ndarray, a: int, b: int) -> float:
    if b <= a:
        return float(P[min(a + 1, len(P) - 1)] - P[a]) if a < len(P) - 1 else 0.0
    x = np.arange(a, b + 1)
    return float(np.polyfit(x, P[a:b + 1], 1)[0])


def _local_slope(P: np.ndarray, half: int) -> np.ndarray:
    """Slope of the IPS by central differences over +-half bins."""
    n = len(P)
    i = np.arange(n)
    lo = np.clip(i - half, 0, n - 1)
    hi = np.clip(i + half, 0, n - 1)
    return (P[hi] - P[lo]) / np.maximum(hi - lo, 1)


class _IPSEstimator(ColumnwiseEstimator):
    """Shared spectrum preparation for the IPS family."""

    common = {
        "dynamic_range_db": 40.0,  # de-compression; <= 0 uses display level directly
        "smooth_bins": 3,  # smoothing of the spectrum along velocity before the IPS
        # Flow presence per side: edge = 0 where fewer than gate_fraction of the
        # side's pixels exceed the image's Otsu threshold. The published methods
        # assume a spectrum with known flow direction; on a bidirectional display
        # every column has a side without flow, where they would report noise.
        # This common step makes the comparison meaningful (Kathpalia et al.
        # likewise determine the flow direction first). Disable for the raw methods.
        "presence_gate": True,
        "gate_run_fraction": 0.015,
    }

    def spectra(self, side: SideImage, ctx: Context) -> np.ndarray:
        L = side.level.astype(float)
        dr = self.params["dynamic_range_db"]
        S = level_to_power(L, dr) if dr and dr > 0 else L
        n = self.params["smooth_bins"]
        if n and n > 1:
            S = ndimage.uniform_filter1d(S, n, axis=0, mode="nearest")
        return S

    def gate(self, side: SideImage, ctx: Context, k: np.ndarray) -> np.ndarray:
        """Edge = 0 on column sides without flow.

        Flow forms vertical runs of supra-threshold pixels (threshold: the
        image's Otsu level, on the display smoothed over 3 rows); speckle noise
        rarely does. A side holds flow when its longest run reaches
        max(3, gate_run_fraction * depth) rows.
        """
        if not self.params["presence_gate"]:
            return k
        L = ndimage.uniform_filter1d(side.level.astype(float), 3, axis=0, mode="nearest")
        M = L >= ctx.noise.otsu
        D = M.shape[0]
        need = max(3, int(round(self.params["gate_run_fraction"] * D)))
        # longest run per column via run-length counting
        run = np.zeros(M.shape[1], int)
        best = np.zeros(M.shape[1], int)
        for r in range(D):
            run = np.where(M[r], run + 1, 0)
            best = np.maximum(best, run)
        k = k.copy()
        k[best < need] = 0.0
        return k

    def edge_side(self, side: SideImage, ctx: Context):
        k, conf = super().edge_side(side, ctx)
        return self.gate(side, ctx, k), conf


class PercentileEstimator(_IPSEstimator):
    """Percentile / integrated-power method: edge where the normalised IPS reaches p."""

    name = "percentile"
    defaults = {**_IPSEstimator.common, "percentile": 0.95}

    def edge_column(self, s, ctx):
        P = np.cumsum(s)
        if P[-1] <= 0:
            return 0.0
        return float(np.searchsorted(P / P[-1], self.params["percentile"]))


class GMEstimator(_IPSEstimator):
    """Geometric method: reference line from the IPS point at peak spectral power to f_H."""

    name = "gm"
    defaults = dict(_IPSEstimator.common)

    def edge_column(self, s, ctx):
        P = np.cumsum(s)
        if P[-1] <= 0:
            return 0.0
        return float(_max_distance_above_line(P, int(np.argmax(s)), len(P) - 1))


class MGMEstimator(_IPSEstimator):
    """Modified geometric method: reference line from the first to the last IPS point."""

    name = "mgm"
    defaults = dict(_IPSEstimator.common)

    def edge_column(self, s, ctx):
        P = np.cumsum(s)
        if P[-1] <= 0:
            return 0.0
        return float(_max_distance_above_line(P, 0, len(P) - 1))


def _snsi_fmax(P: np.ndarray, s: np.ndarray, knee_end: int, fS: int, sig_lo: int,
               x_frac: float, slope_half: int) -> int:
    """Point in the knee region [fS, knee_end] whose IPS slope is closest to
    m = x*mS + (1-x)*mN (Steinman et al.; x = 0.1)."""
    fH = len(P) - 1
    mS = _ls_slope(P, sig_lo, fS)
    knee_end = int(np.clip(knee_end, fS + 1, fH))
    mN = (P[fH] - P[knee_end]) / max(fH - knee_end, 1)
    target = x_frac * mS + (1 - x_frac) * mN
    slope = _local_slope(P, slope_half)
    seg = slope[fS:knee_end + 1]
    return int(fS + np.argmin(np.abs(seg - target)))


class SNSIEstimator(_IPSEstimator):
    """Signal-noise slope intersection (Steinman et al. 2000, as in Kathpalia 2016).

    Strong signal = full width at 70% of the spectral peak; f_S = its end.
    Noise region starts at f_S + D/4 (D = f_H - f_S); mS = LS slope of the IPS
    over the strong-signal range; mN = slope of the line joining the IPS at the
    noise start and at f_H; f_max = knee point where the IPS slope is closest
    to 0.1*mS + 0.9*mN.
    """

    name = "snsi"
    defaults = {**_IPSEstimator.common, "strong_fraction": 0.7, "x_fraction": 0.1,
                "noise_start_fraction": 0.25, "slope_half_window": 1}

    def edge_column(self, s, ctx):
        P = np.cumsum(s)
        if P[-1] <= 0:
            return 0.0
        lo, fS = _strong_signal_run(s, self.params["strong_fraction"])
        fH = len(P) - 1
        if fS >= fH - 2:
            return float(fS)
        fN = int(round(fS + self.params["noise_start_fraction"] * (fH - fS)))
        return float(_snsi_fmax(P, s, fN, fS, lo, self.params["x_fraction"], self.params["slope_half_window"]))


class AdaptiveSNSIEstimator(_IPSEstimator):
    """Kathpalia et al. 2016: SNSI with the knee centre located by an altered GM,
    plus envelope post-processing with adaptive noise estimation.

    Per column (section II-A):
      1. f_S = end of the strong-signal (70%) region.
      2. knee centre = IPS point farthest from the line IPS(f_S)->IPS(f_H).
      3. noise start = f_S + 2*(knee - f_S) (knee centre is the midpoint).
      4. mS, mN as in SNSI; f_max = knee-region point with slope closest to
         0.1*mS + 0.9*mN.
    Envelope (section II-B):
      - one-third condition: detections whose knee centre lies beyond
        f_S + D/3 are excluded;
      - adaptive power threshold: 90th percentile of the power in bins above
        the first-estimate envelope over the first `noise_window_s` seconds;
        time points whose mean spectral power is below it are excluded;
      - excluded runs >= 0.1 s are set to zero, other gaps interpolated;
      - moving average of 7.5 ms.
    Not reproducible from an image: manual aliasing correction by baseline
    shift (not applied), and flow direction from the mean frequency (both
    sides are estimated separately here).
    """

    name = "asnsi"
    defaults = {**_IPSEstimator.common, "strong_fraction": 0.7, "x_fraction": 0.1,
                "one_third": True, "power_threshold": True, "noise_percentile": 90.0,
                "noise_window_s": 2.0, "zero_gap_s": 0.1, "smooth_s": 0.0075,
                "slope_half_window": 1}

    def _first_estimate(self, s):
        P = np.cumsum(s)
        if P[-1] <= 0:
            return 0.0, True
        lo, fS = _strong_signal_run(s, self.params["strong_fraction"])
        fH = len(P) - 1
        if fS >= fH - 2:
            return float(fS), True
        knee = _max_distance_above_line(P, fS, fH)
        ok = (not self.params["one_third"]) or (knee - fS <= (fH - fS) / 3.0)
        fN = min(fH - 1, fS + 2 * (knee - fS))
        fN = max(fN, fS + 1)
        return float(_snsi_fmax(P, s, fN, fS, lo, self.params["x_fraction"], self.params["slope_half_window"])), ok

    def edge_side(self, side: SideImage, ctx: Context):
        S = self.spectra(side, ctx)
        D, W = S.shape
        k = np.full(W, np.nan)
        ok = np.zeros(W, bool)
        data = ~side.sweep
        for c in np.flatnonzero(data):
            k[c], ok[c] = self._first_estimate(S[:, c])
        good = ok & data
        if self.params["power_threshold"] and good.any():
            # noise bins: above the first-estimate envelope, first seconds of data
            dt = ctx.spec.dt
            cols = np.flatnonzero(good)
            order = ctx.spec.order
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            first = cols[rank[cols] <= rank[cols].min() + self.params["noise_window_s"] / dt]
            noise_vals = [S[int(k[c]) + 1:, c] for c in first if int(k[c]) + 1 < D]
            noise_vals = np.concatenate(noise_vals) if noise_vals else np.array([0.0])
            thr = np.percentile(noise_vals, self.params["noise_percentile"])
            good &= S.mean(axis=0) >= thr
        k_out = self._fill(k, good, data, ctx)
        return self.gate(side, ctx, k_out), good.astype(float)

    def _fill(self, k, good, data, ctx):
        """Zero long gaps, interpolate short ones, 7.5 ms moving average (in time order)."""
        spec = ctx.spec
        o = spec.order
        kk, gg, dd = k[o].copy(), good[o], data[o]
        n_zero = max(1, int(round(self.params["zero_gap_s"] / spec.dt)))
        x = np.arange(len(kk))
        out = np.full(len(kk), np.nan)
        # process each contiguous data segment separately
        m = np.concatenate([[False], dd, [False]]).astype(np.int8)
        starts = np.flatnonzero(np.diff(m) == 1)
        ends = np.flatnonzero(np.diff(m) == -1)
        for s0, e0 in zip(starts, ends):
            seg_g = gg[s0:e0]
            seg = kk[s0:e0].copy()
            seg[~seg_g] = np.nan
            bad = np.isnan(seg)
            # zero long gaps
            bm = np.concatenate([[False], bad, [False]]).astype(np.int8)
            for a, b in zip(np.flatnonzero(np.diff(bm) == 1), np.flatnonzero(np.diff(bm) == -1)):
                if b - a >= n_zero:
                    seg[a:b] = 0.0
            bad = np.isnan(seg)
            if (~bad).sum() >= 2:
                xs = x[s0:e0]
                seg[bad] = np.interp(xs[bad], xs[~bad], seg[~bad])
            elif bad.all():
                seg[:] = 0.0
            else:
                seg[bad] = seg[~bad][0]
            n_ma = max(1, int(round(self.params["smooth_s"] / spec.dt)))
            if n_ma > 1:
                seg = ndimage.uniform_filter1d(seg, n_ma, mode="nearest")
            out[s0:e0] = seg
        res = np.full_like(k, np.nan)
        res[o] = out
        return res


class QIAEstimator(_IPSEstimator):
    """Quadratic iteration algorithm (Li et al. 2019).

    1. P = IPS of the spectrum.
    2. Reference line from the first to the last point of P; the crossing of P
       with it (from below to above) is the 'cross point' (max energy point).
    3. S_lowest = minimum of the spectrum between the start and the cross
       point; dV = V_cross - S_lowest; keep P on [V_cross - dV, V_cross + dV].
    4. New reference line through the ends of that segment: the point of
       maximum positive vertical distance is V_max, of maximum negative
       distance V_min.

    mode='per_side' (default) applies it to each one-sided spectrum and uses
    V_max as the outer edge. mode='full_column' applies it once to the whole
    column (most negative -> most positive velocity), as in the paper's
    bidirectional use: V_max gives the upper edge and V_min the lower edge.
    The paper's final smoothing is not specified; use the post-processing stage.
    """

    name = "qia"
    defaults = {**_IPSEstimator.common, "mode": "per_side"}

    @staticmethod
    def qia(s: np.ndarray) -> tuple[int, int]:
        P = np.cumsum(s)
        n = len(P)
        if n < 3 or P[-1] <= 0:
            return 0, 0
        x = np.arange(n)
        d = P - (P[0] + x * (P[-1] - P[0]) / (n - 1))
        up = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
        vc = int(up[0] + 1) if len(up) else int(np.argmax(s))
        s_low = int(np.argmin(s[:vc + 1]))
        dv = max(vc - s_low, 1)
        a, b = max(0, vc - dv), min(n - 1, vc + dv)
        if b - a < 2:
            return vc, vc
        xs = np.arange(a, b + 1)
        dd = P[a:b + 1] - (P[a] + (xs - a) * (P[b] - P[a]) / (b - a))
        return int(a + np.argmax(dd)), int(a + np.argmin(dd))

    def edge_column(self, s, ctx):
        return float(self.qia(s)[0])

    def estimate(self, ctx: Context):
        if self.params["mode"] != "full_column":
            return super().estimate(ctx)
        spec = ctx.spec
        pos = side_view(spec, "pos")
        neg = side_view(spec, "neg")
        # full column ordered from most negative to most positive velocity
        Sp = self.spectra(pos, ctx)
        Sn = self.spectra(neg, ctx)
        W = Sp.shape[1]
        kp = np.full(W, np.nan)
        kn = np.full(W, np.nan)
        nn = Sn.shape[0]
        for c in np.flatnonzero(~spec.sweep):
            col = np.concatenate([Sn[::-1, c], Sp[1:, c]])  # index nn-1 = zero velocity
            vmax, vmin = self.qia(col)
            kp[c] = max(vmax - (nn - 1), 0)
            kn[c] = max((nn - 1) - vmin, 0)
        kp = self.gate(pos, ctx, kp)
        kn = self.gate(neg, ctx, kn)
        return self._result(kp, kn)
