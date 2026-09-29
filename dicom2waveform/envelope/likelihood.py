"""Noise statistics and soft signal membership (Doppler-energy likelihood).

The rendered display is nearly binary (noise ~ a few grey levels, signal
~200) and log-compressed, so gain acts approximately as an additive offset.
Thresholds are therefore referenced to the measured noise floor rather than
to fixed grey levels. Two estimates of where signal starts are combined,
taking the lower:

- a noise margin (median + max(margin, k * robust SD)), which sits just above
  the noise and keeps the faint outer edge of the spectrum, but fails on very
  dense spectra where the median and MAD no longer describe the noise;
- Otsu's threshold, read from the histogram, which survives dense spectra but
  is too high on clean displays.

Membership ramps linearly from 0 just above the noise floor to 1 at that
threshold, so faint flow counts partially instead of being cut (the key
refinement of the MATLAB DP method).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..config import LikelihoodConfig


@dataclass
class NoiseStats:
    level: float  # noise level (median of the image)
    spread: float  # robust SD of the noise
    otsu: float
    threshold: float  # membership reaches 1 here
    noise_top: float  # membership starts rising here


def otsu_threshold(values: np.ndarray, nbins: int = 256, value_range=(0.0, 255.0)) -> tuple[float, float]:
    """Otsu threshold and its effectiveness metric (between-class / total variance)."""
    values = np.asarray(values, float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), 0.0
    hist, edges = np.histogram(values, bins=nbins, range=value_range)
    p = hist.astype(float) / max(hist.sum(), 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    sb[~np.isfinite(sb)] = 0
    i = int(np.argmax(sb))
    var_t = float(np.sum(p * (centres - mu_t) ** 2))
    em = float(sb[i] / var_t) if var_t > 0 else 0.0
    return float(edges[i + 1]), em


def noise_stats(spec, cfg: LikelihoodConfig) -> NoiseStats:
    vals = spec.level[spec.valid]
    if len(vals) == 0:
        return NoiseStats(0.0, 1.0, 128.0, 128.0, 0.0)
    level = float(np.median(vals))
    spread = float(1.4826 * np.median(np.abs(vals - level)))
    otsu, _ = otsu_threshold(vals)
    margin_thr = level + max(cfg.noise_margin, cfg.noise_spread_factor * spread)
    threshold = min(margin_thr, otsu)
    noise_top = level + cfg.soft_low_factor * max(spread, 1.0)
    return NoiseStats(level, spread, otsu, threshold, noise_top)


def membership(level: np.ndarray, ns: NoiseStats) -> np.ndarray:
    """Soft signal membership in [0, 1]."""
    return np.clip((level - ns.noise_top) / max(ns.threshold - ns.noise_top, 1.0), 0.0, 1.0)


def density_window(spec, cfg: LikelihoodConfig) -> tuple[int, int]:
    rows = cfg.density_window_px_rows
    if cfg.density_window_cm_s > 0:
        rows = max(1, int(round(cfg.density_window_cm_s / abs(spec.dy))))
    cols = cfg.density_window_px_cols
    if cfg.density_window_s > 0:
        cols = max(1, int(round(cfg.density_window_s / spec.dt)))
    return rows, cols


def local_density(m: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Moving mean of membership over (rows, cols)."""
    return ndimage.uniform_filter(m.astype(float), size=window, mode="nearest")


def likelihood_image(spec, cfg: LikelihoodConfig, ns: NoiseStats | None = None) -> np.ndarray:
    """Local density of soft membership, i.e. P(pixel is inside the flow region)."""
    ns = ns or noise_stats(spec, cfg)
    m = membership(spec.level, ns)
    m[:, spec.sweep] = 0.0
    return local_density(m, density_window(spec, cfg))
