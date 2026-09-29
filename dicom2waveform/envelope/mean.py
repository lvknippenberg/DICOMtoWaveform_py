"""Mean velocity envelopes, computed inside the detected flow region.

For each column, pixels between the zero-velocity row and the outer edge on
each side form the flow region. The mean velocity is the weighted average of
their velocities:

    v_mean_pos = sum_pos w v / sum_pos w      (positive half only)
    v_mean_neg = sum_neg w v / sum_neg w      (negative half only)
    v_mean     = sum_all w v / sum_all w      (both halves, signed)

Restricting to the flow region stops the noise outside it from pulling the
mean toward the centre of the display. Weights (MeanConfig.weighting):

- 'power': de-compressed display level minus the de-compressed noise level,
  an image-domain proxy of the power-weighted mean frequency (the physical
  definition of mean velocity);
- 'level': display level minus the noise level (log domain);
- 'membership': soft signal membership.

This is an image-domain estimate, not the scanner's Vmean. A side without flow
(edge at the baseline) gets mean 0.
"""

from __future__ import annotations

import numpy as np

from ..config import MeanConfig
from ..render import Spectrogram, level_to_power
from .base import side_view
from .likelihood import NoiseStats, membership


def pixel_weights(spec: Spectrogram, ns: NoiseStats, cfg: MeanConfig) -> np.ndarray:
    L = spec.level.astype(float)
    if cfg.weighting == "power":
        w = level_to_power(L, cfg.dynamic_range_db) - level_to_power(ns.level, cfg.dynamic_range_db)
    elif cfg.weighting == "level":
        w = L - ns.level
    elif cfg.weighting == "membership":
        w = membership(L, ns)
    else:
        raise ValueError(f"unknown weighting '{cfg.weighting}'")
    w = np.clip(w, 0.0, None)
    w[~spec.valid] = 0.0
    return w


def mean_envelopes(spec: Spectrogram, k_pos: np.ndarray, k_neg: np.ndarray,
                   ns: NoiseStats, cfg: MeanConfig) -> dict[str, np.ndarray]:
    """v_mean, v_mean_pos, v_mean_neg (cm/s) per column; NaN in the sweep."""
    w = pixel_weights(spec, ns, cfg)
    W = spec.level.shape[1]
    sums = {}
    k_clutter = cfg.clutter_band_cm_s / abs(spec.dy)
    for side, kk in (("pos", k_pos), ("neg", k_neg)):
        sv = side_view(spec, side, image=w)
        D = sv.depth
        k = np.arange(D)[:, None]
        edge = np.nan_to_num(kk, nan=-1.0)[None, :]
        inside = (k <= edge) & (k >= max(k_clutter, 0.0)) & (edge > 0)
        # the zero row itself carries no directional information
        inside &= k >= 1
        ww = np.where(inside, sv.level, 0.0)
        n = inside.sum(axis=0)
        sw = ww.sum(axis=0)
        swv = (ww * sv.velocity(np.arange(D))[:, None]).sum(axis=0)
        sums[side] = (sw, swv, n)
    out = {}
    for side in ("pos", "neg"):
        sw, swv, n = sums[side]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where((sw > 0) & (n >= cfg.min_pixels), swv / sw, 0.0)
        out[f"v_mean_{side}"] = m
    sw = sums["pos"][0] + sums["neg"][0]
    swv = sums["pos"][1] + sums["neg"][1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["v_mean"] = np.where(sw > 0, swv / sw, 0.0)
    for key in out:
        out[key] = out[key].astype(float)
        out[key][spec.sweep] = np.nan
    return out
