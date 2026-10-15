"""Quality indicators: image-domain SNR proxy, aliasing and coverage flags.

The SNR here is an *image-domain SNR proxy*: the ratio of mean de-compressed
display power inside the detected flow region to that outside it. It is not
an acquisition SNR.
"""

from __future__ import annotations

import numpy as np

from .config import AliasingConfig
from .envelope.base import side_view
from .render import Spectrogram, level_to_power


def flow_mask(spec: Spectrogram, k_pos: np.ndarray, k_neg: np.ndarray) -> np.ndarray:
    """(H, W) mask of the flow region between the two outer edges."""
    H, W = spec.level.shape
    M = np.zeros((H, W), bool)
    for side, kk in (("pos", k_pos), ("neg", k_neg)):
        sv = side_view(spec, side)
        edge = np.nan_to_num(kk, nan=-1.0)
        inside = (np.arange(sv.depth)[:, None] <= edge[None, :]) & (edge[None, :] > 0)
        M[sv.rows] |= inside
    M[:, spec.sweep] = False
    return M


def snr_proxy_db(spec: Spectrogram, flow: np.ndarray, dynamic_range_db: float = 40.0) -> float:
    valid = spec.valid & ~spec.sweep[None, :]
    P = level_to_power(spec.level, dynamic_range_db)
    sig = P[flow & valid]
    bg = P[~flow & valid]
    if len(sig) < 10 or len(bg) < 10:
        return float("nan")
    return float(10 * np.log10(max(sig.mean(), 1e-12) / max(bg.mean(), 1e-12)))


def aliasing_suspected(spec: Spectrogram, k_pos: np.ndarray, k_neg: np.ndarray, cfg: AliasingConfig) -> bool:
    """PW aliasing wraps flow around the display: where it happens, flow leaves
    one velocity limit and re-enters at the opposite one, so in the same
    columns both outer edges reach their display limits. (One edge alone at
    its limit is common: e.g. a mirror/clutter texture filling a small side.)
    """
    if spec.modality == "CW":
        return False
    data = ~spec.sweep
    if not data.any():
        return False
    touch = []
    for side, kk in (("pos", k_pos), ("neg", k_neg)):
        D = side_view(spec, side).depth
        touch.append(np.nan_to_num(kk, nan=0) >= (1 - cfg.alias_edge_fraction) * (D - 1))
    both = touch[0] & touch[1]
    return bool(both[data].mean() > cfg.alias_min_fraction)
