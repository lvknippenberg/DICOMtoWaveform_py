"""ECG extraction from the rendered trace, and R-peak detection.

None of the example DICOMs stores the ECG as a WaveformSequence; it is only
burned into the pixels, along the lower edge of the B-mode region, sharing
the columns (and therefore the time axis) of the spectral display. It has no
amplitude calibration, so amplitudes are arbitrary units.

Differences from the MATLAB implementation (DICOM2waveform.m):
- The trace colour is searched in a band of rows rather than on one row, so a
  trace that does not cross the B-mode lower edge is still found.
- Per column, the sample is the top or bottom of the drawn stroke at local
  extremes (so R and S peaks keep their full amplitude) and the stroke centre
  elsewhere, instead of always the top pixel.
- Where several strokes share a column (greyscale images, where Doppler
  pixels can have the trace's grey value), the one continuing the trace from
  its neighbours is taken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .config import ECGConfig
from .io import DicomImage, Region
from .render import _colour_match, _runs


@dataclass
class ECGTrace:
    row: np.ndarray  # (W,) float row of the trace in full-image coordinates, NaN = none
    mask: np.ndarray  # (H_img, W_img) bool pixels of the drawn trace
    colour: np.ndarray | None
    found: bool
    clipped: bool = False


def _band_rows(img: DicomImage, spec: Region, bmode: Region | None, cfg: ECGConfig) -> tuple[int, int]:
    H = img.pixels.shape[0]
    top_ref = bmode.y1 if bmode is not None else spec.y0
    lo = max(0, top_ref - cfg.search_above_px)
    hi = min(H - 1, spec.y0 + cfg.search_margin_px)
    return lo, hi


def _find_trace_colour(band: np.ndarray, is_grey_image: bool, W: int, centre_row: int,
                       exclude: tuple = ()) -> np.ndarray | None:
    """Colour of the ECG trace in a band of rows (band: (h, W, 3)).

    The trace is the colour that spans most columns without being a
    horizontal line (the drawn baseline also spans every column).
    """
    ch = band.astype(np.int16)
    chroma = ch.max(axis=2) - ch.min(axis=2)
    if is_grey_image:
        # Grey image: the ECG is the most common non-black value on the rows
        # around the B-mode lower edge (as in the MATLAB implementation).
        lo, hi = max(0, centre_row - 3), min(band.shape[0], centre_row + 4)
        vals = band[lo:hi, :, 0].ravel()
        vals = vals[vals > 30]
        if len(vals) == 0:
            return None
        u, c = np.unique(vals, return_counts=True)
        cand = u[np.argsort(c)[::-1][:5]]
    else:
        px = band[chroma > 40]
        if len(px) == 0:
            return None
        u, c = np.unique(px, axis=0, return_counts=True)
        cand = u[np.argsort(c)[::-1][:8]]
    best, best_cov = None, 0.0
    for col in cand:
        colour = np.array([col, col, col]) if np.ndim(col) == 0 else np.asarray(col)
        if any(np.all(np.abs(colour.astype(int) - np.asarray(e, int)) <= 6) for e in exclude):
            continue
        m = _colour_match(band, colour, 4)
        cov = m.any(axis=0).mean()
        # A horizontal line (the drawn baseline) occupies only a few rows
        # overall; even a flat ECG has QRS deflections spanning many rows.
        rows = np.flatnonzero(m.sum(axis=1) > 0)
        line_like = len(rows) == 0 or (rows.max() - rows.min()) < 6
        if cov > best_cov and not line_like:
            best, best_cov = colour, cov
    if best is None or best_cov < 0.2:
        return None
    return best.astype(int)


def _column_samples(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """One row per column from a (h, W) trace mask."""
    h, W = mask.shape
    runs_per_col = [_runs(mask[:, c]) for c in range(W)]
    centre = np.full(W, np.nan)
    for c, runs in enumerate(runs_per_col):
        if len(runs) == 1:
            centre[c] = 0.5 * (runs[0][0] + runs[0][1])
    # Resolve columns with several runs by continuity with the neighbours
    known = np.flatnonzero(~np.isnan(centre))
    for c, runs in enumerate(runs_per_col):
        if len(runs) > 1:
            if len(known):
                ref = np.interp(c, known, centre[known])
            else:
                ref = h / 2
            s, e = min(runs, key=lambda r: min(abs(r[0] - ref), abs(r[1] - ref)))
            centre[c] = 0.5 * (s + e)
            runs_per_col[c] = [(s, e)]
    y = centre.copy()
    for c in range(W):
        if not runs_per_col[c]:
            continue
        s, e = runs_per_col[c][0]
        left = centre[c - 1] if c > 0 else np.nan
        right = centre[c + 1] if c < W - 1 else np.nan
        nb = np.array([left, right])
        nb = nb[~np.isnan(nb)]
        if len(nb) == 0 or e - s < 2:
            continue
        if s < nb.min() - 1:  # stroke reaches above both neighbours: upward peak
            y[c] = s
        elif e > nb.max() + 1:  # reaches below both neighbours: downward peak
            y[c] = e
    # Interpolate short gaps
    valid = ~np.isnan(y)
    if valid.sum() >= 2:
        for s, e in _runs(~valid):
            if s == 0 or e == W - 1 or e - s + 1 > max_gap:
                continue
            y[s:e + 1] = np.interp(np.arange(s, e + 1), [s - 1, e + 1], [y[s - 1], y[e + 1]])
    return y


def extract_ecg(img: DicomImage, spec: Region, bmode: Region | None, cfg: ECGConfig,
                exclude_colours: tuple = ()) -> ECGTrace:
    """Locate the ECG trace over the columns of the spectral region."""
    px = img.pixels
    rgb = px if px.ndim == 3 else np.repeat(px[..., None], 3, axis=2)
    is_grey = px.ndim == 2 or bool(np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))
    lo, hi = _band_rows(img, spec, bmode, cfg)
    band = rgb[lo:hi + 1, spec.x0:spec.x1 + 1]
    W = band.shape[1]
    centre_row = (bmode.y1 - 1 - lo) if bmode is not None else (spec.y0 - lo)
    colour = _find_trace_colour(band, is_grey, W, centre_row, exclude_colours)
    full_mask = np.zeros(px.shape[:2], bool)
    if colour is None:
        return ECGTrace(row=np.full(W, np.nan), mask=full_mask, colour=None, found=False)

    m = _colour_match(band, colour, cfg.colour_tolerance)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n:
        objs = ndimage.find_objects(lab)
        widths = np.array([sl[1].stop - sl[1].start for sl in objs])
        keep = widths >= cfg.min_component_width_frac * W
        if not keep.any():
            keep = widths == widths.max()
        if not is_grey:
            # Colour image: the trace colour is unique, so also keep small
            # pieces within the vertical band of the main trace.
            rows_main = np.concatenate([np.arange(objs[i][0].start, objs[i][0].stop) for i in np.flatnonzero(keep)])
            r0, r1 = rows_main.min(), rows_main.max()
            for i, sl in enumerate(objs):
                if not keep[i] and sl[0].start >= r0 - 2 and sl[0].stop <= r1 + 3:
                    keep[i] = True
        m = np.isin(lab, np.flatnonzero(keep) + 1)
    max_gap = max(1, int(round(cfg.max_gap_s / spec.dx)))
    y = _column_samples(m, max_gap)
    full_mask[lo:hi + 1, spec.x0:spec.x1 + 1] = m
    found = bool(np.mean(~np.isnan(y)) > 0.2)
    # The scanner clips the drawn trace to its display box: flat runs at the
    # extreme row are clipped R (or S) waves.
    clipped = False
    if found:
        yr = y[~np.isnan(y)]
        clipped = bool(max(np.sum(yr <= yr.min() + 0.5), np.sum(yr >= yr.max() - 0.5)) > 0.02 * len(yr))
    return ECGTrace(row=y + lo, mask=full_mask, colour=colour, found=found, clipped=clipped)


def ecg_amplitude(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trace rows -> (a.u. normalised, pixels upward). Isoelectric level ~ 0."""
    up = -(row - np.nanmedian(row)) if np.any(~np.isnan(row)) else row.copy()
    scale = np.nanmax(np.abs(up)) if np.any(~np.isnan(up)) else 1.0
    return up / (scale if scale > 0 else 1.0), up


# ---------------------------------------------------------------------------
# R-peak detection: port of DetectRpeaks.m (QRS-band energy, Pan-Tompkins style)
# ---------------------------------------------------------------------------

def _robust_std(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) + np.finfo(float).eps


def _find_peaks(E: np.ndarray, threshold: float, min_dist: int) -> tuple[np.ndarray, np.ndarray]:
    if len(E) == 0 or E.max() <= threshold:
        return np.array([]), np.array([], int)
    idx, props = signal.find_peaks(E, height=threshold, distance=max(1, min_dist))
    return props["peak_heights"], idx


def detect_rpeaks(ecg: np.ndarray, t: np.ndarray, cfg: ECGConfig, heart_rate_hint: float | None = None) -> np.ndarray:
    """R-peak times (s). NaN samples (e.g. the sweep) never produce a beat.

    heart_rate_hint (bpm, e.g. the DICOM HeartRate tag) sets the expected RR
    interval for search-back. A strip holds only 2-8 beats, so the median RR
    of the detected beats is a poor reference when a beat is missing.
    """
    valid = ~np.isnan(ecg)
    if valid.sum() < 3:
        return np.array([])
    i0, i1 = np.flatnonzero(valid)[[0, -1]]
    x = ecg[i0:i1 + 1].astype(float)
    ts = t[i0:i1 + 1]
    nodata = np.isnan(x)
    if nodata.any():
        k = np.arange(len(x))
        x[nodata] = np.interp(k[nodata], k[~nodata], x[~nodata])
    fs = 1.0 / np.mean(np.diff(ts))

    lo, hi = cfg.qrs_band_hz
    try:
        if hi >= fs / 2:
            raise ValueError
        sos = signal.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xb = signal.sosfiltfilt(sos, x)
    except ValueError:
        w = max(3, int(round(fs / lo)))
        xb = x - ndimage.uniform_filter1d(x, w, mode="nearest")
    E = (xb / _robust_std(xb)) ** 2
    E = ndimage.uniform_filter1d(E, max(1, int(round(cfg.energy_smooth_s * fs))), mode="nearest")
    E = E / (E.max() + np.finfo(float).eps)

    min_dist = int(round(cfg.refractory_s * fs))
    floor = max(np.median(E) * 2, 0.005)
    heights, locs = _find_peaks(E, floor, min_dist)
    if len(locs) == 0:
        return np.array([])
    qrs_level = np.median(heights[heights >= np.median(heights)])
    noise_floor = cfg.noise_floor_fraction * qrs_level
    thr = max(cfg.thresh_fraction * qrs_level, noise_floor)
    keep = heights >= thr
    heights, locs = heights[keep], locs[keep]

    # Search-back in abnormally long RR gaps
    if len(locs) >= 2 or (heart_rate_hint and len(locs) >= 1):
        rr_ref = np.median(np.diff(locs)) if len(locs) >= 2 else np.inf
        if heart_rate_hint and heart_rate_hint > 0:
            rr_ref = min(rr_ref, 60.0 / heart_rate_hint * fs)
        gap_thr = cfg.searchback_gap_ratio * rr_ref
        # Include the stretches before the first and after the last beat
        bounds = np.concatenate([[-min_dist], locs, [len(E) - 1 + min_dist]])
        add_l, add_h = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            is_edge = a < 0 or b >= len(E)
            if (b - a > gap_thr) and not (is_edge and not heart_rate_hint):
                s, e = max(0, a + min_dist), min(len(E) - 1, b - min_dist)
                if e - s >= 3:
                    h, l = _find_peaks(E[s:e + 1], 0.5 * noise_floor, min_dist)
                    l = l + s
                    if len(l) == 0:
                        continue
                    # Pan-Tompkins search-back adds the single largest peak;
                    # prefer candidates at a plausible RR from a neighbour.
                    plaus = np.zeros(len(l), bool)
                    if np.isfinite(rr_ref):
                        for nb in (a, b):
                            if 0 <= nb < len(E):
                                plaus |= np.abs(np.abs(l - nb) - rr_ref) <= 0.3 * rr_ref
                    pool = np.flatnonzero(plaus) if plaus.any() else np.arange(len(l))
                    j = pool[int(np.argmax(h[pool]))]
                    add_l.append(int(l[j]))
                    add_h.append(float(h[j]))
        if add_l:
            locs = np.concatenate([locs, add_l]).astype(int)
            heights = np.concatenate([heights, add_h])
            o = np.argsort(locs)
            locs, heights = locs[o], heights[o]
            kept = np.zeros(len(locs), bool)
            last = -1
            for k in range(len(locs)):
                if last < 0 or locs[k] - locs[last] >= min_dist:
                    kept[k] = True
                    last = k
                elif heights[k] > heights[last]:
                    kept[last] = False
                    kept[k] = True
                    last = k
            locs, heights = locs[kept], heights[kept]

    # T-wave guard (RR-relative window)
    if len(locs) >= 2:
        med_h = np.median(heights)
        keep = np.ones(len(locs), bool)
        last = 0
        for k in range(1, len(locs)):
            rr = np.median(np.diff(locs[keep]))
            if not np.isfinite(rr):
                break
            d = locs[k] - locs[last]
            f0, f1 = cfg.twave_rr_fraction
            if (f0 * rr < d < f1 * rr and heights[k] < cfg.twave_relative_height * heights[last]
                    and heights[k] < cfg.twave_relative_height * med_h):
                keep[k] = False
            else:
                last = k
        locs = locs[keep]

    locs = locs[~nodata[locs]]
    # Refine to the local maximum of |ECG| deflection within +-40 ms
    w = max(1, int(round(0.04 * fs)))
    xc = x - np.median(x)
    refined = []
    for l in locs:
        s, e = max(0, l - w), min(len(x), l + w + 1)
        refined.append(s + int(np.argmax(np.abs(xc[s:e]))))
    return ts[np.array(refined, int)] if refined else np.array([])
