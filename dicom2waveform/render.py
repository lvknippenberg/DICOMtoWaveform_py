"""From the rendered spectral display to a calibrated scalar intensity image.

The DICOM holds a *rendering* of the Doppler spectrogram, not the spectrum:
grey levels (or a colour map for TDI) after log compression, with overlays
drawn on top. This module

1. crops the spectral region given by the header,
2. finds the overlays (drawn baseline, time markers, ECG trace, text) and
   marks them as missing rather than as signal or background,
3. converts colour to a scalar display level (0..255),
4. finds the sweep (write position of the circular display buffer), which
   holds no data,
5. attaches the calibration: velocity per row, time per column.

Intensities are display levels. They are monotonic in Doppler power, but only
a proxy for it: see `level_to_power`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from .config import RenderConfig
from .io import DicomImage, Region

# Luminance of the brightest colour of the Philips copper TDI map is ~150;
# scale luminance so that colour-map images use roughly the same 0..255 range
# as greyscale images (noise margins are in grey levels).
COLORMAP_LEVEL_SCALE = 255.0 / 160.0

LUMA = np.array([0.299, 0.587, 0.114])


@dataclass
class Spectrogram:
    """Calibrated spectral Doppler image.

    Row r has velocity (r - zero_row) * dy (cm/s); column c has time c * dt (s)
    before any sweep alignment. All arrays are in raw display column order;
    `order` gives the permutation to oldest -> newest.
    """

    level: np.ndarray  # (H, W) float32 display level 0..255 (overlays filled)
    valid: np.ndarray  # (H, W) bool: False on overlays and in the sweep
    zero_row: float
    dy: float  # cm/s per row downwards (negative: velocity increases upwards)
    dt: float  # s per column
    sweep: np.ndarray  # (W,) bool: column holds no data
    order: np.ndarray  # (W,) int: column order oldest -> newest
    colormap: str  # 'grey' or 'colour'
    modality: str  # 'PW', 'CW' or 'TDI'
    region: Region
    baseline_rows: np.ndarray  # rows covered by the drawn baseline
    rgb: np.ndarray | None = None  # cropped display for plotting (H, W, 3)
    overlay: np.ndarray | None = None  # (H, W) bool overlay mask
    info: dict[str, Any] = field(default_factory=dict)
    clicks: np.ndarray | None = None  # (W,) bool: valve-click columns
    click_strength: np.ndarray | None = None  # (W,) transient height (robust SDs)
    click_mode: str = "bridge"

    @property
    def shape(self) -> tuple[int, int]:
        return self.level.shape

    @property
    def velocity(self) -> np.ndarray:
        return (np.arange(self.level.shape[0]) - self.zero_row) * self.dy

    @property
    def v_limits(self) -> tuple[float, float]:
        v = self.velocity
        return float(v.min()), float(v.max())

    @property
    def display_height(self) -> int:
        """Rows of the displayed spectral region (unchanged by aliasing unwrapping)."""
        return int(self.region.height)

    @property
    def display_v_limits(self) -> tuple[float, float]:
        """Velocity limits of the display as rendered by the scanner (the Nyquist
        window for PW), also after aliasing unwrapping."""
        z0 = self.info.get("display_zero_row", self.zero_row)
        v = (np.array([0, self.display_height - 1]) - z0) * self.dy
        return float(v.min()), float(v.max())

    def row_of_velocity(self, v: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(v) / self.dy + self.zero_row


def _colour_match(rgb: np.ndarray, colour: np.ndarray, tol: int) -> np.ndarray:
    return np.all(np.abs(rgb.astype(np.int16) - np.asarray(colour, np.int16)) <= tol, axis=-1)


def _dominant_colour(pixels: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Most common non-black colour among (N, 3) pixels, and its count."""
    px = pixels.reshape(-1, 3)
    px = px[px.sum(1) > 0]
    if len(px) == 0:
        return None, 0
    u, c = np.unique(px, axis=0, return_counts=True)
    i = int(np.argmax(c))
    return u[i].astype(int), int(c[i])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of runs of True in a 1-D mask."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def find_baseline(rgb: np.ndarray, zero_row: float, cfg: RenderConfig) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Drawn zero-velocity line near the header's zero row.

    Returns (rows covered by the line, line colour, per-column presence).
    The line is the most common non-black colour on the rows around
    zero_row; rows where that colour covers more than half the columns belong
    to the line.
    """
    H, W, _ = rgb.shape
    z = int(round(zero_row))
    lo, hi = max(0, z - cfg.baseline_search_rows), min(H - 1, z + cfg.baseline_search_rows)
    best, best_count = None, 0
    for r in range(lo, hi + 1):
        col, cnt = _dominant_colour(rgb[r])
        if col is not None and cnt > best_count:
            best, best_count = col, cnt
    if best is None or best_count < 0.3 * W:
        return np.array([], int), None, np.ones(W, bool)
    rows = [r for r in range(lo, hi + 1)
            if _colour_match(rgb[r], best, cfg.overlay_colour_tolerance).mean() > 0.3]
    rows = np.array(rows, int)
    if len(rows) == 0:
        return rows, None, np.ones(W, bool)
    # Keep the contiguous block closest to zero_row
    blocks = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    rows = min(blocks, key=lambda b: abs(b.mean() - zero_row))
    presence = np.any(_colour_match(rgb[rows], best, cfg.overlay_colour_tolerance), axis=0)
    return rows, best, presence


def find_text(level: np.ndarray, overlay: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    """White annotations (labels, scale ticks) drawn inside the spectral region.

    Both text and saturated flow reach the maximum grey level; text is a small
    bright blob on a dark background, saturated flow (and valve clicks) sit
    inside bright signal. Small bright components whose surroundings are dark
    are masked, dilated by one pixel to cover anti-aliasing.
    """
    bright = (level >= cfg.text_level) & ~overlay
    if not bright.any():
        return np.zeros_like(bright)
    lab, n = ndimage.label(bright, structure=np.ones((3, 3)))
    area = ndimage.sum(bright, lab, index=np.arange(1, n + 1))
    ring = ndimage.binary_dilation(bright, iterations=3) & ~bright & ~overlay
    # label the ring by the nearest component
    near = ndimage.grey_dilation(lab, size=(7, 7))
    with np.errstate(invalid="ignore", divide="ignore"):
        ring_mean = ndimage.mean(level, labels=np.where(ring, near, 0), index=np.arange(1, n + 1))
    ring_mean = np.nan_to_num(np.asarray(ring_mean, float), nan=0.0)
    is_text = (area <= cfg.text_max_area) & (ring_mean <= cfg.text_ring_max)
    mask = np.isin(lab, np.flatnonzero(is_text) + 1)
    return ndimage.binary_dilation(mask, iterations=1)


def find_clicks(level: np.ndarray, sweep: np.ndarray, dt: float, cfg: RenderConfig,
                return_strength: bool = False):
    """Columns holding a valve click (short broadband transient).

    With return_strength, also returns the transient height in robust SDs per
    column (used to rank clicks, e.g. for aortic valve closure)."""
    W = level.shape[1]
    E = level.mean(axis=0)
    E[sweep] = np.nan
    # White top-hat of the column-mean level: E minus its grey opening with a
    # flat window one maximum click-width wide. The opening removes only
    # features narrower than a click and keeps steps, so a flow onset/offset
    # gives ~0 while a click keeps its excess, also when it sits on the flank
    # of a jet (aortic closure at the end of ejection).
    m = max(1, int(round(cfg.click_max_width_s / dt)))
    x = np.arange(W)
    good = ~np.isnan(E)
    if good.sum() < 2 * m:
        z = np.zeros(W, bool)
        return (z, np.zeros(W)) if return_strength else z
    Ef = np.interp(x, x[good], E[good])  # bridge the sweep with the edge levels
    hp = Ef - ndimage.grey_opening(Ef, size=m + 2, mode="nearest")
    hp[sweep] = 0.0
    # the first/last columns of the data cannot be confirmed as transients
    edge = np.zeros(W, bool)
    for a, b in _runs(good):
        edge[a:min(a + m, b + 1)] = True
        edge[max(a, b - m + 1):b + 1] = True
    hp[edge] = 0.0
    sig = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6
    cand = hp > cfg.click_min_snr * sig
    clicks = np.zeros(W, bool)
    for a, b in _runs(cand):
        if b - a + 1 <= m:
            clicks[max(0, a - 1):b + 2] = True
    if return_strength:
        # strength for every column (not only clicks): R-gated detectors such
        # as the aortic valve closure may use a lower threshold inside their window
        return clicks, hp / sig
    return clicks


def bridge_columns(level: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace columns `bad` by linear interpolation between the nearest good columns."""
    out = level.copy()
    good = np.flatnonzero(~bad)
    if len(good) < 2:
        return out
    for a, b in _runs(bad):
        left = a - 1 if a > 0 else None
        right = b + 1 if b < level.shape[1] - 1 else None
        if left is None or right is None or bad[left] or bad[right]:
            continue
        w = (np.arange(a, b + 1) - left) / (right - left)
        out[:, a:b + 1] = level[:, [left]] * (1 - w) + level[:, [right]] * w
    return out


def find_sweep(presence: np.ndarray, min_width: int) -> np.ndarray:
    """Columns without a drawn baseline, in runs at least min_width wide."""
    sweep = np.zeros_like(presence, bool)
    if presence.all():
        return sweep
    for s, e in _runs(~presence):
        if e - s + 1 >= min_width:
            sweep[s:e + 1] = True
    return sweep


def sweep_order(sweep: np.ndarray, align: bool) -> np.ndarray:
    """Column permutation that puts the oldest sample first.

    The display is a circular buffer: the band being written separates the
    newest samples (to its left) from the oldest (to its right). Rotating so
    that the record starts at the sweep makes time monotonic.
    """
    W = len(sweep)
    if not align or not sweep.any():
        return np.arange(W)
    runs = _runs(sweep)
    widths = [e - s + 1 for s, e in runs]
    # the buffer is circular: a sweep crossing the right edge continues at column 0
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == W - 1:
        widths[-1] += widths[0]
        widths[0] = 0
    start = runs[int(np.argmax(widths))][0]
    return np.roll(np.arange(W), -start)


def fill_invalid_columns(level: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid pixels by linear interpolation along each column.

    Thin overlays (baseline, markers, trace) are bridged from the pixels above
    and below. Columns without any valid pixel are left at 0.
    """
    out = level.astype(np.float32).copy()
    H = level.shape[0]
    rows = np.arange(H)
    bad_cols = np.flatnonzero((~valid).any(axis=0))
    for c in bad_cols:
        v = valid[:, c]
        if not v.any():
            out[:, c] = 0.0
            continue
        out[~v, c] = np.interp(rows[~v], rows[v], out[v, c])
    return out


def level_to_power(level: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """Undo log compression approximately: display level -> relative linear power.

    Assumes grey levels 0..255 span `dynamic_range_db` dB. The true mapping
    (dynamic range, reject, gain, post-processing curve) is not in the DICOM,
    so integrated-power methods on the image are an image-domain adaptation,
    not the signal-domain algorithms.
    """
    return np.power(10.0, (np.asarray(level, np.float64) / 255.0) * dynamic_range_db / 10.0) - 1.0


def classify_modality(region: Region, colormap: str, v_range: tuple[float, float]) -> str:
    if region.data_type == 4:
        return "CW"
    span = max(abs(v_range[0]), abs(v_range[1]))
    if colormap == "colour" or span < 40.0:
        return "TDI"
    return "PW"


def build_spectrogram(
    img: DicomImage,
    region: Region,
    cfg: RenderConfig,
    extra_overlay: np.ndarray | None = None,
) -> Spectrogram:
    """Crop, clean and calibrate the spectral Doppler region.

    extra_overlay: optional (H_img, W_img) mask of pixels known to be overlays
    (e.g. the ECG trace found by `ecg.extract_ecg`).
    """
    px = img.pixels
    rgb_full = px if px.ndim == 3 else np.repeat(px[..., None], 3, axis=2)
    rgb = rgb_full[region.y0:region.y1 + 1, region.x0:region.x1 + 1].copy()
    H, W, _ = rgb.shape

    # Zero-velocity row: the header is authoritative (the drawn line is several
    # pixels thick, and pixel detection can lock onto dense saturated flow).
    baseline_rows, base_colour, presence = find_baseline(
        rgb, region.ref_y0 if region.ref_y0 is not None else H / 2, cfg)
    if region.ref_y0 is not None:
        zero_row = float(region.ref_y0)
    elif len(baseline_rows):
        zero_row = float(baseline_rows.mean())
    else:
        raise ValueError("No ReferencePixelY0 and no drawn baseline: zero velocity unknown")
    info: dict[str, Any] = {}
    if len(baseline_rows) and not (baseline_rows.min() - 5 <= zero_row <= baseline_rows.max() + 5):
        info["warning_baseline_mismatch"] = (zero_row, baseline_rows.tolist())

    # --- overlays --------------------------------------------------------
    overlay = np.zeros((H, W), bool)
    if extra_overlay is not None:
        overlay |= extra_overlay[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    tol = cfg.overlay_colour_tolerance
    if base_colour is not None:
        chromatic_base = int(base_colour.max() - base_colour.min()) > cfg.chroma_threshold
        if chromatic_base:
            # Coloured baseline: the same colour is used for time markers.
            overlay |= _colour_match(rgb, base_colour, tol)
        else:
            overlay[baseline_rows] |= _colour_match(rgb[baseline_rows], base_colour, tol)
        overlay[baseline_rows] = True

    ch = rgb.astype(np.int16)
    chroma = ch.max(axis=2) - ch.min(axis=2)
    nonblack = ch.max(axis=2) > 8
    candidates = nonblack & ~overlay
    frac_colour = float((chroma[candidates] > cfg.chroma_threshold).mean()) if candidates.any() else 0.0
    colormap = "colour" if frac_colour > cfg.colormap_fraction else "grey"
    if colormap == "colour":
        # Grey pixels on a colour-map display are text, scale ticks or cursors.
        overlay |= nonblack & (chroma <= cfg.chroma_threshold // 2) & (ch.max(axis=2) > 60)
        level = (rgb.astype(np.float32) @ LUMA.astype(np.float32)) * COLORMAP_LEVEL_SCALE
    else:
        # Saturated colours on a grey display are overlays (ECG, markers, cursors).
        overlay |= chroma > cfg.chroma_threshold
        level = rgb.astype(np.float32).mean(axis=2)
        overlay |= find_text(level, overlay, cfg)
    level = np.clip(level, 0, 255)

    # --- sweep -------------------------------------------------------------
    sweep = find_sweep(presence, cfg.min_sweep_width_px) if len(baseline_rows) else np.zeros(W, bool)
    order = sweep_order(sweep, cfg.align_to_sweep)

    valid = ~overlay
    valid[:, sweep] = False
    level_filled = fill_invalid_columns(level, valid)
    level_filled[:, sweep] = 0.0

    velocity = (np.arange(H) - zero_row) * region.dy
    modality = classify_modality(region, colormap, (velocity.min(), velocity.max()))

    # Valve clicks: narrow, broadband, saturated vertical lines (the leaflets
    # themselves, not blood flow). Detected as transients of the column-mean
    # level (as the MATLAB AVC detector does). In 'bridge' mode (default) they
    # are interpolated over so that no envelope estimator follows them; their
    # times are reported separately (t_clicks, t_avc). Bridging applies only to
    # cfg.click_bridge_modalities: on TDI, narrow transients are mostly tissue
    # motion (isovolumic contraction/relaxation spikes), which the envelope
    # must keep, so TDI uses 'keep'.
    if cfg.click_mode not in ("bridge", "follow", "keep", "off"):
        raise ValueError(f"unknown click_mode '{cfg.click_mode}'")
    click_mode = cfg.click_mode
    if click_mode in ("bridge", "follow") and modality not in cfg.click_bridge_modalities:
        click_mode = "keep"
    if click_mode == "off":
        clicks, strength = np.zeros(W, bool), np.zeros(W)
    else:
        clicks, strength = find_clicks(level_filled, sweep, region.dx, cfg, return_strength=True)
    if clicks.any():
        info["click_columns"] = np.flatnonzero(clicks).tolist()
        if click_mode == "bridge":
            level_filled = bridge_columns(level_filled, clicks | sweep)
            level_filled[:, sweep] = 0.0
    info.update(colour_fraction=frac_colour, n_sweep_columns=int(sweep.sum()), display_zero_row=zero_row)

    return Spectrogram(
        level=level_filled, valid=valid, zero_row=zero_row, dy=float(region.dy), dt=float(region.dx),
        sweep=sweep, order=order, colormap=colormap, modality=modality, region=region,
        baseline_rows=baseline_rows, rgb=rgb, overlay=overlay, info=info,
        clicks=clicks, click_strength=strength, click_mode=click_mode,
    )


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=r) if r > 0 else mask
