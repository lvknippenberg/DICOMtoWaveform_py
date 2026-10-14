"""dicom2waveform_standalone: ECG and Doppler envelopes from spectral / tissue Doppler DICOMs.

Single-file version of the `dicom2waveform` package (primary method only: noise-referenced
soft membership + dynamic-programming envelope). No benchmark estimators, synthetic data or
evaluation code.

Given a single-frame spectral Doppler (PW, CW) or tissue Doppler (TDI) DICOM, i.e. the
rendered display as stored by the scanner, it returns on one shared time axis:

    t           s, oldest sample first (the display's circular sweep is undone)
    ecg         a.u., traced from the burned-in ECG; isoelectric ~0, max |deflection| = 1
    v_max       cm/s, outer edge of the positive spectrum (>= 0; 0 when no flow on that side)
    v_min       cm/s, outer edge of the negative spectrum (<= 0; 0 when no flow)
    v_mean      cm/s, intensity-weighted mean velocity inside the detected flow region (signed)
    v_mean_pos  cm/s, same, positive half only
    v_mean_neg  cm/s, same, negative half only

plus event times t_rpeaks (R-peaks), t_clicks (valve clicks), t_avc (aortic valve closure)
and quality flags (modality, aliasing, ECG clipping, SNR proxy, ...).

Velocities carry the sign of the DICOM header (PhysicalDeltaY): on a display the operator
inverted, this is opposite to the on-screen label. NaN marks samples without data (the
sweep band of the display).

Usage as a module
-----------------
    from dicom2waveform_standalone import extract_waveforms
    res = extract_waveforms("file.dcm")                         # built-in defaults
    res = extract_waveforms("file.dcm", "dicom2waveform_config.yaml")
    res = extract_waveforms("file.dcm", {"dp": {"edge_density": 0.4}})
    res.t, res.ecg, res.v_max, res.v_min, res.v_mean, res.t_rpeaks, res.flags
    res.to_csv("out.csv"); plot_result(res, "out.png")

Usage from the command line
---------------------------
    python dicom2waveform_standalone.py file.dcm [more.dcm ...] [--config cfg.yaml]
                                        [--out-dir DIR] [--plot]

Every tunable parameter is documented, with its default, in dicom2waveform_config.yaml.
The defaults below are identical, so the YAML file is optional; give only the keys you
want to change.

Requirements: numpy, scipy, pydicom. Optional: pyyaml (YAML config), matplotlib
(plot_result), pylibjpeg / pylibjpeg-libjpeg (JPEG-compressed DICOMs).

Note: DICOMs and full-frame renderings contain burned-in patient identifiers. The QC
figure shows only the spectral region and ECG strip, but treat it as patient data anyway.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from scipy import ndimage, signal

__version__ = "0.1.0-standalone"


# =============================================================================
# Configuration (documentation of every parameter: dicom2waveform_config.yaml)
# =============================================================================

@dataclass
class RenderConfig:
    chroma_threshold: int = 12
    colormap_fraction: float = 0.5
    baseline_search_rows: int = 6
    min_sweep_width_px: int = 3
    align_to_sweep: bool = True
    overlay_colour_tolerance: int = 6
    text_level: float = 240.0
    text_max_area: int = 600
    text_ring_max: float = 70.0
    click_mode: str = "keep"  # 'keep' | 'bridge' | 'follow' | 'off'
    click_bridge_modalities: tuple[str, ...] = ("PW", "CW")
    click_min_snr: float = 12.0
    click_max_width_s: float = 0.035


@dataclass
class ECGConfig:
    search_margin_px: int = 100
    search_above_px: int = 120
    colour_tolerance: int = 6
    min_component_width_frac: float = 0.03
    max_gap_s: float = 0.05
    detect_rpeaks: bool = True
    avc_window_s: tuple[float, float] = (0.20, 0.50)
    avc_edge_guard_s: float = 0.012
    avc_min_snr: float = 6.0
    avc_centroid_s: float = 0.035
    qrs_band_hz: tuple[float, float] = (8.0, 30.0)
    refractory_s: float = 0.25
    energy_smooth_s: float = 0.10
    thresh_fraction: float = 0.35
    noise_floor_fraction: float = 0.15
    searchback_gap_ratio: float = 1.66
    twave_rr_fraction: tuple[float, float] = (0.20, 0.55)
    twave_relative_height: float = 0.5


@dataclass
class LikelihoodConfig:
    noise_margin: float = 32.0
    noise_spread_factor: float = 4.0
    soft_low_factor: float = 1.5
    density_window_cm_s: float = 0.0
    density_window_px_rows: int = 9
    density_window_s: float = 0.0
    density_window_px_cols: int = 5


@dataclass
class DPConfig:
    reference_accel_cm_s2: float = 640.0
    accel_scale_with_range: bool = True
    reference_range_cm_s: float = 100.0
    edge_density: float = 0.5
    edge_density_tdi: float | None = None
    inner_band_rel: float = 0.05
    height_prior: float = 0.01
    wall_band_rel: float = 0.03
    background_percentile: float | None = 50.0
    normalise_percentile: float | None = 99.0
    normalise_floor: float = 0.2
    local_norm_s: float = 0.01
    local_norm_floor: float = 0.15
    local_norm_modalities: tuple[str, ...] = ("TDI",)


@dataclass
class AliasingConfig:
    correct: bool = True
    alias_edge_fraction: float = 0.02
    alias_min_fraction: float = 0.01
    guard_rel: float = 0.10
    cut_smooth_rows: int = 5
    edge_preference: float = 0.002
    cut_lambda: float = 0.05
    max_cost_ratio: float = 0.5
    min_edge_density: float = 0.05


@dataclass
class MeanConfig:
    weighting: str = "power"  # 'power' | 'level' | 'membership'
    dynamic_range_db: float = 40.0
    clutter_band_cm_s: float = 0.0
    min_pixels: int = 2


@dataclass
class PostConfig:
    method: str = "none"  # 'none' | 'median' | 'savgol'
    median_window_s: float = 0.015
    savgol_window_s: float = 0.025
    savgol_order: int = 2


@dataclass
class Config:
    frame: int = 0
    render: RenderConfig = field(default_factory=RenderConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    aliasing: AliasingConfig = field(default_factory=AliasingConfig)
    mean: MeanConfig = field(default_factory=MeanConfig)
    post: PostConfig = field(default_factory=PostConfig)


def _update(obj: Any, values: dict[str, Any]) -> Any:
    for key, val in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown configuration key '{key}' for {type(obj).__name__}")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _update(cur, val)
        else:
            if isinstance(cur, tuple) and isinstance(val, list):
                val = tuple(val)
            setattr(obj, key, val)
    return obj


def load_config(source: str | Path | dict | Config | None = None) -> Config:
    """Default configuration, optionally overridden by a YAML file or a nested dict.

    Only the keys that are given are changed; unknown keys raise KeyError.
    """
    if isinstance(source, Config):
        return source
    cfg = Config()
    if source is None:
        return cfg
    if isinstance(source, dict):
        return _update(cfg, source)
    import yaml  # only needed for YAML files

    with open(source, encoding="utf8") as fh:
        return _update(cfg, yaml.safe_load(fh) or {})


# =============================================================================
# DICOM reading and ultrasound region calibration
# =============================================================================
# The spectral display is located and calibrated from the DICOM Ultrasound Region
# Calibration module (SequenceOfUltrasoundRegions): region box, time and velocity per
# pixel and the zero-velocity row. Region coordinates are 0-based and inclusive.

UNIT_CM, UNIT_SECONDS, UNIT_CM_PER_S = 3, 4, 7


@dataclass
class Region:
    """One item of SequenceOfUltrasoundRegions."""

    x0: int
    y0: int
    x1: int  # inclusive
    y1: int  # inclusive
    dx: float  # physical units per pixel in x
    dy: float  # physical units per pixel in y (downwards)
    units_x: int
    units_y: int
    data_type: int
    ref_y0: int | None = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def is_spectral(self) -> bool:
        return self.units_x == UNIT_SECONDS and self.units_y == UNIT_CM_PER_S

    @property
    def is_bmode(self) -> bool:
        return self.units_x == UNIT_CM and self.units_y == UNIT_CM


@dataclass
class DicomImage:
    path: str
    pixels: np.ndarray  # (H, W) or (H, W, 3) uint8, RGB
    regions: list[Region]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rgb(self) -> bool:
        return self.pixels.ndim == 3


def _get(ds: Any, name: str, default: Any = None) -> Any:
    try:
        val = getattr(ds, name)
    except AttributeError:
        return default
    return default if val is None or val == "" else val


def _to_uint8(ds: Any, arr: np.ndarray) -> np.ndarray:
    photometric = str(_get(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric.startswith("YBR"):
        try:
            from pydicom.pixels import convert_color_space
        except ImportError:  # pydicom < 3
            from pydicom.pixel_data_handlers.util import convert_color_space
        arr = convert_color_space(arr, photometric, "RGB")
    elif photometric == "PALETTE COLOR":
        try:
            from pydicom.pixels import apply_color_lut
        except ImportError:  # pydicom < 3
            from pydicom.pixel_data_handlers.util import apply_color_lut
        arr = apply_color_lut(arr, ds)
        if arr.dtype != np.uint8:
            arr = (arr / 257).astype(np.uint8)
    elif photometric == "MONOCHROME1":
        arr = arr.max() - arr
    if arr.dtype != np.uint8:
        bits = int(_get(ds, "BitsStored", 8))
        arr = np.clip(arr.astype(np.float64) / (2**bits - 1) * 255, 0, 255).astype(np.uint8)
    return arr


META_TAGS = [
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "Modality", "Rows", "Columns",
    "PhotometricInterpretation", "NumberOfFrames", "HeartRate", "TransducerData",
    "SeriesDescription", "SOPInstanceUID",
]


def load_dicom(path: str | Path, frame: int = 0) -> DicomImage:
    """Pixels, ultrasound regions and non-identifying technical metadata."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    n_frames = int(_get(ds, "NumberOfFrames", 1) or 1)
    photometric = str(_get(ds, "PhotometricInterpretation", ""))
    if n_frames > 1:
        arr = arr[frame]
    elif arr.ndim == 3 and photometric == "MONOCHROME2" and arr.shape[-1] != 3:
        arr = arr[frame]
    arr = _to_uint8(ds, arr)
    regions = []
    for r in _get(ds, "SequenceOfUltrasoundRegions", []) or []:
        regions.append(Region(
            x0=int(r.RegionLocationMinX0), y0=int(r.RegionLocationMinY0),
            x1=int(r.RegionLocationMaxX1), y1=int(r.RegionLocationMaxY1),
            dx=float(_get(r, "PhysicalDeltaX", np.nan)), dy=float(_get(r, "PhysicalDeltaY", np.nan)),
            units_x=int(_get(r, "PhysicalUnitsXDirection", 0)),
            units_y=int(_get(r, "PhysicalUnitsYDirection", 0)),
            data_type=int(_get(r, "RegionDataType", 0)),
            ref_y0=None if _get(r, "ReferencePixelY0") is None else int(r.ReferencePixelY0),
        ))
    meta = {}
    for tag in META_TAGS:
        val = _get(ds, tag)
        if val is not None:
            meta[tag] = val if isinstance(val, (int, float, str)) else str(val)
    return DicomImage(path=str(path), pixels=arr, regions=regions, meta=meta)


def spectral_region(img: DicomImage) -> Region:
    """The spectral Doppler region (s horizontally, cm/s vertically), by unit codes."""
    cands = sorted((r for r in img.regions if r.is_spectral), key=lambda r: r.width * r.height, reverse=True)
    if not cands:
        raise ValueError("No spectral Doppler region (s vs cm/s) in SequenceOfUltrasoundRegions")
    r = cands[0]
    if not np.isfinite(r.dy) or r.dy == 0:
        raise ValueError(f"PhysicalDeltaY must be finite and non-zero, got {r.dy}")
    if not np.isfinite(r.dx) or r.dx <= 0:
        raise ValueError(f"PhysicalDeltaX must be finite and positive, got {r.dx}")
    return r


def bmode_region_above(img: DicomImage, spec: Region) -> Region | None:
    """The B-mode region directly above the spectral region (the ECG runs along its lower edge)."""
    cands = [r for r in img.regions if r.is_bmode and r.y1 <= spec.y0]
    return max(cands, key=lambda r: r.y1) if cands else None


# =============================================================================
# Rendered display -> calibrated scalar image (overlays, sweep, clicks)
# =============================================================================

# Brightest colour of the Philips copper TDI map has luminance ~150; scale so colour-map
# displays use roughly the same 0..255 range as grey displays.
COLORMAP_LEVEL_SCALE = 255.0 / 160.0
LUMA = np.array([0.299, 0.587, 0.114])


@dataclass
class Spectrogram:
    """Calibrated spectral display. Row r has velocity (r - zero_row) * dy (cm/s); arrays are in
    raw display column order, `order` is the permutation to oldest -> newest."""

    level: np.ndarray  # (H, W) display level 0..255, overlays filled
    valid: np.ndarray  # (H, W) False on overlays and in the sweep
    zero_row: float
    dy: float
    dt: float
    sweep: np.ndarray  # (W,) column holds no data
    order: np.ndarray
    colormap: str  # 'grey' | 'colour'
    modality: str  # 'PW' | 'CW' | 'TDI'
    region: Region
    baseline_rows: np.ndarray
    rgb: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    clicks: np.ndarray | None = None
    click_strength: np.ndarray | None = None
    click_mode: str = "keep"

    @property
    def display_height(self) -> int:
        """Rows of the displayed region (unchanged by aliasing unwrapping)."""
        return int(self.region.height)

    @property
    def display_v_limits(self) -> tuple[float, float]:
        """Velocity limits of the display as rendered (the Nyquist window for PW)."""
        z0 = self.info.get("display_zero_row", self.zero_row)
        v = (np.array([0, self.display_height - 1]) - z0) * self.dy
        return float(v.min()), float(v.max())


def _colour_match(rgb: np.ndarray, colour: np.ndarray, tol: int) -> np.ndarray:
    return np.all(np.abs(rgb.astype(np.int16) - np.asarray(colour, np.int16)) <= tol, axis=-1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of runs of True in a 1-D mask."""
    d = np.diff(np.concatenate([[False], mask.astype(bool), [False]]).astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _dominant_colour(pixels: np.ndarray) -> tuple[np.ndarray | None, int]:
    px = pixels.reshape(-1, 3)
    px = px[px.sum(1) > 0]
    if len(px) == 0:
        return None, 0
    u, c = np.unique(px, axis=0, return_counts=True)
    i = int(np.argmax(c))
    return u[i].astype(int), int(c[i])


def find_baseline(rgb: np.ndarray, zero_row: float, cfg: RenderConfig):
    """Drawn zero-velocity line near zero_row: (rows, colour, per-column presence)."""
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
    rows = np.array([r for r in range(lo, hi + 1)
                     if _colour_match(rgb[r], best, cfg.overlay_colour_tolerance).mean() > 0.3], int)
    if len(rows) == 0:
        return rows, None, np.ones(W, bool)
    blocks = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    rows = min(blocks, key=lambda b: abs(b.mean() - zero_row))
    presence = np.any(_colour_match(rgb[rows], best, cfg.overlay_colour_tolerance), axis=0)
    return rows, best, presence


def find_text(level: np.ndarray, overlay: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    """White annotations: small bright components with a dark surround (saturated flow and
    valve clicks have a bright surround)."""
    bright = (level >= cfg.text_level) & ~overlay
    if not bright.any():
        return np.zeros_like(bright)
    lab, n = ndimage.label(bright, structure=np.ones((3, 3)))
    area = ndimage.sum(bright, lab, index=np.arange(1, n + 1))
    ring = ndimage.binary_dilation(bright, iterations=3) & ~bright & ~overlay
    near = ndimage.grey_dilation(lab, size=(7, 7))
    with np.errstate(invalid="ignore", divide="ignore"):
        ring_mean = ndimage.mean(level, labels=np.where(ring, near, 0), index=np.arange(1, n + 1))
    ring_mean = np.nan_to_num(np.asarray(ring_mean, float), nan=0.0)
    is_text = (area <= cfg.text_max_area) & (ring_mean <= cfg.text_ring_max)
    return ndimage.binary_dilation(np.isin(lab, np.flatnonzero(is_text) + 1), iterations=1)


def find_clicks(level: np.ndarray, sweep: np.ndarray, dt: float, cfg: RenderConfig):
    """Valve-click columns (short broadband transients of the column-mean level) and the
    transient height in robust SDs for every column."""
    W = level.shape[1]
    E = level.mean(axis=0)
    E[sweep] = np.nan
    m = max(1, int(round(cfg.click_max_width_s / dt)))
    x = np.arange(W)
    good = ~np.isnan(E)
    if good.sum() < 2 * m:
        return np.zeros(W, bool), np.zeros(W)
    Ef = np.interp(x, x[good], E[good])
    # white top-hat: removes features narrower than a click, keeps steps (flow on/offset)
    hp = Ef - ndimage.grey_opening(Ef, size=m + 2, mode="nearest")
    hp[sweep] = 0.0
    edge = np.zeros(W, bool)
    for a, b in _runs(good):
        edge[a:min(a + m, b + 1)] = True
        edge[max(a, b - m + 1):b + 1] = True
    hp[edge] = 0.0
    sig = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6
    clicks = np.zeros(W, bool)
    for a, b in _runs(hp > cfg.click_min_snr * sig):
        if b - a + 1 <= m:
            clicks[max(0, a - 1):b + 2] = True
    return clicks, hp / sig


def _bridge_columns(level: np.ndarray, bad: np.ndarray) -> np.ndarray:
    out = level.copy()
    if (~bad).sum() < 2:
        return out
    for a, b in _runs(bad):
        left = a - 1 if a > 0 else None
        right = b + 1 if b < level.shape[1] - 1 else None
        if left is None or right is None or bad[left] or bad[right]:
            continue
        w = (np.arange(a, b + 1) - left) / (right - left)
        out[:, a:b + 1] = level[:, [left]] * (1 - w) + level[:, [right]] * w
    return out


def _sweep_order(sweep: np.ndarray, align: bool) -> np.ndarray:
    """Column permutation oldest -> newest: the circular buffer starts right of the sweep."""
    W = len(sweep)
    if not align or not sweep.any():
        return np.arange(W)
    runs = _runs(sweep)
    widths = [e - s + 1 for s, e in runs]
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == W - 1:
        widths[-1] += widths[0]
        widths[0] = 0
    return np.roll(np.arange(W), -runs[int(np.argmax(widths))][0])


def _fill_invalid(level: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid pixels by linear interpolation along each column."""
    out = level.astype(np.float32).copy()
    rows = np.arange(level.shape[0])
    for c in np.flatnonzero((~valid).any(axis=0)):
        v = valid[:, c]
        if not v.any():
            out[:, c] = 0.0
            continue
        out[~v, c] = np.interp(rows[~v], rows[v], out[v, c])
    return out


def level_to_power(level, dynamic_range_db: float) -> np.ndarray:
    """Display level -> approximate relative linear power (grey levels 0..255 span dynamic_range_db)."""
    return np.power(10.0, (np.asarray(level, np.float64) / 255.0) * dynamic_range_db / 10.0) - 1.0


def build_spectrogram(img: DicomImage, region: Region, cfg: RenderConfig,
                      extra_overlay: np.ndarray | None = None) -> Spectrogram:
    """Crop, clean (overlays masked and filled) and calibrate the spectral region."""
    px = img.pixels
    rgb_full = px if px.ndim == 3 else np.repeat(px[..., None], 3, axis=2)
    rgb = rgb_full[region.y0:region.y1 + 1, region.x0:region.x1 + 1].copy()
    H, W, _ = rgb.shape

    # zero-velocity row: the header is authoritative
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

    # overlays
    overlay = np.zeros((H, W), bool)
    if extra_overlay is not None:
        overlay |= extra_overlay[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    tol = cfg.overlay_colour_tolerance
    if base_colour is not None:
        if int(base_colour.max() - base_colour.min()) > cfg.chroma_threshold:
            overlay |= _colour_match(rgb, base_colour, tol)  # coloured baseline = time markers too
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
        # grey pixels on a colour-map (TDI) display are text, ticks or cursors
        overlay |= nonblack & (chroma <= cfg.chroma_threshold // 2) & (ch.max(axis=2) > 60)
        level = (rgb.astype(np.float32) @ LUMA.astype(np.float32)) * COLORMAP_LEVEL_SCALE
    else:
        # saturated colours on a grey display are overlays (ECG, markers, cursors)
        overlay |= chroma > cfg.chroma_threshold
        level = rgb.astype(np.float32).mean(axis=2)
        overlay |= find_text(level, overlay, cfg)
    level = np.clip(level, 0, 255)

    # sweep: columns without a drawn baseline
    sweep = np.zeros(W, bool)
    if len(baseline_rows) and not presence.all():
        for s, e in _runs(~presence):
            if e - s + 1 >= cfg.min_sweep_width_px:
                sweep[s:e + 1] = True
    order = _sweep_order(sweep, cfg.align_to_sweep)

    valid = ~overlay
    valid[:, sweep] = False
    level_filled = _fill_invalid(level, valid)
    level_filled[:, sweep] = 0.0

    velocity = (np.arange(H) - zero_row) * region.dy
    span = max(abs(velocity.min()), abs(velocity.max()))
    modality = "CW" if region.data_type == 4 else ("TDI" if colormap == "colour" or span < 40.0 else "PW")

    if cfg.click_mode not in ("bridge", "follow", "keep", "off"):
        raise ValueError(f"unknown click_mode '{cfg.click_mode}'")
    click_mode = cfg.click_mode
    if click_mode in ("bridge", "follow") and modality not in cfg.click_bridge_modalities:
        click_mode = "keep"
    if click_mode == "off":
        clicks, strength = np.zeros(W, bool), np.zeros(W)
    else:
        clicks, strength = find_clicks(level_filled, sweep, region.dx, cfg)
    if clicks.any() and click_mode == "bridge":
        level_filled = _bridge_columns(level_filled, clicks | sweep)
        level_filled[:, sweep] = 0.0
    info.update(colour_fraction=frac_colour, n_sweep_columns=int(sweep.sum()), display_zero_row=zero_row)

    return Spectrogram(level=level_filled, valid=valid, zero_row=zero_row, dy=float(region.dy),
                       dt=float(region.dx), sweep=sweep, order=order, colormap=colormap, modality=modality,
                       region=region, baseline_rows=baseline_rows, rgb=rgb, info=info, clicks=clicks,
                       click_strength=strength, click_mode=click_mode)


# =============================================================================
# ECG trace extraction and R-peak detection
# =============================================================================
# The ECG is only burned into the pixels, along the lower edge of the B-mode region,
# sharing the columns (time axis) of the spectral display. No amplitude calibration.

@dataclass
class ECGTrace:
    row: np.ndarray  # (W,) row of the trace in full-image coordinates, NaN = none
    mask: np.ndarray  # (H_img, W_img) pixels of the drawn trace
    found: bool
    clipped: bool = False


def _find_trace_colour(band: np.ndarray, is_grey: bool, centre_row: int, exclude: tuple = ()):
    """Colour spanning most columns without being a horizontal line."""
    ch = band.astype(np.int16)
    chroma = ch.max(axis=2) - ch.min(axis=2)
    if is_grey:
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
        rows = np.flatnonzero(m.sum(axis=1) > 0)
        line_like = len(rows) == 0 or (rows.max() - rows.min()) < 6
        if cov > best_cov and not line_like:
            best, best_cov = colour, cov
    if best is None or best_cov < 0.2:
        return None
    return best.astype(int)


def _column_samples(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """One row per column: stroke top/bottom at local extremes, centre elsewhere."""
    h, W = mask.shape
    runs_per_col = [_runs(mask[:, c]) for c in range(W)]
    centre = np.full(W, np.nan)
    for c, runs in enumerate(runs_per_col):
        if len(runs) == 1:
            centre[c] = 0.5 * (runs[0][0] + runs[0][1])
    known = np.flatnonzero(~np.isnan(centre))
    for c, runs in enumerate(runs_per_col):
        if len(runs) > 1:  # several strokes: take the one continuing the trace
            ref = np.interp(c, known, centre[known]) if len(known) else h / 2
            s, e = min(runs, key=lambda r: min(abs(r[0] - ref), abs(r[1] - ref)))
            centre[c] = 0.5 * (s + e)
            runs_per_col[c] = [(s, e)]
    y = centre.copy()
    for c in range(W):
        if not runs_per_col[c]:
            continue
        s, e = runs_per_col[c][0]
        nb = np.array([centre[c - 1] if c > 0 else np.nan, centre[c + 1] if c < W - 1 else np.nan])
        nb = nb[~np.isnan(nb)]
        if len(nb) == 0 or e - s < 2:
            continue
        if s < nb.min() - 1:
            y[c] = s
        elif e > nb.max() + 1:
            y[c] = e
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
    top_ref = bmode.y1 if bmode is not None else spec.y0
    lo = max(0, top_ref - cfg.search_above_px)
    hi = min(px.shape[0] - 1, spec.y0 + cfg.search_margin_px)
    band = rgb[lo:hi + 1, spec.x0:spec.x1 + 1]
    W = band.shape[1]
    centre_row = (bmode.y1 - 1 - lo) if bmode is not None else (spec.y0 - lo)
    colour = _find_trace_colour(band, is_grey, centre_row, exclude_colours)
    full_mask = np.zeros(px.shape[:2], bool)
    if colour is None:
        return ECGTrace(row=np.full(W, np.nan), mask=full_mask, found=False)

    m = _colour_match(band, colour, cfg.colour_tolerance)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n:
        objs = ndimage.find_objects(lab)
        widths = np.array([sl[1].stop - sl[1].start for sl in objs])
        keep = widths >= cfg.min_component_width_frac * W
        if not keep.any():
            keep = widths == widths.max()
        if not is_grey:  # unique trace colour: keep small pieces within the main trace's band
            rows_main = np.concatenate([np.arange(objs[i][0].start, objs[i][0].stop) for i in np.flatnonzero(keep)])
            r0, r1 = rows_main.min(), rows_main.max()
            for i, sl in enumerate(objs):
                if not keep[i] and sl[0].start >= r0 - 2 and sl[0].stop <= r1 + 3:
                    keep[i] = True
        m = np.isin(lab, np.flatnonzero(keep) + 1)
    y = _column_samples(m, max(1, int(round(cfg.max_gap_s / spec.dx))))
    full_mask[lo:hi + 1, spec.x0:spec.x1 + 1] = m
    found = bool(np.mean(~np.isnan(y)) > 0.2)
    clipped = False
    if found:  # flat runs at the extreme row = R/S waves clipped to the scanner's ECG box
        yr = y[~np.isnan(y)]
        clipped = bool(max(np.sum(yr <= yr.min() + 0.5), np.sum(yr >= yr.max() - 0.5)) > 0.02 * len(yr))
    return ECGTrace(row=y + lo, mask=full_mask, found=found, clipped=clipped)


def _robust_std(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) + np.finfo(float).eps


def _find_peaks(E: np.ndarray, threshold: float, min_dist: int):
    if len(E) == 0 or E.max() <= threshold:
        return np.array([]), np.array([], int)
    idx, props = signal.find_peaks(E, height=threshold, distance=max(1, min_dist))
    return props["peak_heights"], idx


def detect_rpeaks(ecg: np.ndarray, t: np.ndarray, cfg: ECGConfig, heart_rate_hint: float | None = None) -> np.ndarray:
    """R-peak times (s) from QRS-band energy (Pan-Tompkins style), with search-back and an
    RR-relative T-wave guard. heart_rate_hint (bpm, DICOM HeartRate) sets the expected RR."""
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
        xb = signal.sosfiltfilt(signal.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos"), x)
    except ValueError:
        xb = x - ndimage.uniform_filter1d(x, max(3, int(round(fs / lo))), mode="nearest")
    E = (xb / _robust_std(xb)) ** 2
    E = ndimage.uniform_filter1d(E, max(1, int(round(cfg.energy_smooth_s * fs))), mode="nearest")
    E = E / (E.max() + np.finfo(float).eps)

    min_dist = int(round(cfg.refractory_s * fs))
    heights, locs = _find_peaks(E, max(np.median(E) * 2, 0.005), min_dist)
    if len(locs) == 0:
        return np.array([])
    qrs_level = np.median(heights[heights >= np.median(heights)])  # upper-half median
    noise_floor = cfg.noise_floor_fraction * qrs_level
    keep = heights >= max(cfg.thresh_fraction * qrs_level, noise_floor)
    heights, locs = heights[keep], locs[keep]

    # search-back in abnormally long RR gaps
    if len(locs) >= 2 or (heart_rate_hint and len(locs) >= 1):
        rr_ref = np.median(np.diff(locs)) if len(locs) >= 2 else np.inf
        if heart_rate_hint and heart_rate_hint > 0:
            rr_ref = min(rr_ref, 60.0 / heart_rate_hint * fs)
        gap_thr = cfg.searchback_gap_ratio * rr_ref
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
                    plaus = np.zeros(len(l), bool)  # prefer a plausible RR from a neighbour
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
        f0, f1 = cfg.twave_rr_fraction
        for k in range(1, len(locs)):
            rr = np.median(np.diff(locs[keep]))
            if not np.isfinite(rr):
                break
            d = locs[k] - locs[last]
            if (f0 * rr < d < f1 * rr and heights[k] < cfg.twave_relative_height * heights[last]
                    and heights[k] < cfg.twave_relative_height * med_h):
                keep[k] = False
            else:
                last = k
        locs = locs[keep]

    locs = locs[~nodata[locs]]
    # refine to the largest |ECG| deflection within +-40 ms
    w = max(1, int(round(0.04 * fs)))
    xc = x - np.median(x)
    refined = [max(0, l - w) + int(np.argmax(np.abs(xc[max(0, l - w):min(len(x), l + w + 1)]))) for l in locs]
    return ts[np.array(refined, int)] if refined else np.array([])


# =============================================================================
# Envelope: soft membership -> local density -> dynamic-programming edge per side
# =============================================================================

@dataclass
class NoiseStats:
    level: float  # noise level (median of the image)
    spread: float  # robust SD of the noise
    otsu: float
    threshold: float  # membership reaches 1 here
    noise_top: float  # membership starts rising here


def _otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    hist, edges = np.histogram(values, bins=256, range=(0.0, 255.0))
    p = hist.astype(float) / max(hist.sum(), 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = 0
    return float(edges[int(np.argmax(sb)) + 1])


def noise_stats(spec: Spectrogram, cfg: LikelihoodConfig) -> NoiseStats:
    """Signal threshold = min(noise + margin, Otsu); membership ramps up from just above the noise."""
    vals = spec.level[spec.valid]
    if len(vals) == 0:
        return NoiseStats(0.0, 1.0, 128.0, 128.0, 0.0)
    level = float(np.median(vals))
    spread = float(1.4826 * np.median(np.abs(vals - level)))
    otsu = _otsu_threshold(vals)
    threshold = min(level + max(cfg.noise_margin, cfg.noise_spread_factor * spread), otsu)
    return NoiseStats(level, spread, otsu, threshold, level + cfg.soft_low_factor * max(spread, 1.0))


def membership(level: np.ndarray, ns: NoiseStats) -> np.ndarray:
    """Soft signal membership in [0, 1]."""
    return np.clip((level - ns.noise_top) / max(ns.threshold - ns.noise_top, 1.0), 0.0, 1.0)


@dataclass
class SideImage:
    """One half of the spectrogram, re-indexed outward from the baseline (k = 0 at zero velocity)."""

    level: np.ndarray  # (D, W)
    rows: np.ndarray  # (D,) original row of each k
    speed_per_row: float
    sign: int

    @property
    def depth(self) -> int:
        return self.level.shape[0]


def side_view(spec: Spectrogram, side: str, image: np.ndarray | None = None) -> SideImage:
    img = spec.level if image is None else image
    H = img.shape[0]
    z = int(round(spec.zero_row))
    up_is_positive = spec.dy < 0
    rows = np.arange(z, -1, -1) if (side == "pos") == up_is_positive else np.arange(z, H)
    rows = rows[(rows >= 0) & (rows < H)]
    return SideImage(level=img[rows], rows=rows, speed_per_row=abs(spec.dy), sign=1 if side == "pos" else -1)


def _l1_min_convolution(f: np.ndarray, lam: float):
    """h(i) = min_k f(k) + lam*|i-k| and the minimising k, in O(n)."""
    n = len(f)
    idx = np.arange(n)
    a = f - lam * idx
    cm = np.minimum.accumulate(a)
    arg_f = np.maximum.accumulate(np.where(a <= cm, idx, 0))
    h_f = cm + lam * idx
    b = (f + lam * idx)[::-1]
    cmb = np.minimum.accumulate(b)
    pos = np.maximum.accumulate(np.where(b <= cmb, np.arange(n), 0))
    arg_b = idx[::-1][pos][::-1]
    h_b = cmb[::-1] - lam * idx
    use_b = h_b < h_f
    return np.where(use_b, h_b, h_f), np.where(use_b, arg_b, arg_f)


def dp_solve(cost: np.ndarray, lam: float, free: np.ndarray | None = None) -> np.ndarray:
    """Exact minimiser of sum_x cost[e_x, x] + lam*|e_x - e_{x-1}|. Transitions into columns
    where free[x] are unpenalised (decouples the path across the sweep)."""
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


def edge_cost(density: np.ndarray, theta: float, inner_rows: int, height_prior: float, wall_rows: int,
              wall_weight: float = 0.01) -> np.ndarray:
    """Outer-edge cost per (row k, column):

        sum_{j > k} max(d_j - theta, 0)                 signal left outside the edge
      + sum_{j in (k-h, k]} w_j max(theta - d_j, 0)     no signal just inside the edge
      + height_prior * k                                weak prior toward no flow

    The interior deeper than h rows is free (hollow PW / TDI spectra); rows in the wall band
    are nearly free to include."""
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
    return beyond + (sc - prev) + height_prior * np.arange(D)[:, None]


def _column_margin(cost: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Per-column confidence 0..1: how much better the chosen edge is than average."""
    best = cost[path, np.arange(cost.shape[1])]
    spread = cost.max(axis=0) - cost.min(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (cost.mean(axis=0) - best) / spread
    m[~np.isfinite(m)] = 0.0
    return np.clip(m, 0.0, 1.0)


def signal_density(spec: Spectrogram, ns: NoiseStats, lcfg: LikelihoodConfig, dcfg: DPConfig) -> np.ndarray:
    """Local density of soft membership, background-subtracted and normalised to the
    typical in-flow density."""
    m = membership(spec.level, ns)
    m[:, spec.sweep] = 0.0
    rows = lcfg.density_window_px_rows
    if lcfg.density_window_cm_s > 0:
        rows = max(1, int(round(lcfg.density_window_cm_s / abs(spec.dy))))
    cols = lcfg.density_window_px_cols
    if lcfg.density_window_s > 0:
        cols = max(1, int(round(lcfg.density_window_s / spec.dt)))
    m = ndimage.uniform_filter(m.astype(float), size=(rows, cols), mode="nearest")
    valid = spec.valid & ~spec.sweep[None, :]
    q = dcfg.background_percentile
    if q is not None and q > 0:
        bg = float(np.percentile(m[valid], q)) if valid.any() else 0.0
        if bg < 0.9:
            m = np.clip((m - bg) / (1.0 - bg), 0.0, 1.0)
    qn = dcfg.normalise_percentile
    if qn is not None and qn > 0:
        top = float(np.percentile(m[valid], qn)) if valid.any() else 1.0
        m = np.clip(m / max(top, dcfg.normalise_floor), 0.0, 1.0)
    return m


def dp_envelope(spec: Spectrogram, ns: NoiseStats, lcfg: LikelihoodConfig, dcfg: DPConfig) -> dict[str, Any]:
    """Outer edges per side, in rows from the baseline (NaN in the sweep), with confidence."""
    L = signal_density(spec, ns, lcfg, dcfg)
    a_ref = dcfg.reference_accel_cm_s2
    if dcfg.accel_scale_with_range:
        lo, hi = spec.display_v_limits
        a_ref = a_ref * max(abs(lo), abs(hi)) / dcfg.reference_range_cm_s
    lam = (abs(spec.dy) / spec.dt) / a_ref  # physiological acceleration prior, in rows
    free = spec.sweep.copy()
    free[1:] |= spec.sweep[:-1]
    if spec.click_mode == "follow" and spec.clicks is not None and spec.clicks.any():
        free |= spec.clicks
        free[1:] |= spec.clicks[:-1]
    theta = dcfg.edge_density
    if spec.modality == "TDI" and dcfg.edge_density_tdi is not None:
        theta = dcfg.edge_density_tdi
    H = spec.display_height
    wall = int(round(dcfg.wall_band_rel * H))
    out: dict[str, Any] = {"density": L}
    for side in ("pos", "neg"):
        dens = side_view(spec, side, image=L).level
        if dcfg.local_norm_s > 0 and spec.modality in dcfg.local_norm_modalities:
            colmax = dens.max(axis=0)
            colmax[spec.sweep] = 0.0
            n = max(1, int(round(dcfg.local_norm_s / spec.dt)))
            runmax = ndimage.maximum_filter1d(colmax, size=2 * n + 1, mode="nearest")
            dens = np.clip(dens / np.maximum(runmax, dcfg.local_norm_floor)[None, :], 0.0, 1.0)
        cost = edge_cost(dens, theta, max(1, int(round(dcfg.inner_band_rel * H))), dcfg.height_prior, wall)
        cost[:, spec.sweep] = 0.0
        path = dp_solve(cost, lam, free).astype(float)
        path[path <= wall] = 0.0  # edge inside the wall band = no detectable flow
        conf = _column_margin(cost, path.astype(int))
        path[spec.sweep] = np.nan
        conf[spec.sweep] = np.nan
        out[f"k_{side}"], out[f"conf_{side}"] = path, conf
    return out


# =============================================================================
# Aliasing (PW / TDI): wrap-around test and automatic per-column baseline shift
# =============================================================================

def aliasing_suspected(spec: Spectrogram, k_pos: np.ndarray, k_neg: np.ndarray, cfg: AliasingConfig) -> bool:
    """Wrapped flow leaves one display limit and re-enters at the other: both outer edges at
    their limits in the same columns."""
    if spec.modality == "CW":
        return False
    data = ~spec.sweep
    if not data.any():
        return False
    touch = []
    for side, kk in (("pos", k_pos), ("neg", k_neg)):
        D = side_view(spec, side).depth
        touch.append(np.nan_to_num(kk, nan=0) >= (1 - cfg.alias_edge_fraction) * (D - 1))
    return bool((touch[0] & touch[1])[data].mean() > cfg.alias_min_fraction)


def unwrap_aliasing(spec: Spectrogram, density: np.ndarray, cfg: AliasingConfig):
    """Treat the display as circular, find the least-signal cut path away from the baseline (DP),
    and move rows beyond the cut by one display span. Returns (spectrogram, info)."""
    H, W = spec.level.shape
    z = spec.zero_row
    zi = int(round(z))
    guard = max(1, int(round(cfg.guard_rel * H)))
    bottom = np.arange(min(H - 1, zi + guard), H)
    top = np.arange(0, max(0, zi - guard) + 1)
    rows = np.concatenate([bottom, top])  # one coordinate through the display edge H-1 -> 0
    i_edge = len(bottom) - 1
    d = ndimage.uniform_filter1d(density, max(1, cfg.cut_smooth_rows), axis=0, mode="wrap")
    cost = d[rows].copy()
    u = np.arange(len(rows))
    cost += cfg.edge_preference * np.minimum(np.abs(u - i_edge), np.abs(u - (i_edge + 1)))[:, None]
    cost[:, spec.sweep] = 0.0
    free = spec.sweep.copy()
    free[1:] |= spec.sweep[:-1]
    cut = rows[dp_solve(cost, cfg.cut_lambda, free)]
    data = ~spec.sweep
    path_cost = float(d[cut, np.arange(W)][data].mean()) if data.any() else 0.0
    edge_cost_ = float(0.5 * (d[0] + d[H - 1])[data].mean()) if data.any() else 0.0

    n_up = np.where(cut > zi, H - 1 - cut, 0)
    n_down = np.where(cut < zi, cut, 0)
    n_up[spec.sweep] = 0
    n_down[spec.sweep] = 0
    moved = (n_up > 0) | (n_down > 0)
    info = {"corrected": False, "edge_density": edge_cost_, "cut_density": path_cost}
    if not moved.any() or not (path_cost < cfg.max_cost_ratio * edge_cost_ and edge_cost_ > cfg.min_edge_density):
        return spec, info

    T, B = int(n_up.max()), int(n_down.max())
    Hn = H + T + B
    bg = float(np.median(spec.level[spec.valid])) if spec.valid.any() else 0.0

    def remap(arr: np.ndarray, fill) -> np.ndarray:
        out = np.full((Hn,) + arr.shape[1:], fill, dtype=arr.dtype)
        out[T:T + H] = arr
        for x in np.flatnonzero(n_up > 0):
            k = n_up[x]
            out[T - k:T, x] = arr[H - k:H, x]
            out[T + H - k:T + H, x] = fill
        for x in np.flatnonzero(n_down > 0):
            k = n_down[x]
            out[T + H:T + H + k, x] = arr[0:k, x]
            out[T:T + k, x] = fill
        return out

    new = copy.copy(spec)
    new.level = remap(spec.level.astype(np.float32), np.float32(bg))
    new.valid = remap(spec.valid, True)
    new.valid[:, spec.sweep] = False
    if spec.rgb is not None:
        new.rgb = remap(spec.rgb, np.uint8(0))
    new.zero_row = z + T
    new.baseline_rows = spec.baseline_rows + T
    new.info = dict(spec.info)
    info.update(corrected=True, columns=int(moved.sum()), rows_up=T, rows_down=B)
    return new, info


# =============================================================================
# Mean envelopes, smoothing, quality
# =============================================================================

def mean_envelopes(spec: Spectrogram, k_pos: np.ndarray, k_neg: np.ndarray, ns: NoiseStats,
                   cfg: MeanConfig) -> dict[str, np.ndarray]:
    """Weighted mean velocity inside the flow region (baseline .. outer edge) per side and overall."""
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
    k_clutter = cfg.clutter_band_cm_s / abs(spec.dy)
    sums = {}
    for side, kk in (("pos", k_pos), ("neg", k_neg)):
        sv = side_view(spec, side, image=w)
        k = np.arange(sv.depth)[:, None]
        edge = np.nan_to_num(kk, nan=-1.0)[None, :]
        inside = (k <= edge) & (k >= max(k_clutter, 0.0)) & (edge > 0) & (k >= 1)
        ww = np.where(inside, sv.level, 0.0)
        v = sv.sign * np.arange(sv.depth) * sv.speed_per_row
        sums[side] = (ww.sum(axis=0), (ww * v[:, None]).sum(axis=0), inside.sum(axis=0))
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for side in ("pos", "neg"):
            sw, swv, n = sums[side]
            out[f"v_mean_{side}"] = np.where((sw > 0) & (n >= cfg.min_pixels), swv / sw, 0.0)
        sw = sums["pos"][0] + sums["neg"][0]
        out["v_mean"] = np.where(sw > 0, (sums["pos"][1] + sums["neg"][1]) / sw, 0.0)
    for key in out:
        out[key] = out[key].astype(float)
        out[key][spec.sweep] = np.nan
    return out


def smooth_series(v: np.ndarray, dt: float, cfg: PostConfig) -> np.ndarray:
    """Optional smoothing of a velocity series (time order), never across NaN gaps."""
    if cfg.method == "none":
        return v
    if cfg.method not in ("median", "savgol"):
        raise ValueError(f"unknown post-processing '{cfg.method}'")
    out = v.copy()
    for a, b in _runs(~np.isnan(v)):
        seg = v[a:b + 1]
        if cfg.method == "median":
            out[a:b + 1] = ndimage.median_filter(seg, size=max(1, int(round(cfg.median_window_s / dt))) | 1,
                                                 mode="nearest")
        else:
            n = max(cfg.savgol_order + 2, int(round(cfg.savgol_window_s / dt)) | 1)
            if len(seg) > n:
                out[a:b + 1] = signal.savgol_filter(seg, n, cfg.savgol_order, mode="interp")
    return out


def _flow_mask(spec: Spectrogram, k_pos: np.ndarray, k_neg: np.ndarray) -> np.ndarray:
    M = np.zeros(spec.level.shape, bool)
    for side, kk in (("pos", k_pos), ("neg", k_neg)):
        sv = side_view(spec, side)
        edge = np.nan_to_num(kk, nan=-1.0)
        M[sv.rows] |= (np.arange(sv.depth)[:, None] <= edge[None, :]) & (edge[None, :] > 0)
    M[:, spec.sweep] = False
    return M


def snr_proxy_db(spec: Spectrogram, flow: np.ndarray, dynamic_range_db: float) -> float:
    """Image-domain SNR proxy: mean de-compressed power inside / outside the flow region."""
    valid = spec.valid & ~spec.sweep[None, :]
    P = level_to_power(spec.level, dynamic_range_db)
    sig, bg = P[flow & valid], P[~flow & valid]
    if len(sig) < 10 or len(bg) < 10:
        return float("nan")
    return float(10 * np.log10(max(sig.mean(), 1e-12) / max(bg.mean(), 1e-12)))


# =============================================================================
# Valve clicks and aortic valve closure
# =============================================================================

def click_events(spec: Spectrogram, t: np.ndarray) -> np.ndarray:
    """Valve-click times (s): half-maximum centroid of each run of click columns."""
    if spec.clicks is None or not spec.clicks.any():
        return np.array([])
    o = spec.order
    c = spec.clicks[o]
    s = np.where(c, spec.click_strength[o], 0.0)
    times = []
    for a, b in _runs(c):
        seg = s[a:b + 1]
        w = np.clip(seg - 0.5 * seg.max(), 0.0, None)
        times.append(float(np.sum(w * t[a:b + 1]) / np.sum(w)) if w.sum() > 0 else float(0.5 * (t[a] + t[b])))
    return np.array(times)


def aortic_valve_closure(spec: Spectrogram, t: np.ndarray, t_rpeaks: np.ndarray, cfg: ECGConfig) -> np.ndarray:
    """Per beat, the strongest click transient within cfg.avc_window_s after the R-peak (at least
    cfg.avc_min_snr robust SDs, not at the window edges), timed at its half-maximum centroid."""
    if spec.click_strength is None or len(t_rpeaks) == 0:
        return np.array([])
    s = np.nan_to_num(spec.click_strength[spec.order])
    lo, hi = cfg.avc_window_s
    g = cfg.avc_edge_guard_s
    half = max(1, int(round(0.5 * cfg.avc_centroid_s / (t[1] - t[0]))))
    out = []
    for r in t_rpeaks:
        win = np.flatnonzero((t >= r + lo) & (t <= r + hi))
        if len(win) == 0:
            continue
        i = win[int(np.argmax(s[win]))]
        if s[i] < cfg.avc_min_snr or not (r + lo + g <= t[i] <= r + hi - g):
            continue
        a, b = max(0, i - half), min(len(s), i + half + 1)
        w = np.clip(s[a:b] - 0.5 * s[i], 0.0, None)
        out.append(float(np.sum(w * t[a:b]) / np.sum(w)) if w.sum() > 0 else float(t[i]))
    return np.array(out)


# =============================================================================
# Pipeline
# =============================================================================

@dataclass
class WaveformResult:
    """All signals share the time axis `t` (s), oldest sample first; NaN = no data (sweep)."""

    t: np.ndarray
    ecg: np.ndarray  # a.u., isoelectric ~0, max |deflection| = 1
    v_mean: np.ndarray
    v_mean_pos: np.ndarray
    v_mean_neg: np.ndarray
    v_max: np.ndarray  # outer edge of the positive spectrum (>= 0)
    v_min: np.ndarray  # outer edge of the negative spectrum (<= 0)
    conf_pos: np.ndarray  # per-column confidence of v_max (0..1)
    conf_neg: np.ndarray  # per-column confidence of v_min (0..1)
    t_rpeaks: np.ndarray  # R-peaks (s)
    t_clicks: np.ndarray  # valve clicks (s)
    t_avc: np.ndarray  # aortic valve closure (s)
    ecg_px: np.ndarray  # ECG in display pixels, upward positive
    flags: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    spectrogram: Spectrogram | None = None  # for plot_result

    SIGNALS = ("ecg", "v_mean", "v_mean_pos", "v_mean_neg", "v_max", "v_min")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"t": self.t, **{k: getattr(self, k) for k in self.SIGNALS}}

    def to_csv(self, path: str | Path) -> None:
        d = self.as_dict()
        np.savetxt(path, np.column_stack(list(d.values())), delimiter=",", header=",".join(d),
                   comments="", fmt="%.6g")


def extract_waveforms(source: str | Path | DicomImage,
                      config: Config | dict | str | Path | None = None) -> WaveformResult:
    """Extract ECG and Doppler envelopes from a spectral / tissue Doppler DICOM.

    source: DICOM path or a DicomImage from load_dicom.
    config: Config, a nested dict of overrides, a YAML path, or None (defaults).
    """
    t0 = time.perf_counter()
    cfg = load_config(config)
    img = source if isinstance(source, DicomImage) else load_dicom(source, frame=cfg.frame)
    region = spectral_region(img)
    bmode = bmode_region_above(img, region)

    # baseline colour first, so the ECG search can exclude it
    rgb = img.pixels if img.is_rgb else np.repeat(img.pixels[..., None], 3, axis=2)
    crop = rgb[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    zr = region.ref_y0 if region.ref_y0 is not None else crop.shape[0] / 2
    _, base_colour, _ = find_baseline(crop, zr, cfg.render)
    exclude = (tuple(base_colour),) if base_colour is not None else ()

    trace = extract_ecg(img, region, bmode, cfg.ecg, exclude_colours=exclude)
    overlay = ndimage.binary_dilation(trace.mask, iterations=1) if trace.found else None
    spec = build_spectrogram(img, region, cfg.render, extra_overlay=overlay)

    # envelope, with aliasing test (PW/TDI) and unwrap
    ns = noise_stats(spec, cfg.likelihood)
    env = dp_envelope(spec, ns, cfg.likelihood, cfg.dp)
    alias = {"suspected": False, "corrected": False}
    if spec.modality != "CW":
        alias["suspected"] = aliasing_suspected(spec, env["k_pos"], env["k_neg"], cfg.aliasing)
        if alias["suspected"] and cfg.aliasing.correct:
            spec_u, info = unwrap_aliasing(spec, env["density"], cfg.aliasing)
            alias.update(info)
            if info["corrected"]:
                spec = spec_u
                ns = noise_stats(spec, cfg.likelihood)
                env = dp_envelope(spec, ns, cfg.likelihood, cfg.dp)
    k_pos, k_neg = env["k_pos"].astype(float), env["k_neg"].astype(float)
    s = abs(spec.dy)
    o = spec.order
    if cfg.post.method != "none":  # smoothed in cm/s, in time order
        for k in (k_pos, k_neg):
            k[o] = np.clip(smooth_series(k[o] * s, spec.dt, cfg.post) / s, 0, None)
    means = mean_envelopes(spec, k_pos, k_neg, ns, cfg.mean)

    # ECG on the same columns; no data in the sweep
    if np.any(~np.isnan(trace.row)):
        ecg_px = -(trace.row - np.nanmedian(trace.row))
        scale = np.nanmax(np.abs(ecg_px))
        ecg = ecg_px / (scale if scale > 0 else 1.0)
    else:
        ecg_px, ecg = trace.row.copy(), trace.row.copy()
    ecg[spec.sweep] = np.nan
    ecg_px[spec.sweep] = np.nan

    t = np.arange(len(o)) * spec.dt
    try:
        hr_hint = float(img.meta.get("HeartRate")) if img.meta.get("HeartRate") else None
    except (TypeError, ValueError):
        hr_hint = None
    t_r = (detect_rpeaks(ecg[o], t, cfg.ecg, heart_rate_hint=hr_hint)
           if (trace.found and cfg.ecg.detect_rpeaks) else np.array([]))
    t_clicks = click_events(spec, t)
    t_avc = aortic_valve_closure(spec, t, t_r, cfg.ecg) if spec.click_mode != "off" else np.array([])

    flags = {
        "modality": spec.modality,
        "colormap": spec.colormap,
        "ecg_found": trace.found,
        "ecg_clipped": trace.clipped,
        "sweep_columns": int(spec.sweep.sum()),
        "aliasing_suspected": bool(alias["suspected"]),
        "aliasing_corrected": bool(alias["corrected"]),
        "n_clicks": int(len(t_clicks)),
        "snr_proxy_db": snr_proxy_db(spec, _flow_mask(spec, k_pos, k_neg), cfg.mean.dynamic_range_db),
    }
    flags.update({k: v for k, v in spec.info.items() if k.startswith("warning")})
    meta = {
        "version": __version__,
        "config": dataclasses.asdict(cfg),
        "source": str(img.path),
        "dicom": img.meta,
        "region": {"x0": region.x0, "y0": region.y0, "x1": region.x1, "y1": region.y1,
                   "dx_s": region.dx, "dy_cm_s": region.dy, "ref_y0": region.ref_y0},
        "velocity_limits_cm_s": spec.display_v_limits,
        "aliasing": alias,
        "noise": vars(ns),
        "runtime_s": time.perf_counter() - t0,
    }
    return WaveformResult(
        t=t, ecg=ecg[o], ecg_px=ecg_px[o],
        v_mean=means["v_mean"][o], v_mean_pos=means["v_mean_pos"][o], v_mean_neg=means["v_mean_neg"][o],
        v_max=(k_pos * s)[o], v_min=(-k_neg * s)[o],
        conf_pos=env["conf_pos"][o], conf_neg=env["conf_neg"][o],
        t_rpeaks=t_r, t_clicks=t_clicks, t_avc=t_avc, flags=flags, meta=meta, spectrogram=spec,
    )


# =============================================================================
# QC figure (optional, needs matplotlib)
# =============================================================================

def plot_result(res: WaveformResult, path: str | Path | None = None, title: str = ""):
    """Spectral display with envelopes, ECG with R-peaks / clicks / AVC, and the velocity signals.
    Shows only the spectral region and ECG (no header area), but treat it as patient data."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = res.spectrogram
    rgb = spec.rgb[:, spec.order]
    t, f = res.t, res.flags
    v_top = (0 - spec.zero_row) * spec.dy
    v_bot = (rgb.shape[0] - 1 - spec.zero_row) * spec.dy
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 1, 2]))
    ax[0].imshow(rgb, aspect="auto", extent=(t[0], t[-1] + spec.dt, v_bot, v_top), origin="upper",
                 interpolation="nearest")
    if v_top < v_bot:
        ax[0].invert_yaxis()
    ax[0].plot(t, res.v_max, "r", lw=1.0, label="v_max")
    ax[0].plot(t, res.v_min, "c", lw=1.0, label="v_min")
    ax[0].plot(t, res.v_mean_pos, "y", lw=0.8, label="v_mean_pos")
    ax[0].plot(t, res.v_mean_neg, "m", lw=0.8, label="v_mean_neg")
    if f.get("aliasing_corrected"):
        for v in spec.display_v_limits:  # the scanner's display window
            ax[0].axhline(v, color="w", ls="--", lw=0.6)
    alias = ", aliasing corrected" if f.get("aliasing_corrected") else (
        ", ALIASING?" if f.get("aliasing_suspected") else "")
    ax[0].set_title(f"{title}  [{f.get('modality')}, SNR proxy {f.get('snr_proxy_db', float('nan')):.1f} dB"
                    f"{alias}]", fontsize=10)
    ax[0].set_ylabel("velocity (cm/s)")
    ax[0].legend(loc="upper right", fontsize=7, ncol=4)
    ax[1].plot(t, res.ecg, "g", lw=0.8)
    for tr in res.t_rpeaks:
        for a in ax:
            a.axvline(tr, color="k", ls=":", lw=0.6)
    for tc in res.t_clicks:
        ax[1].axvline(tc, color="0.6", ls="--", lw=0.5)
    for ta in res.t_avc:
        ax[1].axvline(ta, color="m", ls="--", lw=0.9)
    ax[1].set_ylabel("ECG (a.u.)")
    ax[1].set_title("ECG" + (" (clipped by display)" if f.get("ecg_clipped") else "")
                    + "   R-peaks dotted, valve clicks dashed grey, aortic valve closure dashed magenta",
                    fontsize=9)
    ax[2].plot(t, res.v_max, "r", lw=1, label="v_max")
    ax[2].plot(t, res.v_min, "b", lw=1, label="v_min")
    ax[2].plot(t, res.v_mean, "k", lw=1, label="v_mean")
    ax[2].plot(t, res.v_mean_pos, color="orange", lw=0.8, label="v_mean_pos")
    ax[2].plot(t, res.v_mean_neg, color="purple", lw=0.8, label="v_mean_neg")
    ax[2].axhline(0, color="gray", lw=0.5)
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("velocity (cm/s)")
    ax[2].legend(loc="upper right", fontsize=7, ncol=5)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=80)
        plt.close(fig)
    return fig


# =============================================================================
# Command line
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description="ECG and Doppler envelopes from spectral / tissue Doppler DICOMs.")
    p.add_argument("dicoms", nargs="+", help="DICOM file(s)")
    p.add_argument("--config", help="YAML file with parameter overrides (see dicom2waveform_config.yaml)")
    p.add_argument("--out-dir", default=".", help="output folder (default: current folder)")
    p.add_argument("--plot", action="store_true", help="also write a QC figure per file (needs matplotlib)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for f in a.dicoms:
        name = Path(f).stem
        try:
            res = extract_waveforms(f, cfg)
        except Exception as e:  # keep going over a folder of files
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
            n_fail += 1
            continue
        res.to_csv(out / f"{name}.csv")
        events = {"t_rpeaks": res.t_rpeaks.tolist(), "t_clicks": res.t_clicks.tolist(),
                  "t_avc": res.t_avc.tolist(), "flags": res.flags}
        with open(out / f"{name}_events.json", "w", encoding="utf8") as fh:
            json.dump(events, fh, indent=2, default=str)
        if a.plot:
            plot_result(res, out / f"{name}.png", title=name)
        print(f"{name}: {res.flags['modality']}, {len(res.t_rpeaks)} R-peaks, {len(res.t_avc)} AVC, "
              f"aliasing {'corrected' if res.flags['aliasing_corrected'] else 'no'}, "
              f"{res.meta['runtime_s']:.2f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
