"""Configuration for the extraction pipeline.

All tunable parameters live here, as dataclasses with documented defaults.
Wherever a parameter describes a physical extent it is given in physical
units (s, cm/s, cm/s^2) and converted to pixels per image, because pixel
sizes differ by a factor of ~30 in velocity between TDI and CW recordings
and by ~1.3 in time between sweep speeds.

A configuration can be overridden from YAML (see `load_config`); only the
keys that are given are changed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RenderConfig:
    """Conversion of the rendered Doppler region to a scalar intensity image."""

    # Pixels whose colour channels differ by more than this are "chromatic".
    chroma_threshold: int = 12
    # Fraction of non-black, non-overlay pixels that must be chromatic for the
    # spectral display to be treated as a colour map (TDI copper map).
    colormap_fraction: float = 0.5
    # Rows either side of the zero-velocity row searched for the drawn baseline.
    baseline_search_rows: int = 6
    # A run of columns without a drawn baseline at least this long is the
    # sweep (write position of the circular display buffer).
    min_sweep_width_px: int = 3
    # Rotate outputs so that samples run oldest -> newest (monotonic time).
    align_to_sweep: bool = True
    # Colour tolerance (max abs channel difference) for matching overlay colours.
    overlay_colour_tolerance: int = 6
    # White annotations inside the region: components of pixels >= text_level,
    # at most text_max_area pixels, whose surroundings average <= text_ring_max.
    text_level: float = 240.0
    text_max_area: int = 600
    text_ring_max: float = 70.0
    # Valve clicks: column-mean transients above click_min_snr robust SDs over a
    # moving median of click_background_s, at most click_max_width_s wide.
    # click_mode: 'keep' (default) keeps them without special treatment, so the
    # envelope traces the displayed spectrum including the click (chosen by the
    # user on the cardiac examples, 2026-10-05: the envelope should trace the
    # spectrum as displayed); 'bridge' replaces click columns by interpolation so
    # the envelope ignores them (a click is the valve leaflet, not blood flow);
    # 'follow' keeps them and lets the DP envelope jump to them; 'off' disables
    # detection. Click times are reported (t_clicks, t_avc) in every mode except 'off'.
    click_mode: str = "keep"
    # 'bridge'/'follow' apply to these modalities; others use 'keep'. On TDI,
    # narrow transients are mostly tissue motion (IVC/IVR spikes), not clicks.
    click_bridge_modalities: tuple[str, ...] = ("PW", "CW")
    click_min_snr: float = 12.0
    click_background_s: float = 0.15
    click_max_width_s: float = 0.035


@dataclass
class ECGConfig:
    # Rows below the top of the Doppler region the ECG may extend into.
    search_margin_px: int = 100
    # Rows above the bottom of the B-mode region the ECG may extend into.
    search_above_px: int = 120
    # Colour tolerance when matching the ECG trace colour.
    colour_tolerance: int = 6
    # Components narrower than this fraction of the width are dropped unless
    # they lie within the vertical band of the main trace.
    min_component_width_frac: float = 0.03
    # Longest gap (s) in the drawn trace that is interpolated (never across the sweep).
    max_gap_s: float = 0.05
    detect_rpeaks: bool = True
    # Aortic valve closure = strongest valve click in this window after each
    # R-peak (MATLAB GetAorticValveClosure; closure fell 341-377 ms after R on
    # the EMW set), rejected within edge_guard_s of the window edges.
    avc_window_s: tuple[float, float] = (0.20, 0.50)
    avc_edge_guard_s: float = 0.012
    # Minimum transient height (robust SDs) inside the gated window (MATLAB: 6),
    # and the span over which the closure time is centred.
    avc_min_snr: float = 6.0
    avc_centroid_s: float = 0.035
    # R-peak detector (port of DetectRpeaks.m)
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
    """Noise-referenced soft signal membership (shared by several estimators)."""

    # Threshold = noise + max(noise_margin, noise_spread_factor * robust SD),
    # capped at the Otsu threshold (see envelope/likelihood.py).
    noise_margin: float = 32.0
    noise_spread_factor: float = 4.0
    soft_low_factor: float = 1.5
    # Window over which local membership density is averaged (physical units).
    density_window_cm_s: float = 0.0  # 0 -> use density_window_px_rows
    density_window_px_rows: int = 9
    density_window_s: float = 0.0  # 0 -> use density_window_px_cols
    density_window_px_cols: int = 5


@dataclass
class DPConfig:
    # Acceleration (cm/s^2) at which moving the boundary costs as much as one
    # misclassified pixel. Lower = smoother (and may clip systolic peaks).
    reference_accel_cm_s2: float = 640.0
    # Scale the reference acceleration with the velocity range of the display,
    # relative to a reference range; 0 disables. (Open issue in MATLAB:
    # a fixed a_ref clips upstrokes on high-velocity-scale records.)
    accel_scale_with_range: bool = True
    reference_range_cm_s: float = 100.0
    # Data term: 'edge' (outer-edge model, default) or 'split' (MATLAB
    # two-region split, assumes a filled flow region).
    cost: str = "edge"
    # 'edge': local signal density at which the edge is placed. 0.5 was chosen
    # by leave-one-image-out on the vendor-traced images (held-out MAE 2.2
    # cm/s); lower values place the edge further out on the faint fringe ...
    edge_density: float = 0.5
    # ... rows just inside the edge that must hold signal (fraction of height) ...
    inner_band_rel: float = 0.05
    # ... and a weak prior toward "no flow" (cost per row of edge height).
    height_prior: float = 0.01
    # Band next to the baseline (fraction of the display height) treated as
    # the wall-filter zone: nearly free to include in the flow region.
    wall_band_rel: float = 0.03
    # Percentile of the likelihood image taken as the noise background and
    # subtracted (None/0 = no correction).
    background_percentile: float | None = 50.0
    # Normalise the density by this percentile (the typical in-flow density),
    # at least normalise_floor; None/0 = no normalisation. With edge_density
    # 0.5 this places the edge at half the typical in-flow density. Chosen on
    # the synthetic dev split + vendor traces (see docs/BENCHMARK.md).
    normalise_percentile: float | None = 99.0
    normalise_floor: float = 0.2
    # Local normalisation ("half maximum per column"): additionally normalise
    # each side's density by its maximum within +-local_norm_s, but never by
    # less than local_norm_floor (relative to the globally normalised level 1),
    # so noise is not lifted. Applied only to the listed modalities. On TDI,
    # faint low-velocity segments are real signal (16-80x more non-black pixels
    # than motion-free regions in 01_10638) but only 0.2-0.35 of the band
    # density, so a global edge level drops them. On CW it stretches clutter
    # columns, hence TDI only. local_norm_s = 0 disables.
    local_norm_s: float = 0.01
    local_norm_floor: float = 0.15
    local_norm_modalities: tuple[str, ...] = ("TDI",)
    # Edge density for TDI displays (None = edge_density).
    edge_density_tdi: float | None = None
    # 'follow' click mode: transitions into/out of click columns are free, so
    # the edge can jump to the click (set automatically by render.click_mode).


@dataclass
class AliasingConfig:
    """Automatic per-column baseline shift (aliasing.py)."""

    correct: bool = True
    # Cut rows must stay this far (fraction of display height) from the zero row.
    guard_rel: float = 0.10
    # Smoothing of the density (rows) before choosing the cut.
    cut_smooth_rows: int = 5
    # Cost per row of moving the cut away from the display edge (prefers no change).
    edge_preference: float = 0.002
    # Smoothness of the cut path (cost per row moved between columns).
    cut_lambda: float = 0.05
    # Apply only if the cut path crosses less than this fraction of the signal
    # density at the display edge, and the edge density exceeds min_edge_density.
    max_cost_ratio: float = 0.5
    min_edge_density: float = 0.05


@dataclass
class MeanConfig:
    # Weighting of pixels for the mean velocity: 'level' (noise-subtracted
    # display level), 'membership' (soft membership), or 'power'
    # (de-compressed display level, see dynamic_range_db).
    weighting: str = "power"
    # Assumed display dynamic range (dB) spanned by grey levels 0..255, used
    # to undo log compression when weighting == 'power'.
    dynamic_range_db: float = 40.0
    # Band around the baseline (cm/s, each side) excluded from the pos/neg
    # means (wall filter / clutter). 0 = keep everything.
    clutter_band_cm_s: float = 0.0
    # Minimum number of weighted pixels in a column side for a mean to be
    # reported; otherwise the side mean is 0 (no flow).
    min_pixels: int = 2


@dataclass
class PostConfig:
    # Envelope post-processing: 'none', 'median', 'savgol', 'kalman_cv', 'kalman_ca'
    method: str = "none"
    median_window_s: float = 0.015
    savgol_window_s: float = 0.025
    savgol_order: int = 2
    # Kalman process noise, expressed as an acceleration (cv) or jerk (ca) SD.
    kalman_q: float = 2000.0
    # Measurement noise SD in cm/s (0 -> 2 pixels of the display).
    kalman_r_cm_s: float = 0.0


@dataclass
class EstimatorConfig:
    """Which envelope estimator is used, and its parameters (free-form dict)."""

    method: str = "dp"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityConfig:
    # A side edge within this fraction of the display limit counts as touching it.
    alias_edge_fraction: float = 0.02
    # Aliasing is suspected when this fraction of data columns touch the edge.
    alias_min_fraction: float = 0.01


@dataclass
class PipelineConfig:
    frame: int = 0
    render: RenderConfig = field(default_factory=RenderConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    mean: MeanConfig = field(default_factory=MeanConfig)
    post: PostConfig = field(default_factory=PostConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    aliasing: AliasingConfig = field(default_factory=AliasingConfig)
    keep_diagnostics: bool = False


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


def load_config(source: str | Path | dict | None = None) -> PipelineConfig:
    """Default configuration, optionally overridden by a YAML file or dict."""
    cfg = PipelineConfig()
    if source is None:
        return cfg
    if isinstance(source, dict):
        return _update(cfg, source)
    import yaml

    with open(source, encoding="utf8") as fh:
        values = yaml.safe_load(fh) or {}
    return _update(cfg, values)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def write_config(cfg: Any, path: str | Path, header: str = "") -> None:
    """Write a configuration as YAML (tuples as lists), loadable by load_config."""
    import yaml

    def plain(v):
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v

    with open(path, "w", encoding="utf8") as fh:
        if header:
            fh.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(plain(config_to_dict(cfg)), fh, sort_keys=False)
