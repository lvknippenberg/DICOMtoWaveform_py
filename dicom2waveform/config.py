"""Configuration for the extraction pipeline.

All tunable parameters live here, as dataclasses with documented defaults. This file is the single
source of truth: the commented YAML reference (standalone/dicom2waveform_config.yaml) is generated
from it by `config_yaml()`, comments included. Document a parameter in the comment block directly
above its field; the class docstring is the heading of its YAML section. [TUNE] marks the settings
most worth tuning.

Wherever a parameter describes a physical extent it is given in physical units (s, cm/s, cm/s^2)
and converted to pixels per image, because pixel sizes differ by a factor of ~30 in velocity
between TDI and CW recordings and by ~1.3 in time between sweep speeds.

A configuration can be overridden from YAML or a dict (see `load_config`); only the keys that are
given are changed, unknown keys raise KeyError.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RenderConfig:
    """render: from the rendered spectral display to a clean scalar intensity image"""

    # Pixels whose RGB channels differ by more than this (grey levels) count as "chromatic".
    # On grey displays chromatic pixels are overlays (ECG, markers, cursors) and are masked.
    chroma_threshold: int = 12
    # If more than this fraction of the non-black, non-overlay pixels is chromatic, the display
    # is a colour map (TDI copper map) and is converted by luminance instead of grey level.
    colormap_fraction: float = 0.5
    # Rows either side of the header's zero-velocity row searched for the drawn baseline.
    baseline_search_rows: int = 6
    # A run of columns without the drawn baseline at least this wide (px) is the sweep: the write
    # position of the circular display buffer. It holds no data and is returned as NaN.
    min_sweep_width_px: int = 3
    # Rotate all outputs so samples run oldest -> newest (monotonic time). false = display order.
    align_to_sweep: bool = True
    # Colour tolerance (max abs channel difference) when matching overlay colours (baseline).
    overlay_colour_tolerance: int = 6
    # White text / scale ticks inside the region are masked: connected components of pixels with
    # level >= text_level, at most text_max_area px, whose surroundings average <= text_ring_max.
    # (Saturated flow and valve clicks also reach white, but have a bright surround.)
    text_level: float = 240.0
    text_max_area: int = 600
    text_ring_max: float = 70.0
    # [TUNE] Valve clicks (short broadband vertical lines: the leaflets, not blood flow):
    #   keep   - envelope traces the displayed spectrum including the clicks (default; chosen on
    #            the cardiac examples, 2026-10-05: the envelope should trace the spectrum as shown)
    #   bridge - click columns are interpolated over, so the envelope ignores them
    #   follow - clicks kept, and the envelope may jump to them without smoothness penalty
    #   off    - no click detection at all (then also no t_clicks and no t_avc)
    # Click times (t_clicks) and aortic valve closure (t_avc) are reported in every mode but 'off'.
    click_mode: str = "keep"
    # 'bridge' / 'follow' apply only to these modalities; others fall back to 'keep'. On TDI,
    # narrow transients are mostly tissue motion (IVC/IVR spikes), which must be kept.
    click_bridge_modalities: tuple[str, ...] = ("PW", "CW")
    # A click is a column-mean transient at least click_min_snr robust SDs high (white top-hat
    # filtered), at most click_max_width_s wide.
    click_min_snr: float = 12.0
    click_max_width_s: float = 0.035


@dataclass
class ECGConfig:
    """ecg: tracing the burned-in ECG, R-peaks, aortic valve closure"""

    # The ECG is searched in a band of rows from search_above_px above the lower edge of the
    # B-mode region down to search_margin_px into the Doppler region.
    search_margin_px: int = 100
    search_above_px: int = 120
    # Colour tolerance when matching the ECG trace colour.
    colour_tolerance: int = 6
    # Trace pieces narrower than this fraction of the region width are dropped (unless they lie
    # within the vertical band of the main trace on colour images).
    min_component_width_frac: float = 0.03
    # Longest gap (s) in the drawn trace that is interpolated (never across the sweep).
    max_gap_s: float = 0.05
    # Run the R-peak detector.
    detect_rpeaks: bool = True
    # [TUNE] Aortic valve closure (AVC) = strongest valve click in this window (s) after each
    # R-peak (closure fell 341-377 ms after R on the EMW validation set) ...
    avc_window_s: tuple[float, float] = (0.20, 0.50)
    # ... not within this distance (s) of the window edges ...
    avc_edge_guard_s: float = 0.012
    # ... at least this many robust SDs high (lower than click_min_snr: the R-gating makes that
    # safe). One true closure at 5.7 was missed with 6.0; lowering it risks false closures ...
    avc_min_snr: float = 6.0
    # ... timed at the half-maximum centroid over this span (s).
    avc_centroid_s: float = 0.035
    # R-peak detector (Pan-Tompkins style on QRS-band energy). The 8 Hz low edge is a
    # compromise: 5 Hz passed tall T-waves, 10 Hz dropped wide/premature beats.
    qrs_band_hz: tuple[float, float] = (8.0, 30.0)
    # Minimum distance between beats (s).
    refractory_s: float = 0.25
    # Moving-average window of the QRS energy (s).
    energy_smooth_s: float = 0.10
    # Peak threshold as a fraction of the typical QRS energy (upper-half median of candidates).
    thresh_fraction: float = 0.35
    # Noise floor as a fraction of the typical QRS energy (search-back uses half of it).
    noise_floor_fraction: float = 0.15
    # Search back for a missed beat in RR gaps longer than this times the expected RR
    # (expected RR from the DICOM HeartRate tag if present, else the median RR).
    searchback_gap_ratio: float = 1.66
    # T-wave guard: a peak this fraction of RR after the previous beat ...
    twave_rr_fraction: tuple[float, float] = (0.20, 0.55)
    # ... and lower than this fraction of the previous and of the median peak is dropped.
    twave_relative_height: float = 0.5


@dataclass
class LikelihoodConfig:
    """likelihood: noise-referenced soft signal membership and its local density"""

    # Signal threshold = noise median + max(noise_margin, noise_spread_factor * robust SD of the
    # noise), capped at the Otsu threshold. Membership ramps 0 -> 1 from just above the noise
    # (median + soft_low_factor * SD) to that threshold, so faint flow counts partially.
    # Lower noise_margin = more faint signal included (grey levels).
    noise_margin: float = 32.0
    noise_spread_factor: float = 4.0
    soft_low_factor: float = 1.5
    # [TUNE] Window over which membership is averaged into a local density. Given in cm/s / s if
    # > 0, otherwise in pixels (rows x columns). A smaller window keeps narrow, faint peak tips
    # (5 x 3 px was better on PW mitral E/A peaks, but pushes CW envelopes further out).
    density_window_cm_s: float = 0.0
    density_window_px_rows: int = 9
    density_window_s: float = 0.0
    density_window_px_cols: int = 5


@dataclass
class DPConfig:
    """dp: outer envelope by dynamic programming (globally optimal smooth edge per side)"""

    # [TUNE] Smoothness prior: acceleration (cm/s^2) at which moving the edge costs as much as one
    # misclassified pixel. Lower = smoother envelope, but may clip steep systolic upstrokes.
    reference_accel_cm_s2: float = 640.0
    # Scale reference_accel_cm_s2 with the velocity range of the display (relative to
    # reference_range_cm_s), so high-range CW displays are not over-smoothed (a fixed value
    # clipped upstrokes on high-velocity-scale records in the MATLAB tool).
    accel_scale_with_range: bool = True
    reference_range_cm_s: float = 100.0
    # Data term: 'edge' (outer-edge model, default) or 'split' (the MATLAB two-region split,
    # which assumes a flow region filled from the baseline and collapses on hollow spectra).
    cost: str = "edge"
    # [TUNE] The main envelope setting: the edge is placed where the local signal density drops to
    # this fraction of the typical in-flow density. Lower = edge further out on the faint fringe.
    # 0.5 was chosen by leave-one-image-out on vendor-traced vascular images (MAE ~2 cm/s).
    # Against cardiac annotations it sits ~2.5 cm/s inside on PW and ~10 cm/s outside on CW.
    edge_density: float = 0.5
    # Separate edge density for TDI displays (null = use edge_density).
    edge_density_tdi: float | None = None
    # Rows just inside the edge (fraction of display height) that must hold signal.
    inner_band_rel: float = 0.05
    # Weak prior toward "no flow" (cost per row of edge height).
    height_prior: float = 0.01
    # Band next to the baseline (fraction of display height) treated as the wall-filter zone:
    # nearly free to include; an edge inside it is reported as 0 (no flow on that side).
    wall_band_rel: float = 0.03
    # Percentile of the density taken as the noise background and subtracted (null/0 = off).
    # Without it, uniform noise pulls the edge out to the display limit.
    background_percentile: float | None = 50.0
    # Normalise the density by this percentile (the typical in-flow density), at least
    # normalise_floor (null/0 = off). Lets one edge_density serve dense (CW, vascular) and
    # sparse (speckled TDI) renderings. Chosen on the synthetic dev split + vendor traces.
    normalise_percentile: float | None = 99.0
    normalise_floor: float = 0.2
    # Local normalisation ("half maximum per column"): additionally divide each side's density by
    # its maximum within +-local_norm_s (s), but never by less than local_norm_floor. Keeps faint
    # low-velocity TDI segments (real signal at only 0.2-0.35 of the band density). On CW it
    # stretches clutter columns, hence only local_norm_modalities; 0 disables.
    local_norm_s: float = 0.01
    local_norm_floor: float = 0.15
    local_norm_modalities: tuple[str, ...] = ("TDI",)


@dataclass
class AliasingConfig:
    """aliasing: PW / TDI wrap-around detection and automatic per-column baseline shift"""

    # Correct detected aliasing (v_max / v_min may then exceed the display limits).
    # false = only flag it (flags.aliasing_suspected).
    correct: bool = True
    # Detection: aliasing is suspected when, in more than alias_min_fraction of the data columns,
    # both outer edges are within alias_edge_fraction of their display limits.
    alias_edge_fraction: float = 0.02
    alias_min_fraction: float = 0.01
    # Correction: a least-signal cut path through the circular display; rows beyond it move by
    # one display span. The cut stays at least guard_rel (fraction of height) from the baseline ...
    guard_rel: float = 0.10
    # ... the density is smoothed over cut_smooth_rows rows first ...
    cut_smooth_rows: int = 5
    # ... moving the cut away from the display edge costs edge_preference per row ...
    edge_preference: float = 0.002
    # ... the cut path costs cut_lambda per row moved between columns, plus cut_cross_weight per
    # row of signal density it passes over (density present in both columns: the cut may not jump
    # through a flow band, e.g. through the hollow PW spectral window near the baseline) ...
    cut_lambda: float = 0.05
    cut_cross_weight: float = 0.25
    # ... and it is applied only if the cut crosses less than max_cost_ratio of the signal density
    # at the display edge, and that edge density exceeds min_edge_density.
    max_cost_ratio: float = 0.5
    min_edge_density: float = 0.05


@dataclass
class MeanConfig:
    """mean: mean velocity envelopes inside the detected flow region"""

    # Pixel weights: 'power' (de-compressed display level, proxy of the power-weighted mean
    # velocity), 'level' (display level minus noise) or 'membership' (soft membership).
    weighting: str = "power"
    # Assumed dynamic range (dB) spanned by grey levels 0..255, to undo log compression for
    # 'power' weighting and the SNR proxy. Not stored in the DICOM.
    dynamic_range_db: float = 40.0
    # Band around the baseline (cm/s, each side) excluded from the means (wall filter / clutter).
    clutter_band_cm_s: float = 0.0
    # Minimum number of pixels in a column side for a side mean; otherwise 0 (no flow).
    min_pixels: int = 2


@dataclass
class PostConfig:
    """post: optional smoothing of v_max / v_min (in time order, never across the sweep)"""

    # 'none', 'median', 'savgol' (Savitzky-Golay), 'kalman_cv' or 'kalman_ca' (constant velocity /
    # acceleration Kalman filter with RTS smoother). The DP already enforces smoothness.
    method: str = "none"
    median_window_s: float = 0.015
    savgol_window_s: float = 0.025
    savgol_order: int = 2
    # Kalman process noise, expressed as an acceleration (cv) or jerk (ca) SD.
    kalman_q: float = 2000.0
    # Kalman measurement noise SD in cm/s (0 -> 2 pixels of the display).
    kalman_r_cm_s: float = 0.0


@dataclass
class EstimatorConfig:
    """estimator: which envelope estimator is used (benchmark option)"""

    # 'dp' (the method described in docs/explainer). The package also offers the benchmark
    # estimators (threshold, otsu, percentile, gm, mgm, snsi, asnsi, qia, latham, segmentation);
    # the standalone script has 'dp' only.
    method: str = "dp"
    # Keyword arguments for the estimator (free-form).
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Complete configuration."""

    # Frame to read from a multi-frame DICOM (0 = first).
    frame: int = 0
    render: RenderConfig = field(default_factory=RenderConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    aliasing: AliasingConfig = field(default_factory=AliasingConfig)
    mean: MeanConfig = field(default_factory=MeanConfig)
    post: PostConfig = field(default_factory=PostConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    # Keep intermediate results (spectrogram, estimate, ECG trace, flow mask) in
    # WaveformResult.diagnostics; plot_result and plot_overlay need them.
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


def load_config(source: str | Path | dict | PipelineConfig | None = None) -> PipelineConfig:
    """Default configuration, optionally overridden by a YAML file or a nested dict.

    Only the keys that are given are changed; unknown keys raise KeyError.
    """
    if isinstance(source, PipelineConfig):
        return source
    cfg = PipelineConfig()
    if source is None:
        return cfg
    if isinstance(source, dict):
        return _update(cfg, source)
    import yaml  # only needed for YAML files

    with open(source, encoding="utf8") as fh:
        values = yaml.safe_load(fh) or {}
    return _update(cfg, values)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------------------------
# Commented YAML, generated from the dataclasses and the comments above their fields
# ---------------------------------------------------------------------------------------------

YAML_HEADER = """\
Configuration for dicom2waveform (package and standalone script).

GENERATED from dicom2waveform/config.py by scripts/build_standalone.py: do not edit by hand.
Every value below is the built-in default, so this file is optional. To change a setting, copy
this file and keep only the keys you want to change (any nesting level); missing keys keep
their default.
  python dicom2waveform_standalone.py file.dcm --config my_config.yaml
  extract_waveforms("file.dcm", "my_config.yaml")
  extract_waveforms("file.dcm", {"dp": {"edge_density": 0.4}})     # same, as a dict

Physical extents are given in s, cm/s or cm/s^2 where possible and converted to pixels per
image (pixel sizes differ ~30x in velocity between TDI and CW and ~1.3x in time between
sweep speeds). "px" means pixels of the stored image.

The settings most worth tuning are marked [TUNE]. The rest are rendering details of the
scanner display (tested on Philips iE33, EPIQ CVx, Affiniti 70C, iU22) and rarely need changes."""


def _field_comments() -> dict[str, tuple[str, dict[str, list[str]]]]:
    """{class name: (docstring, {field: comment lines directly above the field})}, read from the
    source of this module (also works when it is part of the single-file standalone script)."""
    import ast
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    lines = src.splitlines()
    out = {}
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.ClassDef) and node.name.endswith("Config")):
            continue
        fields = {}
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                block, i = [], st.lineno - 2
                while i >= 0 and lines[i].lstrip().startswith("#"):
                    block.insert(0, lines[i].lstrip()[1:].rstrip())
                    i -= 1
                fields[st.target.id] = [b[1:] if b.startswith(" ") else b for b in block]
        out[node.name] = (ast.get_docstring(node) or "", fields)
    return out


def _yaml_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        plain = v and v.replace("_", "").isalnum() and v.lower() not in ("null", "true", "false", "yes", "no")
        return v if plain else '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_yaml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot write {type(v).__name__} to YAML")


def config_yaml(cfg: PipelineConfig | None = None, header: str = YAML_HEADER) -> str:
    """The configuration as YAML, with the documentation comments of this module."""
    cfg = cfg or PipelineConfig()
    docs = _field_comments()
    rule = "# " + "-" * 93
    out = [f"# {line}".rstrip() for line in header.splitlines()]

    def comments(cls_name: str, name: str, indent: str) -> list[str]:
        return [f"{indent}# {c}".rstrip() for c in docs.get(cls_name, ("", {}))[1].get(name, [])]

    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if dataclasses.is_dataclass(val):
            title = docs.get(type(val).__name__, (f.name, {}))[0].splitlines()[0]
            out += ["", rule, f"# {title}", rule, f"{f.name}:"]
            for g in dataclasses.fields(val):
                out += comments(type(val).__name__, g.name, "  ")
                out.append(f"  {g.name}: {_yaml_value(getattr(val, g.name))}")
        else:
            out += [""] + comments(type(cfg).__name__, f.name, "")
            out.append(f"{f.name}: {_yaml_value(val)}")
    return "\n".join(out) + "\n"


def write_config(cfg: PipelineConfig | None, path: str | Path, header: str = YAML_HEADER) -> None:
    """Write a configuration as commented YAML, loadable by load_config."""
    Path(path).write_text(config_yaml(cfg, header), encoding="utf8")
