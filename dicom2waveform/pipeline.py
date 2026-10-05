"""Top-level pipeline: DICOM -> ECG + Doppler envelopes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from . import __version__
from .config import PipelineConfig, config_to_dict, load_config
from .ecg import ECGTrace, detect_rpeaks, ecg_amplitude, extract_ecg
from .aliasing import unwrap
from .envelope import EnvelopeEstimate, EnvelopeEstimator, make_estimator
from .envelope.base import Context
from .envelope.dp import DPEstimator
from .envelope.mean import mean_envelopes
from .envelope.postprocess import smooth_series
from .io import DicomImage, bmode_region_above, load_dicom, spectral_region
from .quality import aliasing_suspected, flow_mask, snr_proxy_db
from .render import Spectrogram, build_spectrogram, find_baseline


@dataclass
class WaveformResult:
    """All signals share the time axis `t` (s), oldest sample first.

    Velocities in cm/s with the sign of the DICOM header (PhysicalDeltaY):
    on an operator-inverted display this is opposite to the on-screen label.
    NaN marks samples without data (the sweep).
    """

    t: np.ndarray
    ecg: np.ndarray  # a.u., isoelectric ~ 0, max |deflection| = 1
    v_mean: np.ndarray
    v_mean_pos: np.ndarray
    v_mean_neg: np.ndarray
    v_max: np.ndarray  # outer edge of the positive spectrum (>= 0)
    v_min: np.ndarray  # outer edge of the negative spectrum (<= 0)
    conf_pos: np.ndarray | None = None
    conf_neg: np.ndarray | None = None
    t_rpeaks: np.ndarray = field(default_factory=lambda: np.array([]))
    t_clicks: np.ndarray = field(default_factory=lambda: np.array([]))  # valve clicks (s)
    t_avc: np.ndarray = field(default_factory=lambda: np.array([]))  # aortic valve closure (s)
    ecg_px: np.ndarray | None = None  # ECG in display pixels, upward positive
    flags: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    SIGNALS = ("ecg", "v_mean", "v_mean_pos", "v_mean_neg", "v_max", "v_min")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"t": self.t, **{k: getattr(self, k) for k in self.SIGNALS}}

    def to_csv(self, path: str | Path) -> None:
        d = self.as_dict()
        cols = list(d)
        data = np.column_stack([d[c] for c in cols])
        np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")


def envelopes_from_spectrogram(
    spec: Spectrogram,
    cfg: PipelineConfig,
    estimator: EnvelopeEstimator | None = None,
) -> tuple[dict[str, np.ndarray], EnvelopeEstimate, Context]:
    """Envelopes in RAW column order (not yet sweep-aligned). Used by the
    pipeline and by the synthetic benchmark.

    Aliasing (PW/TDI): a DP pass on the display is tested for wrap-around; if it
    fires, the display is unwrapped (aliasing.unwrap) and the estimator runs on
    the unwrapped spectrogram, which is then `ctx.spec` (taller than the
    display; same columns). `ctx.aliasing` holds the detection/unwrap result.
    """
    est = estimator or make_estimator(cfg.estimator.method, **cfg.estimator.params)
    ctx = Context(spec, cfg.likelihood, cfg.dp)
    e = est.estimate(ctx)

    alias_info = {"suspected": False, "corrected": False}
    if spec.modality != "CW":
        # wrap-around test on the DP edges (the framework's detector, whatever the estimator)
        dp = e if e.method == "dp" else DPEstimator().estimate(ctx)
        alias_info["suspected"] = aliasing_suspected(spec, dp.k_pos, dp.k_neg, cfg.quality)
        if alias_info["suspected"] and cfg.aliasing.correct:
            density = dp.diagnostics.get("likelihood")
            if density is None:
                density = DPEstimator().likelihood(ctx)
            u = unwrap(spec, density, cfg.aliasing)
            alias_info.update(corrected=u.applied, columns=u.n_columns, rows_up=u.shift_up,
                              rows_down=u.shift_down, edge_density=u.edge_cost, cut_density=u.path_cost)
            if u.applied:
                spec = u.spec
                ctx = Context(spec, cfg.likelihood, cfg.dp)
                e = est.estimate(ctx)
    ctx.aliasing = alias_info
    k_pos, k_neg = e.k_pos.astype(float), e.k_neg.astype(float)

    # optional post-processing, in time order, per side
    s = abs(spec.dy)
    if cfg.post.method != "none":
        # smoothed in velocity units (cm/s), in time order
        o = spec.order
        smoothed = []
        for k in (k_pos, k_neg):
            ks = k.copy()
            ks[o] = smooth_series(k[o] * s, spec.dt, cfg.post, pixel_cm_s=s) / s
            smoothed.append(np.clip(ks, 0, None))
        k_pos, k_neg = smoothed
    out = {"v_max": k_pos * s, "v_min": -k_neg * s}
    out.update(mean_envelopes(spec, k_pos, k_neg, ctx.noise, cfg.mean))
    out["k_pos"], out["k_neg"] = k_pos, k_neg
    return out, e, ctx


def extract_waveforms(
    source: str | Path | DicomImage,
    config: PipelineConfig | dict | str | Path | None = None,
    estimator: EnvelopeEstimator | str | None = None,
) -> WaveformResult:
    """Extract ECG and Doppler envelopes from a spectral/tissue Doppler DICOM.

    Parameters
    ----------
    source : path to a DICOM file, or an already loaded DicomImage.
    config : PipelineConfig, a dict/YAML path of overrides, or None (defaults).
    estimator : envelope estimator instance or registry name; default from config (DP).

    Returns
    -------
    WaveformResult with t, ecg, v_mean, v_mean_pos, v_mean_neg, v_max, v_min.
    """
    t0 = time.perf_counter()
    cfg = config if isinstance(config, PipelineConfig) else load_config(config)
    if isinstance(estimator, str):
        estimator = make_estimator(estimator, **(cfg.estimator.params if estimator == cfg.estimator.method else {}))
    img = source if isinstance(source, DicomImage) else load_dicom(source, frame=cfg.frame)
    region = spectral_region(img)
    bmode = bmode_region_above(img, region)

    # Baseline colour first, so the ECG search can exclude it.
    rgb = img.pixels if img.is_rgb else np.repeat(img.pixels[..., None], 3, axis=2)
    crop = rgb[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    zr = region.ref_y0 if region.ref_y0 is not None else crop.shape[0] / 2
    _, base_colour, _ = find_baseline(crop, zr, cfg.render)
    exclude = (tuple(base_colour),) if base_colour is not None else ()

    trace: ECGTrace = extract_ecg(img, region, bmode, cfg.ecg, exclude_colours=exclude)
    overlay = ndimage.binary_dilation(trace.mask, iterations=1) if trace.found else None
    spec = build_spectrogram(img, region, cfg.render, extra_overlay=overlay)

    env, est_result, ctx = envelopes_from_spectrogram(spec, cfg, estimator)
    display_spec = spec  # as displayed (before any aliasing unwrap), for plotting
    spec = ctx.spec  # unwrapped if aliasing was corrected (same columns)

    # ECG on the same columns; no data in the sweep
    ecg, ecg_px = ecg_amplitude(trace.row)
    ecg[spec.sweep] = np.nan
    ecg_px[spec.sweep] = np.nan

    o = spec.order
    W = len(o)
    t = np.arange(W) * spec.dt
    hr_hint = img.meta.get("HeartRate")
    try:
        hr_hint = float(hr_hint) if hr_hint else None
    except (TypeError, ValueError):
        hr_hint = None
    t_r = detect_rpeaks(ecg[o], t, cfg.ecg, heart_rate_hint=hr_hint) if (trace.found and cfg.ecg.detect_rpeaks) else np.array([])

    t_clicks, click_strength = click_events(spec, t)
    t_avc = aortic_valve_closure(spec, t, t_r, cfg.ecg) if spec.click_mode != "off" else np.array([])

    flow = flow_mask(spec, env["k_pos"], env["k_neg"])
    flags = {
        "modality": spec.modality,
        "colormap": spec.colormap,
        "ecg_found": trace.found,
        "ecg_clipped": trace.clipped,
        "sweep_columns": int(spec.sweep.sum()),
        "aliasing_suspected": bool(ctx.aliasing.get("suspected", False)),
        "aliasing_corrected": bool(ctx.aliasing.get("corrected", False)),
        "n_clicks": int(len(t_clicks)),
        "snr_proxy_db": snr_proxy_db(spec, flow, cfg.mean.dynamic_range_db),
    }
    flags.update({k: v for k, v in spec.info.items() if k.startswith("warning")})

    meta = {
        "version": __version__,
        "method": est_result.method,
        "method_params": est_result.params,
        "config": config_to_dict(cfg),
        "source": str(img.path),
        "dicom": img.meta,
        "region": {"x0": region.x0, "y0": region.y0, "x1": region.x1, "y1": region.y1,
                   "dx_s": region.dx, "dy_cm_s": region.dy, "ref_y0": region.ref_y0,
                   "data_type": region.data_type_name},
        "zero_row": spec.zero_row,
        "velocity_limits_cm_s": spec.display_v_limits,
        "aliasing": ctx.aliasing,
        "noise": vars(ctx.noise),
        "runtime_s": time.perf_counter() - t0,
    }
    diag = {}
    if cfg.keep_diagnostics:
        diag = {"spectrogram": spec, "display_spectrogram": display_spec, "estimate": est_result,
                "ecg_trace": trace, "flow_mask": flow, "order": o}

    def r(a):
        return None if a is None else np.asarray(a, float)[o]

    return WaveformResult(
        t=t, ecg=ecg[o], ecg_px=ecg_px[o],
        v_mean=r(env["v_mean"]), v_mean_pos=r(env["v_mean_pos"]), v_mean_neg=r(env["v_mean_neg"]),
        v_max=r(env["v_max"]), v_min=r(env["v_min"]),
        conf_pos=r(est_result.conf_pos), conf_neg=r(est_result.conf_neg),
        t_rpeaks=t_r, t_clicks=t_clicks, t_avc=t_avc, flags=flags, meta=meta, diagnostics=diag,
    )


def click_events(spec: Spectrogram, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valve-click times (s, time order): centre of each run of click columns
    (the run is padded by one column each side), and its strength (transient
    height in robust SDs)."""
    if spec.clicks is None or not spec.clicks.any():
        return np.array([]), np.array([])
    o = spec.order
    c = spec.clicks[o]
    s = np.where(c, spec.click_strength[o], 0.0) if spec.click_strength is not None else c.astype(float)
    m = np.concatenate([[False], c, [False]]).astype(np.int8)
    starts = np.flatnonzero(np.diff(m) == 1)
    ends = np.flatnonzero(np.diff(m) == -1)
    times, strength = [], []
    for a, b in zip(starts, ends):
        seg = s[a:b]
        # half-maximum centroid of the transient: robust to a run that also
        # covers part of an adjacent transient
        w = np.clip(seg - 0.5 * seg.max(), 0.0, None)
        times.append(float(np.sum(w * t[a:b]) / np.sum(w)) if w.sum() > 0 else float(0.5 * (t[a] + t[b - 1])))
        strength.append(float(seg.max()))
    return np.array(times), np.array(strength)


def aortic_valve_closure(spec: Spectrogram, t: np.ndarray, t_rpeaks: np.ndarray, cfg) -> np.ndarray:
    """Aortic valve closure per beat (port of the MATLAB GetAorticValveClosure):
    the strongest click transient within cfg.avc_window_s after each R-peak,
    at least cfg.avc_min_snr robust SDs (lower than the general click threshold:
    the R-gating makes that safe), not pinned to the window edges. Timed at the
    half-maximum centroid of the transient."""
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
