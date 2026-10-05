"""Benchmark runner: every method variant on every case, with metrics.

A *variant* is (label, estimator name, estimator params, post-processing,
config overrides). The same rendered spectrogram of a case is shared by all
variants, so differences come from the estimators only.
"""

from __future__ import annotations

import copy
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import PipelineConfig, load_config
from ..envelope import BENCHMARK_ORDER, make_estimator
from ..io import bmode_region_above, spectral_region
from ..pipeline import envelopes_from_spectrogram
from ..quality import flow_mask, snr_proxy_db
from ..render import build_spectrogram, find_baseline
from ..ecg import extract_ecg
from .metrics import evaluate_signals


@dataclass
class Variant:
    label: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    post: str = "none"
    overrides: dict[str, Any] = field(default_factory=dict)
    group: str = "main"  # main | 2d_experiment | ablation | tuning


def main_variants() -> list[Variant]:
    """The ten benchmark methods (Plan section 23), with published/default settings."""
    return [Variant(m, m) for m in BENCHMARK_ORDER]


def experiment_2d_variants() -> list[Variant]:
    """Plan section 54: does 2-D information help?
    A column-wise, B column-wise + 1-D smoothing, C 2-D segmentation, D 2-D likelihood + global curve."""
    return [
        Variant("A_colwise_asnsi_raw", "asnsi", {"power_threshold": False, "one_third": False, "smooth_s": 0.0}, group="2d"),
        Variant("A_colwise_percentile", "percentile", group="2d"),
        Variant("A_colwise_mgm_nogate", "mgm", {"presence_gate": False}, group="2d"),
        Variant("A_colwise_qia_nogate", "qia", {"presence_gate": False}, group="2d"),
        Variant("A_colwise_qia_fullcolumn", "qia", {"mode": "full_column"}, group="2d"),
        Variant("B_colwise_asnsi_median", "asnsi", {"power_threshold": False, "one_third": False, "smooth_s": 0.0}, post="median", group="2d"),
        Variant("B_colwise_percentile_median", "percentile", post="median", group="2d"),
        Variant("B_colwise_percentile_kalman", "percentile", post="kalman_cv", group="2d"),
        Variant("C_segmentation", "segmentation", group="2d"),
        Variant("D_dp", "dp", group="2d"),
        Variant("D_dp_savgol", "dp", post="savgol", group="2d"),
    ]


def ablation_variants() -> list[Variant]:
    """Plan section 45 (image pipeline chain) and DP component ablations."""
    seg = {"components": "none", "morphology": [], "extract": "outermost"}
    return [
        # chain: raw -> smoothing -> threshold -> morphology -> components -> tracking
        Variant("abl1_raw_fixed", "segmentation", {**seg, "preprocess": "none", "binarize": "fixed"}, group="ablation"),
        Variant("abl2_smooth_fixed", "segmentation", {**seg, "preprocess": "gaussian", "binarize": "fixed"}, group="ablation"),
        Variant("abl3_smooth_otsu", "segmentation", {**seg, "preprocess": "gaussian", "binarize": "otsu"}, group="ablation"),
        Variant("abl4_+morphology", "segmentation", {**seg, "preprocess": "gaussian", "binarize": "otsu",
                                                      "morphology": [["close", 3], ["open", 3], ["fill"], ["remove_small", 60]]}, group="ablation"),
        Variant("abl5_+components", "segmentation", {"preprocess": "gaussian", "binarize": "otsu",
                                                      "morphology": [["close", 3], ["open", 3], ["fill"], ["remove_small", 60]],
                                                      "components": "plausible", "extract": "outermost"}, group="ablation"),
        Variant("abl6_+largest_run", "segmentation", {"preprocess": "gaussian", "binarize": "otsu",
                                                       "morphology": [["close", 3], ["open", 3], ["fill"], ["remove_small", 60]],
                                                       "components": "plausible", "extract": "largest_run"}, group="ablation"),
        Variant("abl7_bg_threshold", "segmentation", {"preprocess": "none", "binarize": "background",
                                                       "morphology": [["close", 3], ["open", 3], ["fill"], ["remove_small", 60]],
                                                       "components": "plausible", "extract": "outermost"}, group="ablation"),
        Variant("abl8_+dp_tracking", "dp", {"hard_threshold": True}, group="ablation"),
        # DP component ablations
        Variant("dp_matlab_split", "dp", overrides={"dp": {"cost": "split", "background_percentile": 0,
                                                            "normalise_percentile": 0, "wall_band_rel": 0.0}},
                group="ablation"),
        Variant("dp_split_cost", "dp", overrides={"dp": {"cost": "split"}}, group="ablation"),
        Variant("dp_no_wallband", "dp", overrides={"dp": {"wall_band_rel": 0.0}}, group="ablation"),
        Variant("dp_no_normalise", "dp", overrides={"dp": {"normalise_percentile": 0}}, group="ablation"),
        Variant("dp_theta_0.2", "dp", overrides={"dp": {"edge_density": 0.2}}, group="ablation"),
        Variant("dp_theta_0.35", "dp", overrides={"dp": {"edge_density": 0.35}}, group="ablation"),
        Variant("dp_theta_0.65", "dp", overrides={"dp": {"edge_density": 0.65}}, group="ablation"),
        Variant("dp_no_bgcorr", "dp", overrides={"dp": {"background_percentile": 0}}, group="ablation"),
        Variant("dp_hard_membership", "dp", {"hard_threshold": True}, group="ablation"),
        Variant("dp_no_density", "dp", {"use_density": False}, group="ablation"),
        Variant("dp_fixed_accel", "dp", {"accel_scale_with_range": False}, group="ablation"),
        Variant("dp_no_click_bridge", "dp", overrides={"render": {"click_mode": "keep"}}, group="ablation"),
        Variant("dp_follow_clicks", "dp", overrides={"render": {"click_mode": "follow"}}, group="ablation"),
        Variant("dp_no_alias_correction", "dp", overrides={"aliasing": {"correct": False}}, group="ablation"),
        Variant("dp_no_local_norm", "dp", overrides={"dp": {"local_norm_s": 0.0}}, group="ablation"),
        Variant("dp_local_norm_all", "dp", overrides={"dp": {"local_norm_modalities": ["PW", "CW", "TDI"]}}, group="ablation"),
        Variant("dp_tdi_theta_0.35", "dp", overrides={"dp": {"edge_density_tdi": 0.35}}, group="ablation"),
        Variant("dp_mean_level_w", "dp", overrides={"mean": {"weighting": "level"}}, group="ablation"),
        Variant("dp_mean_membership_w", "dp", overrides={"mean": {"weighting": "membership"}}, group="ablation"),
    ]


def _cfg_for(base: PipelineConfig, v: Variant) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    if v.overrides:
        from ..config import _update
        _update(cfg, copy.deepcopy(v.overrides))
    cfg.post.method = v.post
    return cfg


def _render(image, cfg: PipelineConfig):
    region = spectral_region(image)
    bmode = bmode_region_above(image, region)
    rgb = image.pixels if image.is_rgb else np.repeat(image.pixels[..., None], 3, axis=2)
    crop = rgb[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    zr = region.ref_y0 if region.ref_y0 is not None else crop.shape[0] / 2
    _, base_colour, _ = find_baseline(crop, zr, cfg.render)
    exclude = (tuple(base_colour),) if base_colour is not None else ()
    trace = extract_ecg(image, region, bmode, cfg.ecg, exclude_colours=exclude)
    from scipy import ndimage
    overlay = ndimage.binary_dilation(trace.mask, iterations=1) if trace.found else None
    return build_spectrogram(image, region, cfg.render, extra_overlay=overlay)


def run_variants_on_image(image, truth: dict[str, np.ndarray] | None, variants: list[Variant],
                          base: PipelineConfig | None = None, r_times=None, case_info: dict | None = None,
                          keep_series: bool = False, eval_mask_fn=None) -> list[dict[str, Any]]:
    """Run all variants on one image. `truth` in time order (same length as the display)."""
    base = base or load_config()
    rows = []
    spec_cache: dict[str, Any] = {}
    snr_true = np.nan
    for v in variants:
        cfg = _cfg_for(base, v)
        key = repr(sorted(cfg.render.__dict__.items()))
        try:
            if key not in spec_cache:
                spec_cache[key] = _render(image, cfg)
            spec = spec_cache[key]
            est = make_estimator(v.method, **v.params)
            t0 = time.perf_counter()
            env, e, ctx = envelopes_from_spectrogram(spec, cfg, est)
            runtime = time.perf_counter() - t0
            o = spec.order
            series = {k: np.asarray(env[k], float)[o] for k in ("v_max", "v_min", "v_mean", "v_mean_pos", "v_mean_neg")}
            span = spec.display_v_limits[1] - spec.display_v_limits[0]
            row = {**(case_info or {}), "variant": v.label, "method": v.method, "group": v.group,
                   "post": v.post, "runtime_s": runtime, "modality_detected": spec.modality}
            if truth is not None:
                if np.isnan(snr_true):
                    kp = np.nan_to_num(truth["v_max"] / abs(spec.dy))
                    kn = np.nan_to_num(-truth["v_min"] / abs(spec.dy))
                    inv = np.empty_like(o)
                    inv[o] = np.arange(len(o))
                    snr_true = snr_proxy_db(spec, flow_mask(spec, kp[inv], kn[inv]), cfg.mean.dynamic_range_db)
                mask = eval_mask_fn(spec) if eval_mask_fn else None
                row.update(evaluate_signals(series, truth, truth["t"], span, abs(spec.dy), r_times, mask=mask))
                row["snr_proxy_db"] = snr_true
            if keep_series:
                row["_series"] = series
            rows.append(row)
        except Exception as ex:  # record failures; the benchmark must not stop
            rows.append({**(case_info or {}), "variant": v.label, "method": v.method, "group": v.group,
                         "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc(limit=3)})
    return rows


# ---------------------------------------------------------------------------
# Synthetic benchmark (parallel)
# ---------------------------------------------------------------------------

def _synthetic_job(args):
    spec_dict, variant_dicts = args
    from ..synthetic import SyntheticSpec, generate
    sp = SyntheticSpec(**spec_dict)
    case = generate(sp)
    info = {"case": case.image.path, "modality": sp.modality + ("_alias" if sp.alias else ""), "alias": sp.alias, "waveform": sp.waveform, "snr_db": sp.snr_db,
            "seed": sp.seed, "split": "dev" if sp.seed % 2 == 0 else "test", "span": sp.v_top - sp.v_bottom,
            "clutter_db": sp.clutter_db, "mirror_db": sp.mirror_db, "clicks": sp.clicks,
            "tissue": sp.modality == "TDI"}
    variants = [Variant(**d) for d in variant_dicts]
    return run_variants_on_image(case.image, case.truth, variants, r_times=case.t_rpeaks, case_info=info)


def run_synthetic(specs, variants: list[Variant], workers: int = 8, progress=True) -> list[dict[str, Any]]:
    from dataclasses import asdict
    jobs = [(vars(s).copy(), [asdict(v) for v in variants]) for s in specs]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if workers <= 1:
        for i, j in enumerate(jobs):
            rows.extend(_synthetic_job(j))
            if progress and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} cases, {time.perf_counter() - t0:.0f} s", flush=True)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_synthetic_job, jobs, chunksize=2)):
            rows.extend(r)
            if progress and (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} cases, {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Real images with a vendor (scanner) envelope trace drawn on the display
# ---------------------------------------------------------------------------

VENDOR_TRACE_RGB = (98, 164, 255)


def vendor_trace(image, colour=VENDOR_TRACE_RGB, tol: int = 12):
    """Velocity (cm/s, raw column order) of the vendor trace drawn on the display.

    The trace pixels are chromatic, so the pipeline masks them as an overlay
    and every estimator sees the image with the trace removed.
    """
    region = spectral_region(image)
    sub = image.pixels[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    m = np.all(np.abs(sub.astype(int) - np.asarray(colour)) <= tol, axis=2)
    rows = np.arange(m.shape[0])[:, None]
    n = m.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (m * rows).sum(axis=0) / n
    r[n == 0] = np.nan
    return (r - region.ref_y0) * region.dy


def run_vendor(files, variants: list[Variant], base: PipelineConfig | None = None) -> list[dict[str, Any]]:
    """Score every variant against the vendor trace.

    The trace follows the dominant flow; per column it is compared with v_max
    where it is positive and with v_min where it is negative.
    """
    from ..io import load_dicom
    from .metrics import pointwise

    base = base or load_config()
    out = []
    for f in files:
        img = load_dicom(f)
        v_gt_raw = vendor_trace(img)
        rows = run_variants_on_image(img, None, variants, base=base, keep_series=True,
                                     case_info={"case": str(f)})
        spec = _render(img, base)
        o = spec.order
        v_gt = v_gt_raw[o]
        span = spec.display_v_limits[1] - spec.display_v_limits[0]
        for row in rows:
            ser = row.pop("_series", None)
            if ser is None:
                out.append(row)
                continue
            est = np.where(v_gt >= 0, ser["v_max"], ser["v_min"])
            m = pointwise(est, v_gt, span, abs(spec.dy))
            row.update({f"trace.{k}": v for k, v in m.items()})
            row["span"] = span
            row["_est"] = est
            row["_gt"] = v_gt
            out.append(row)
    return out
