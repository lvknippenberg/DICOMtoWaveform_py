"""One-at-a-time sensitivity of every tunable parameter (for docs/explainer, Section "Parameters").

    python scripts/parameter_sensitivity.py <out_dir> [--workers 12]

Each parameter is set to a lower and a higher value (usually half and double the default), all
others at their defaults, and scored on:
  - the synthetic suite (224 displays with known truth): median normalised envelope error
    (% of the display span), overall, at SNR <= 3 dB, on TDI and on the aliasing cases, and the
    mean-velocity error;
  - the 5 vascular images with the scanner's own envelope trace: mean absolute error (cm/s);
  - the 44 real cardiac images (no reference): how much the output changes against the defaults
    (mean |change| of v_max/v_min in cm/s and in % of the display span, number of R-peaks and
    aortic valve closures that change).
Writes sensitivity_summary.csv (aggregates only, no patient data) to <out_dir>. Real-image
locations are those of the development machine (see HANDOFF.md).
"""

from __future__ import annotations

import argparse
import copy
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.evaluation.benchmark import Variant, run_synthetic, run_vendor
from dicom2waveform.synthetic import benchmark_suite

warnings.filterwarnings("ignore")

VENDOR_DIR = Path(r"C:\Github\DICOMtoWaveform\Annotated examples")
CARDIAC_DIRS = [Path(r"C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples"),
                Path(r"C:\Github\DICOMtoWaveform\DICOM examples"), Path(r"C:\Github\DICOMtoWaveform\EMW examples")]

# (section, key, [lower, higher]); a key may also be a tuple of keys set together
SWEEP = [
    ("likelihood", "noise_margin", [16.0, 64.0]),
    ("likelihood", "noise_spread_factor", [2.0, 8.0]),
    ("likelihood", "soft_low_factor", [0.5, 3.0]),
    ("likelihood", ("density_window_px_rows", "density_window_px_cols"), [(5, 3), (15, 9)]),
    ("dp", "edge_density", [0.35, 0.65]),
    ("dp", "reference_accel_cm_s2", [320.0, 1280.0]),
    ("dp", "accel_scale_with_range", [False]),
    ("dp", "inner_band_rel", [0.025, 0.1]),
    ("dp", "height_prior", [0.002, 0.05]),
    ("dp", "wall_band_rel", [0.0, 0.06]),
    ("dp", "background_percentile", [0.0, 25.0, 75.0]),
    ("dp", "normalise_percentile", [0.0, 95.0, 99.9]),
    ("dp", "normalise_floor", [0.05, 0.2]),
    ("dp", "local_norm_s", [0.0, 0.02]),
    ("dp", "local_norm_floor", [0.075, 0.3]),
    ("aliasing", "alias_edge_fraction", [0.01, 0.04]),
    ("aliasing", "alias_min_fraction", [0.005, 0.02]),
    ("aliasing", "guard_rel", [0.05, 0.2]),
    ("aliasing", "cut_smooth_rows", [1, 9]),
    ("aliasing", "edge_preference", [0.0005, 0.008]),
    ("aliasing", "cut_lambda", [0.025, 0.1]),
    ("aliasing", "cut_cross_weight", [0.0, 0.5]),
    ("aliasing", "max_cost_ratio", [0.25, 0.75]),
    ("aliasing", "min_edge_density", [0.025, 0.1]),
    ("mean", "dynamic_range_db", [30.0, 50.0]),
    ("render", "click_min_snr", [8.0, 16.0]),
    ("ecg", "avc_min_snr", [4.0, 8.0]),
    ("ecg", "avc_window_s", [(0.15, 0.55), (0.25, 0.45)]),
    ("ecg", "qrs_band_hz", [(5.0, 30.0), (10.0, 30.0)]),
    ("ecg", "refractory_s", [0.2, 0.3]),
    ("ecg", "thresh_fraction", [0.25, 0.5]),
    ("ecg", "searchback_gap_ratio", [1.4, 2.0]),
]
ENVELOPE_SECTIONS = ("likelihood", "dp", "aliasing", "mean")


def _label(key, val) -> str:
    k = "+".join(key) if isinstance(key, tuple) else key
    return f"{k}={val}"


def overrides(section, key, val) -> dict:
    if isinstance(key, tuple):
        return {section: dict(zip(key, val))}
    return {section: {key: list(val) if isinstance(val, tuple) else val}}


def variants() -> list[tuple[str, str, str, dict]]:
    """(section, parameter, label, overrides), the default first."""
    out = [("default", "default", "default", {})]
    for section, key, vals in SWEEP:
        for v in vals:
            out.append((section, "+".join(key) if isinstance(key, tuple) else key, _label(key, v),
                        overrides(section, key, v)))
    return out


def _real_job(args):
    path, var = args
    rows = []
    for section, param, label, ov in var:
        try:
            r = extract_waveforms(path, copy.deepcopy(ov) or None)
            lo, hi = r.meta["velocity_limits_cm_s"]
            rows.append({"case": Path(path).name, "label": label, "v_max": r.v_max, "v_min": r.v_min,
                         "span": hi - lo, "n_r": len(r.t_rpeaks), "n_avc": len(r.t_avc), "t_avc": r.t_avc,
                         "modality": r.flags["modality"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"case": Path(path).name, "label": label, "error": str(e)})
    return rows


def real_changes(var, workers):
    files = [str(f) for d in CARDIAC_DIRS for f in sorted(d.iterdir()) if f.is_file()
             and f.suffix.lower() not in (".png", ".txt", ".pdf")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for rs in ex.map(_real_job, [(f, var) for f in files]) for r in rs]
    by_case: dict[str, dict] = {}
    for r in rows:
        by_case.setdefault(r["case"], {})[r["label"]] = r
    out = []
    for _, _, label, _ in var:
        d_cm, d_pct, d_r, d_avc, n = [], [], 0, 0, 0
        for case, rs in by_case.items():
            a, b = rs.get("default"), rs.get(label)
            if not a or not b or "error" in a or "error" in b:
                continue
            n += 1
            diff = np.r_[np.abs(b["v_max"] - a["v_max"]), np.abs(b["v_min"] - a["v_min"])]
            m = float(np.nanmean(diff)) if np.isfinite(diff).any() else 0.0
            d_cm.append(m)
            d_pct.append(100 * m / a["span"])
            d_r += int(b["n_r"] != a["n_r"])
            d_avc += int(b["n_avc"] != a["n_avc"] or (len(a["t_avc"]) and np.max(np.abs(b["t_avc"] - a["t_avc"])) > 0.003))
        out.append({"label": label, "real_n": n, "real_change_cm_s": float(np.mean(d_cm)) if d_cm else np.nan,
                    "real_change_pct": float(np.mean(d_pct)) if d_pct else np.nan,
                    "real_change_max_cm_s": float(np.max(d_cm)) if d_cm else np.nan,
                    "real_rpeak_files_changed": d_r, "real_avc_files_changed": d_avc})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    var = variants()
    env_var = [v for v in var if v[0] in ("default",) + ENVELOPE_SECTIONS]
    vlist = [Variant(label, "dp", overrides=ov, group="tuning") for _, _, label, ov in env_var]

    print("synthetic ...", flush=True)
    syn = pd.DataFrame(run_synthetic(benchmark_suite(), vlist, workers=a.workers, progress=False))
    syn["pct"] = 100 * syn["env_nmae"]
    g = syn.groupby("label" if "label" in syn else "variant")
    s = pd.DataFrame({
        "syn_median_pct": g["pct"].median(),
        "syn_lowsnr_pct": syn[syn.snr_db <= 3].groupby("variant")["pct"].median(),
        "syn_tdi_pct": syn[syn.modality == "TDI"].groupby("variant")["pct"].median(),
        "syn_alias_pct": syn[syn.alias].groupby("variant")["pct"].median(),
        "syn_fail_pct": g["pct"].apply(lambda x: 100 * float(np.mean(x > 5))),
        "syn_mean_pct": 100 * g["mean_nmae"].median(),
    }).reset_index().rename(columns={"variant": "label"})

    print("vendor ...", flush=True)
    files = sorted(f for f in VENDOR_DIR.iterdir() if f.is_file())
    ven = pd.DataFrame(run_vendor(files, vlist))
    v = ven.groupby("variant").agg(vendor_mae=("trace.mae", "mean"), vendor_bias=("trace.bias", "mean")) \
        .reset_index().rename(columns={"variant": "label"})

    print("real cardiac ...", flush=True)
    real = real_changes(var, a.workers)

    meta = pd.DataFrame([{"section": sec, "parameter": par, "label": lab} for sec, par, lab, _ in var])
    tab = meta.merge(s, on="label", how="left").merge(v, on="label", how="left").merge(real, on="label", how="left")
    tab.to_csv(out / "sensitivity_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(tab.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
