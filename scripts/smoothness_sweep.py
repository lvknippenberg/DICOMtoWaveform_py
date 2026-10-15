"""Smoothness sweep with the standalone script: the same DICOM traced at several reference accelerations.

usage: python scripts/smoothness_sweep.py standalone/dicom2waveform_standalone.py <dicom_dir> <out_dir>

The images show the spectral strip of patient images: write them outside the repository.
Accuracy against references with a known answer: scripts/smoothness_accuracy.py.

The smoothness prior is dp.reference_accel_cm_s2 (a0): moving the edge at this acceleration costs as
much as about two misclassified pixels; LOWER a0 = SMOOTHER envelope. Per DICOM one PNG with one row per
a0 (brightness-normalised spectrum strip with v_max / v_min), and roughness.csv with, per file and a0,
the total variation of the envelopes per second and the mean change against the default.
"""
import csv
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402

warnings.filterwarnings("ignore")
script, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec_ = importlib.util.spec_from_file_location("dicom2waveform_standalone", script)
S = importlib.util.module_from_spec(spec_)
sys.modules[spec_.name] = S
spec_.loader.exec_module(S)
out.mkdir(parents=True, exist_ok=True)

A0 = [40.0, 80.0, 160.0, 320.0, 640.0, 1280.0]  # 640 = default
DEFAULT = 640.0
OUTLINE = [patheffects.Stroke(linewidth=3.4, foreground="black"), patheffects.Normal()]


def tv_per_s(v, dt):
    d = np.abs(np.diff(v))
    return float(np.nansum(d) / (np.sum(~np.isnan(d)) * dt)) if np.any(~np.isnan(d)) else np.nan


rows = []
for f in sorted(p for p in src.iterdir() if p.is_file()):
    name = f.name
    img = S.load_dicom(f)
    results = {a: S.extract_waveforms(img, {"keep_diagnostics": True, "dp": {"reference_accel_cm_s2": a}}) for a in A0}
    fig, axes = plt.subplots(len(A0), 1, figsize=(16, 2.35 * len(A0)), sharex=True, facecolor="white")
    base = results[DEFAULT]
    for ax, a in zip(axes, A0):
        res = results[a]
        sp = res.diagnostics["spectrogram"]
        rgb = S.normalise_brightness(sp)[:, sp.order]
        t = res.t
        v_top = (0 - sp.zero_row) * sp.dy
        v_bot = (rgb.shape[0] - 1 - sp.zero_row) * sp.dy
        ax.imshow(rgb, aspect="auto", extent=(t[0], t[-1] + sp.dt, v_bot, v_top), origin="upper",
                  interpolation="nearest")
        if v_top < v_bot:
            ax.invert_yaxis()
        ax.plot(t, res.v_max, color="#ff3b30", lw=1.6, path_effects=OUTLINE)
        ax.plot(t, res.v_min, color="#0a84ff", lw=1.6, path_effects=OUTLINE)
        lo, hi = sp.display_v_limits
        a_ref = a * max(abs(lo), abs(hi)) / 100.0
        lam = (abs(sp.dy) / sp.dt) / a_ref
        tv = np.nanmean([tv_per_s(res.v_max, sp.dt), tv_per_s(res.v_min, sp.dt)])
        diff = np.nanmean(np.r_[np.abs(res.v_max - base.v_max), np.abs(res.v_min - base.v_min)])
        tag = "  (default)" if a == DEFAULT else ""
        ax.set_title(f"a0 = {a:g} cm/s$^2${tag}   [lambda = {lam:.2f} per row; roughness {tv:.0f} cm/s per s; "
                     f"mean change vs default {diff:.1f} cm/s]", loc="left", fontsize=10)
        ax.set_ylabel("cm/s")
        rows.append({"file": name, "modality": res.flags["modality"], "a0_cm_s2": a, "lambda_per_row": round(lam, 3),
                     "roughness_cm_s_per_s": round(tv, 1), "mean_change_vs_default_cm_s": round(float(diff), 2),
                     "v_max_peak": round(float(np.nanmax(res.v_max)), 1),
                     "v_min_peak": round(float(np.nanmin(res.v_min)), 1)})
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{name} ({base.flags['modality']}): smoothness sweep, lower a0 = smoother envelope "
                 f"(red v_max, blue v_min)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / f"{name}_smoothness.png", dpi=90)
    plt.close(fig)
    print(name, flush=True)

with open(out / "roughness.csv", "w", newline="", encoding="utf8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
