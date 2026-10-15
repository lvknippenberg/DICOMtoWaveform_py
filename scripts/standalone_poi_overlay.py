"""Standalone overlays with the user's manual points of interest drawn on top.

usage: python scripts/standalone_poi_overlay.py standalone/dicom2waveform_standalone.py <dicom_dir> <poi_dir> <out_dir>

Outputs contain patient details: write them outside the repository and keep them within the institution.

Marks (see Readme.txt in the poi folder): dot = a velocity to measure (time and magnitude), vertical
line = a timing (start / end of a wave). Mark pixels = blue pixels where the annotated PNG differs from
the DICOM (as in scripts/external_validation/poi_marks.py). For every dot the envelope on the same side
is read at the dot's time, and the difference is listed in poi_marks.csv.
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
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

warnings.filterwarnings("ignore")
script, src, poi, out = (Path(a) for a in sys.argv[1:5])
spec = importlib.util.spec_from_file_location("dicom2waveform_standalone", script)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)
out.mkdir(parents=True, exist_ok=True)

MARK = "#30d158"  # green: not used by the other overlay signals


def read_marks(img, png: Path, res):
    d = img.pixels if img.pixels.ndim == 3 else np.repeat(img.pixels[..., None], 3, 2)
    a = np.asarray(Image.open(png).convert("RGB")).astype(int)
    if a.shape != d.shape:
        raise ValueError(f"annotated PNG {a.shape} != DICOM {d.shape}")
    changed = np.abs(a - d.astype(int)).max(axis=2) > 30
    blue = (a[..., 2] > 150) & (a[..., 2] - a[..., 0] > 60) & (a[..., 2] - a[..., 1] > 20)
    lab, n = ndimage.label(ndimage.binary_closing(changed & blue, iterations=1))
    sp = res.diagnostics["spectrogram"]
    region = sp.region
    inv = np.argsort(sp.order)  # raw column -> time index
    dots, lines = [], []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = lab[sl] == i + 1
        if comp.sum() < 6:
            continue
        h, w = comp.shape
        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
        col = int(round(cx)) - region.x0
        if not (0 <= col < region.width):
            continue
        k = int(inv[col])
        t = float(res.t[k])
        if h > 3 * w and h > 12:  # vertical line: timing mark
            lines.append(dict(x=cx, t=t, y0=sl[0].start, y1=sl[0].stop - 1))
        else:
            v = float((cy - region.y0 - region.ref_y0) * region.dy)
            env = res.v_max[k] if v >= 0 else res.v_min[k]
            dots.append(dict(x=cx, y=cy, t=t, v=v, env=float(env)))
    return dots, lines


rows = []
for f in sorted(p for p in src.iterdir() if p.is_file()):
    name = f.name
    png = poi / f"{name}.png"
    img = S.load_dicom(f)
    res = S.extract_waveforms(img, {"keep_diagnostics": True})
    dots, lines = read_marks(img, png, res) if png.exists() else ([], [])
    fig = S.plot_overlay(res, img, None, title=name)
    axes = fig.axes  # [full frame, normalised strip] (same pixel coordinates)
    stroke = [patheffects.Stroke(linewidth=4.0, foreground="black"), patheffects.Normal()]
    region = res.diagnostics["spectrogram"].region
    trace = res.diagnostics["ecg_trace"]
    ecg_top = np.nanmin(trace.row) - 10 if trace.found and np.isfinite(trace.row).any() else region.y0
    label_fx = [patheffects.withStroke(linewidth=4, foreground="black")]
    for ax in axes:
        prev_x, level = -1e9, 0
        for ln in sorted(lines, key=lambda d: d["x"]):
            # a timing mark is a moment: draw it across the ECG and the whole spectrum
            top, bottom = min(ln["y0"], ecg_top), max(ln["y1"], region.y1)
            ax.plot([ln["x"], ln["x"]], [top, bottom], color=MARK, lw=2.6, ls=(0, (5, 2)),
                    path_effects=[patheffects.Stroke(linewidth=5.0, foreground="black"), patheffects.Normal()])
            level = level + 1 if ln["x"] - prev_x < 70 else 0  # stagger labels of close marks
            prev_x = ln["x"]
            ax.text(ln["x"] + 6, top + 4 + 16 * level, f"{ln['t']:.3f} s", color=MARK, fontsize=13, va="top",
                    fontweight="bold", path_effects=label_fx)
        for dt in dots:
            ax.plot([dt["x"]], [dt["y"]], marker="o", ms=20, mfc="none", mec="black", mew=5.0)
            ax.plot([dt["x"]], [dt["y"]], marker="o", ms=20, mfc="none", mec=MARK, mew=2.8)
            diff = abs(dt["env"]) - abs(dt["v"])
            ax.text(dt["x"] + 13, dt["y"], f"{dt['v']:.1f} ({diff:+.1f})", color=MARK, fontsize=14, va="center",
                    fontweight="bold", path_effects=label_fx)
    leg = axes[0].get_legend()
    handles, labels = list(leg.legend_handles), [t.get_text() for t in leg.texts]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=MARK, mew=2, ms=9),
                Line2D([], [], color=MARK, ls=(0, (4, 2)), lw=2)]
    labels += ["manual velocity mark: cm/s (|envelope| - |mark|)", "manual timing mark (time in s)"]
    axes[0].legend(handles, labels, loc="lower left", fontsize=9, framealpha=0.7, facecolor="black",
                   labelcolor="white", ncol=len(handles))
    fig.savefig(out / f"{name}_overlay_poi.png", dpi=100)
    plt.close(fig)
    for dt in dots:
        rows.append({"file": name, "modality": res.flags["modality"], "kind": "velocity", "t_s": round(dt["t"], 4),
                     "v_mark_cm_s": round(dt["v"], 1), "v_envelope_cm_s": round(dt["env"], 1),
                     "envelope_minus_mark_cm_s": round(abs(dt["env"]) - abs(dt["v"]), 1)})
    for ln in lines:
        rows.append({"file": name, "modality": res.flags["modality"], "kind": "timing", "t_s": round(ln["t"], 4)})
    print(f"{name}: {res.flags['modality']}, {len(dots)} velocity marks, {len(lines)} timing marks", flush=True)

keys = ["file", "modality", "kind", "t_s", "v_mark_cm_s", "v_envelope_cm_s", "envelope_minus_mark_cm_s"]
with open(out / "poi_marks.csv", "w", newline="", encoding="utf8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} marks -> {out}")
