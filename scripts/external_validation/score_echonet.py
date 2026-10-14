"""Score EchoNet-Measurements and our DP envelope against the user's manual marks (blue dots).

References: dots from poi_marks.read_marks, labelled by timing relative to the R-peaks:
  TDI: a' = dot <= 0.30 s before the next R-peak, e' otherwise (dots with |v| < 1.5 cm/s are onset marks, skipped)
  PW mitral: A likewise, E otherwise; CW TR: every dot is a jet peak.
EchoNet: the official folder-mode CSVs (one point per image; two for E/A). Its point is converted to time with the
  standalone time axis and to signed velocity with the DICOM calibration.
DP (ours): extreme of v_max (positive waves) or v_min (negative waves) within +-WIN of a time:
  - at the manual dot (DP @ mark: timing given, value automatic),
  - at EchoNet's point (DP @ EchoNet: fully automatic, same beat as EchoNet).
Errors are in |velocity| (cm/s): positive = reads higher than the reference.
Also: agreement EchoNet vs DP on the 26 CW recordings of the AVC sets (presumed aortic outflow; no manual reference).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import pandas as pd

sys.path.insert(0, r"C:\Claude\Doppler\validation")
sys.path.insert(0, r"C:\Github\DICOMtoWaveform_py\standalone")
import dicom2waveform_standalone as S  # noqa: E402
from poi_marks import DICOM_DIR, read_marks  # noqa: E402

OUT = Path(r"C:\Claude\Doppler\validation\echonet_out")
WIN = 0.06
OLD = [Path(r"C:\Github\DICOMtoWaveform\DICOM examples"), Path(r"C:\Github\DICOMtoWaveform\EMW examples")]


def find_dicom(name):
    for d in [DICOM_DIR, *OLD]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def to_time(res, x_full):
    region = res.spectrogram.region
    col = int(np.clip(round(x_full) - region.x0, 0, region.width - 1))
    return float(res.t[np.argsort(res.spectrogram.order)[col]])


def to_velocity(res, y_full):
    r = res.spectrogram.region
    return float((y_full - r.y0 - r.ref_y0) * r.dy)


def dp_peak(res, t0, sign):
    """Extreme of the DP envelope within +-WIN of t0, ignoring detected valve-click columns (widened by 3 columns):
    the envelope traces clicks in the default 'keep' mode; a peak reading must not."""
    m = np.abs(res.t - t0) <= WIN
    spec = res.spectrogram
    if spec.clicks is not None:
        clicks = ndimage.binary_dilation(spec.clicks[spec.order], iterations=3)
        m &= ~clicks
    v = res.v_max if sign > 0 else res.v_min
    seg = v[m]
    seg = seg[~np.isnan(seg)]
    if seg.size == 0:
        return np.nan
    return float(np.max(seg) if sign > 0 else np.min(seg))


AF_FILES = {"03_10673_"}  # matched by substring  # atrial fibrillation: e' only (user's Readme)
HEAT = OUT / "heatmaps"


def label_dots(dots, t_r, kind, name=""):
    out = []
    for d in dots:
        if abs(d["v"]) < 1.5:
            continue
        nxt = t_r[t_r > d["t"]]
        dt_next = float(nxt[0] - d["t"]) if nxt.size else np.inf
        if kind == "tr":
            lab = "jet"
        elif kind == "tdi":
            lab = "e'" if any(k in name for k in AF_FILES) else ("a'" if dt_next <= 0.30 else "e'")
        else:
            lab = "A" if dt_next <= 0.30 else "E"
        out.append(dict(d, label=lab, dt_next_r=dt_next))
    return out


def echonet_at(res, name, model, t0, sign, channel=0):
    """EchoNet heatmap maximum within +-WIN of t0, on the side of the baseline given by sign, inside the display
    region -> signed velocity (cm/s). NaN if the response there is < 20% of the image maximum (no detection)."""
    z = np.load(HEAT / f"{name}__{model}.npz")
    heat, crop0 = z["heat"].astype(np.float32)[channel], int(z["crop0"])
    region = res.spectrogram.region
    idx = np.flatnonzero(np.abs(res.t - t0) <= WIN)
    cols = res.spectrogram.order[idx] + region.x0
    rows_full = np.arange(heat.shape[0]) + crop0
    v_rows = (rows_full - region.y0 - region.ref_y0) * region.dy
    ok = (np.sign(v_rows) == sign) & (rows_full >= region.y0) & (rows_full <= region.y1)
    sub = np.where(ok[:, None], heat[:, cols], 0.0)
    if sub.max() < 0.2 * heat.max():
        return np.nan
    y, _ = np.unravel_index(np.argmax(sub), sub.shape)
    return to_velocity(res, y + crop0)


rows = []
cache = {}


def result(name):
    if name not in cache:
        cache[name] = S.extract_waveforms(find_dicom(name))
    return cache[name]


# --- single-point models: TDI e' (lateral & septal models), TR Vmax (+AV/MR models as sensitivity)
for task, model, kind, target in [("tdi", "latevel", "tdi", "e'"), ("tdi", "medevel", "tdi", "e'"),
                                  ("tr", "trvmax", "tr", "jet"), ("tr", "avvmax", "tr", "jet"),
                                  ("tr", "mrvmax", "tr", "jet")]:
    df = pd.read_csv(OUT / f"{task}_{model}" / f"metadata_{model}.csv")
    for _, p in df.iterrows():
        name = Path(p["filename"]).stem
        res = result(name)
        dots, _, _ = read_marks(name, res)
        lab = [d for d in label_dots(dots, res.t_rpeaks, kind, name) if d["label"] == target]
        if not lab:
            continue
        t_en = to_time(res, p["predicted_x"])
        v_en = to_velocity(res, p["predicted_y"])
        has_heat = (HEAT / f"{name}__{model}.npz").exists()
        for d in lab:
            sign = np.sign(d["v"])
            rows.append(dict(file=name, model=model, measure=target, ref_t=d["t"], ref_v=d["v"],
                             same_beat_as_echonet=abs(d["t"] - t_en) < 0.15,
                             echonet_t=t_en, echonet_v=v_en,
                             echonet_at_mark=echonet_at(res, name, model, d["t"], sign) if has_heat else np.nan,
                             dp_at_mark=dp_peak(res, d["t"], sign), dp_at_echonet=dp_peak(res, t_en, sign)))

# --- mitral E/A
df = pd.read_csv(OUT / "mitral_mvpeak" / "metadata_mvpeak.csv")
for _, p in df.iterrows():
    name = Path(p["filename"]).stem
    res = result(name)
    dots, _, _ = read_marks(name, res)
    lab = label_dots(dots, res.t_rpeaks, "mitral", name)
    for target, xk, yk, ch in (("E", "predicted_xe", "predicted_ye", 1), ("A", "predicted_xa", "predicted_ya", 0)):
        # EchoNet points are in crop coordinates starting at row 342 (as in their script)
        t_en = to_time(res, p[xk])
        v_en = to_velocity(res, p[yk] + 342)
        for d in [d for d in lab if d["label"] == target]:
            rows.append(dict(file=name, model="mvpeak", measure=target, ref_t=d["t"], ref_v=d["v"],
                             same_beat_as_echonet=abs(d["t"] - t_en) < 0.15, echonet_t=t_en, echonet_v=v_en,
                             echonet_at_mark=echonet_at(res, name, "mvpeak_2c", d["t"], 1, ch),
                             dp_at_mark=dp_peak(res, d["t"], 1), dp_at_echonet=dp_peak(res, t_en, 1)))

R = pd.DataFrame(rows)
R["err_echonet_at_mark"] = R["echonet_at_mark"].abs() - R["ref_v"].abs()
for c in ("echonet", "dp_at_mark", "dp_at_echonet"):
    col = "echonet_v" if c == "echonet" else c
    R[f"err_{c}"] = R[col].abs() - R["ref_v"].abs()
R.to_csv(OUT / "scores_vs_marks.csv", index=False)
pd.set_option("display.width", 220)
print(R[["file", "model", "measure", "ref_t", "ref_v", "echonet_t", "echonet_v", "echonet_at_mark", "dp_at_mark",
         "dp_at_echonet", "err_echonet", "err_echonet_at_mark", "err_dp_at_mark", "same_beat_as_echonet"]]
      .round(2).to_string())


def summ(g):
    return pd.Series({
        "n": len(g),
        "EchoNet MAE": g["err_echonet"].abs().mean(), "EchoNet bias": g["err_echonet"].mean(),
        "EchoNet@mark MAE": g["err_echonet_at_mark"].abs().mean(),
        "EchoNet@mark bias": g["err_echonet_at_mark"].mean(),
        "DP@mark MAE": g["err_dp_at_mark"].abs().mean(), "DP@mark bias": g["err_dp_at_mark"].mean(),
        "DP@EchoNet MAE": g["err_dp_at_echonet"].abs().mean(), "DP@EchoNet bias": g["err_dp_at_echonet"].mean(),
    })


print("\nSummary (cm/s; error = |estimate| - |manual mark|):")
S1 = R.groupby(["measure", "model"]).apply(summ).round(2)
print(S1.to_string())
S1.to_csv(OUT / "summary_vs_marks.csv")

# --- agreement on CW of the AVC sets (presumed aortic outflow)
df = pd.read_csv(OUT / "aortic_cw_avvmax" / "metadata_avvmax.csv")
agg = []
for _, p in df.iterrows():
    name = Path(p["filename"]).stem
    res = result(name)
    t_en, v_en = to_time(res, p["predicted_x"]), to_velocity(res, p["predicted_y"])
    sign = np.sign(v_en) if v_en != 0 else -1
    agg.append(dict(file=name, echonet_v=v_en, dp_at_echonet=dp_peak(res, t_en, sign),
                    dp_max_beat=float(np.nanmin(res.v_min)) if sign < 0 else float(np.nanmax(res.v_max))))
A = pd.DataFrame(agg)
# the EMW set and the new examples share some recordings: drop pixel-identical duplicates
import hashlib  # noqa: E402

A["pixel_md5"] = [hashlib.md5(S.load_dicom(find_dicom(f)).pixels.tobytes()).hexdigest() for f in A["file"]]
dups = A[A.duplicated("pixel_md5")]["file"].tolist()
print("duplicates removed:", dups)
A = A.drop_duplicates("pixel_md5").drop(columns="pixel_md5")
A["diff"] = A["dp_at_echonet"].abs() - A["echonet_v"].abs()
A.to_csv(OUT / "agreement_aortic_cw.csv", index=False)
print("\nAortic CW (AVC sets), EchoNet AV Vmax vs DP at the same time, n =", len(A))
print(A.round(1).to_string())
Ad = A.dropna(subset=["diff"])
d = Ad["diff"]
print(f"DP - EchoNet (n={len(Ad)}): mean {d.mean():.1f}, SD {d.std():.1f}, median |diff| {d.abs().median():.1f} cm/s,"
      f" within 10%: {(d.abs() <= 0.1 * Ad['echonet_v'].abs()).mean():.0%},"
      f" r = {np.corrcoef(Ad['echonet_v'].abs(), Ad['dp_at_echonet'].abs())[0, 1]:.3f}")
