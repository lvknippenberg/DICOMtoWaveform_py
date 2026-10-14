"""Head-to-head on the EchoXFlow subset: EchoNet-Measurements vs dicom2waveform_standalone (DP), both reading the same
synthetic DICOM (echoxflow_to_dicom.py), scored against the clinicians' EchoPAC measurements.

Per recording and measurement:
  reference  = mean |v| of the clinician's marks of that measurement in the recording (they measure 1-3 beats);
               for hand-drawn traces, the trace peak;
  EchoNet    = its single automatic point (official folder-mode output), |v|;
  DP @ marks = mean over the marked beats of the DP peak within +-60 ms of the mark (for traces: over the trace span),
               on the mark's side of the baseline, ignoring valve-click columns  [timing given, value automatic];
  DP @ EchoNet = DP peak within +-150 ms of EchoNet's point  [fully automatic: EchoNet's beat, DP's value].
Beat-level: EchoNet 'hits' a marked beat when its point is within 0.15 s of a mark of that measurement.
Errors: |estimate| - reference (cm/s). Paired test: Wilcoxon signed-rank on |error| (DP @ marks vs EchoNet).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

sys.path.insert(0, r"C:\Github\DICOMtoWaveform_py\standalone")
import dicom2waveform_standalone as S  # noqa: E402

IN = Path(r"C:\Claude\Doppler\validation\echoxflow_dicom")
EN = Path(r"C:\Claude\Doppler\validation\echoxflow_echonet")
OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_out")
WIN = 0.06
BEAT = 0.15  # EchoNet's point localises the beat (its x is ~0.1 s coarse); DP takes the peak within +-BEAT
X0, Y0 = 42, 342
# task -> (EchoNet CSV name, [(measurement name, reference labels, EchoNet x/y columns)])
TASKS = {
    "tdi_lat": ("metadata_latevel.csv", [("e' lateral", ["e' lateral", "e'"], ("predicted_x", "predicted_y", 0))]),
    "tdi_med": ("metadata_medevel.csv", [("e' septal", ["e' septal", "e'"], ("predicted_x", "predicted_y", 0))]),
    "tr": ("metadata_trvmax.csv", [("TR Vmax", ["TR Vmax"], ("predicted_x", "predicted_y", 0))]),
    "av": ("metadata_avvmax.csv", [("AV Vmax", ["AV Vmax", "trace"], ("predicted_x", "predicted_y", 0))]),
    "lvot": ("metadata_lvotvmax.csv", [("LVOT Vmax (PW trace peak)", ["trace"], ("predicted_x", "predicted_y", 0))]),
    "mitral": ("metadata_mvpeak.csv", [("A", ["A (tool)", "A"], ("predicted_xa", "predicted_ya", Y0)),
                                       ("E", ["E"], ("predicted_xe", "predicted_ye", Y0))]),
}


def dp_read(res, t0, ta, tb, sign):
    """Extreme of the DP envelope between times ta..tb (recording clock), click columns excluded."""
    t = res.t + t0
    m = (t >= ta) & (t <= tb)
    spec = res.spectrogram
    m &= ~ndimage.binary_dilation(spec.clicks[spec.order], iterations=3)
    seg = (res.v_max if sign > 0 else res.v_min)[m]
    seg = seg[~np.isnan(seg)]
    return np.nan if seg.size == 0 else float(np.abs(seg).max())


rows = []
for task, (csv, measures) in TASKS.items():
    f = EN / task / csv
    if not f.exists():
        print("missing", f)
        continue
    en = pd.read_csv(f)
    en["recording"] = en["filename"].map(lambda p: Path(p).stem)
    en = en.set_index("recording")
    for side_f in sorted((IN / task).glob("*.json")):
        side = json.loads(side_f.read_text())
        rid = side["recording"]
        dcm = IN / task / f"{rid}.dcm"
        res = S.extract_waveforms(dcm)
        r = res.spectrogram.region
        t0, dt = side["t0"], side["dt"]
        for mname, labels, (xc, yc, yoff) in measures:
            refs = [x for x in side["refs"] if x["label"] in labels]
            if task == "av":  # CW traces in the AV folder: only when no explicit AV Vmax
                if any(x["label"] == "AV Vmax" for x in refs):
                    refs = [x for x in refs if x["label"] == "AV Vmax"]
            if not refs:
                continue
            sign = np.sign(np.median([x["v"] for x in refs]))
            ref = float(np.mean([abs(x["v"]) for x in refs]))
            dp_marks = []
            for x in refs:
                if x["label"] == "trace":
                    dp_marks.append(dp_read(res, t0, x["t_start"], x["t_end"], sign))
                else:
                    dp_marks.append(dp_read(res, t0, x["t"] - WIN, x["t"] + WIN, sign))
            row = dict(task=task, measure=mname, recording=rid, exam=side["exam"], n_marks=len(refs), ref=ref,
                       dp_marks=float(np.nanmean(dp_marks)) if np.isfinite(dp_marks).any() else np.nan)
            if rid in en.index:
                p = en.loc[rid]
                x_pix, y_pix = float(p[xc]), float(p[yc]) + yoff
                t_en = t0 + (x_pix - X0) * dt
                v_en = (y_pix - Y0 - r.ref_y0) * r.dy
                row.update(echonet=abs(v_en), echonet_side_ok=bool(np.sign(v_en) == sign), echonet_t=t_en,
                           echonet_in_region=bool(r.x0 <= x_pix <= r.x1),
                           hit=bool(min(abs(t_en - x["t"]) for x in refs) < 0.15),
                           dp_at_echonet=dp_read(res, t0, t_en - BEAT, t_en + BEAT, sign))
            else:
                row.update(echonet=np.nan, echonet_failed=True)
            rows.append(row)
    print(task, "done", flush=True)

R = pd.DataFrame(rows)
for c in ("echonet", "dp_marks", "dp_at_echonet"):
    R[f"err_{c}"] = R[c] - R["ref"]
R.to_csv(OUT / "echonet_vs_dp_echoxflow.csv", index=False)


def summary(g):
    both = g.dropna(subset=["err_echonet", "err_dp_marks"])
    out = {"recordings": len(g), "exams": g["exam"].nunique(), "ref mean": g["ref"].mean(),
           "EchoNet failed": int(g["echonet"].isna().sum())}
    for c, name in (("echonet", "EchoNet"), ("dp_marks", "DP@marks"), ("dp_at_echonet", "DP@EchoNet")):
        e = g[f"err_{c}"].dropna()
        out[f"{name} MAE"] = e.abs().mean()
        out[f"{name} bias"] = e.mean()
        out[f"{name} rel"] = (e.abs() / g.loc[e.index, "ref"]).median()
    if len(both) >= 6:
        d = both["err_dp_marks"].abs() - both["err_echonet"].abs()
        out["DP better (%)"] = 100 * (d < 0).mean()
        out["Wilcoxon p"] = stats.wilcoxon(both["err_dp_marks"].abs(), both["err_echonet"].abs()).pvalue
    out["EchoNet hits marked beat (%)"] = 100 * g["hit"].mean() if "hit" in g else np.nan
    out["EchoNet wrong side (%)"] = 100 * (~g["echonet_side_ok"].fillna(True).astype(bool)).mean()
    return pd.Series(out)


S1 = R.groupby("measure", sort=False).apply(summary)
pd.set_option("display.width", 250)
print(S1.round(2).T.to_string())
S1.round(3).to_csv(OUT / "echonet_vs_dp_echoxflow_summary.csv")
hits = R[R["hit"] == True]  # noqa: E712
print("\nRecordings where EchoNet chose a marked beat: n =", len(hits))
print(hits.groupby("measure").apply(lambda g: pd.Series({
    "n": len(g), "EchoNet MAE": g["err_echonet"].abs().mean(), "DP@EchoNet MAE": g["err_dp_at_echonet"].abs().mean(),
    "DP@marks MAE": g["err_dp_marks"].abs().mean()})).round(2).to_string())
