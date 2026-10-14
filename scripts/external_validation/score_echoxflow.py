"""Score the DP envelope (dicom2waveform_standalone) on EchoXFlow spectral Doppler against clinician annotations.

Compatibility: EchoXFlow stores the spectral display as a (time x velocity-bin) matrix of display values (4..255)
with a calibrated velocity axis, not as a rendered DICOM screen. The standalone front end (header calibration,
overlays, sweep, ECG trace) is therefore bypassed, and the matrix is rendered to the geometry of the displays the
method was developed on (Philips, 1024x768): RENDER_ROWS rows over the velocity span and RENDER_DT s per column.
The envelope stage then runs unchanged (noise statistics, density, DP, aliasing test/unwrap, valve clicks, means)
with the default configuration. --native skips the re-rendering (sensitivity analysis).

References (GE Vivid E95, EchoPAC annotations, clinicians):
  traces: 'Manual' (hand-drawn single-beat velocity curves, PW and CW) -> envelope error along the trace;
  points: peak-velocity marks (e', s', TR/AV Vmax, E, A, pulmonary vein S/D, and the peak of the 'A Velocity' and
          'PV Acc Time' tools) -> DP peak within +-60 ms on the same side, ignoring valve-click columns.
Errors in cm/s: positive = DP reads higher |velocity| than the clinician.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from echoxflow import load_croissant, open_recording
from scipy import ndimage

sys.path.insert(0, r"C:\Github\DICOMtoWaveform_py\standalone")
import dicom2waveform_standalone as S  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Claude\Doppler\external\echoxflow\data")
OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_out")
OUT.mkdir(exist_ok=True)
NATIVE = "--native" in sys.argv
# Display mapping (scanner reject/gain emulation): EchoXFlow stores the spectral matrix before the display grey map,
# with a bright noise background. Black point = this percentile of the image (the background dominates any Doppler
# display), white point = 99.5th percentile. Fixed a priori at 70; 60/80 only as sensitivity (--black N);
# --black 0 disables the mapping.
BLACK_PCT = float(sys.argv[sys.argv.index("--black") + 1]) if "--black" in sys.argv else 70.0
FIX_OFFSET = "--no-offset-fix" not in sys.argv  # see annotation_offset_mps
RENDER_ROWS = 390  # Philips spectral display region height (rows 342..732)
RENDER_DT = 0.0035  # s per display column (Philips sweep: 2.6-3.5 ms)
WIN = 0.06
POINT_LABELS = {"MV Eprime Lateral Velocity": "e' lateral", "MV Eprime Septal Velocity": "e' septal",
                "MV Eprime Velocity": "e'", "RV Sprime Velocity": "RV s'", "TR Vmax": "TR Vmax",
                "AV Vmax": "AV Vmax", "MV E Velocity": "E", "MV A Velocity": "A", "P_Vein S": "PV S",
                "P_Vein D": "PV D", "A Velocity": "A (tool)", "PV Acc Time": "PV peak (tool)"}
CFG = S.Config()
if "--no-alias" in sys.argv:
    CFG.aliasing.correct = False


def render(level_tf: np.ndarray, t: np.ndarray, axis_mps: np.ndarray):
    """(time x bins) display matrix -> (rows x cols) image with row 0 = highest velocity, as on a screen.
    Returns image, dy (cm/s per row, downward), zero_row, dt, column times."""
    img = level_tf.T.astype(np.float32)  # rows = velocity bins, row 0 = +Nyquist
    if BLACK_PCT > 0:
        lo, hi = np.percentile(img, BLACK_PCT), np.percentile(img, 99.5)
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
    v = axis_mps * 100.0
    if v[0] < v[-1]:  # make row 0 the most positive velocity
        img, v = img[::-1], v[::-1]
    if NATIVE:
        dt = float(np.median(np.diff(t)))
        return img, float(v[1] - v[0]), float(np.interp(0.0, -v, np.arange(len(v)))), dt, t
    tc = np.arange(t[0], t[-1], RENDER_DT)
    edges = np.searchsorted(t, np.r_[tc - RENDER_DT / 2, tc[-1] + RENDER_DT / 2])
    cols = np.stack([img[:, max(a, 0):max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    vr = np.linspace(v[0], v[-1], RENDER_ROWS)
    rows = np.stack([np.interp(-vr, -v, cols[:, j]) for j in range(cols.shape[1])], axis=1)
    return rows.astype(np.float32), float(vr[1] - vr[0]), float(np.interp(0.0, -vr, np.arange(RENDER_ROWS))), RENDER_DT, tc


def envelope(img, dy, zero_row, dt, data_type):
    """The envelope stage of extract_waveforms() on an already clean, calibrated spectral image."""
    H, W = img.shape
    region = S.Region(x0=0, y0=0, x1=W - 1, y1=H - 1, dx=dt, dy=dy, units_x=4, units_y=7, data_type=data_type,
                      ref_y0=int(round(zero_row)))
    span = max(abs(zero_row * dy), abs((H - 1 - zero_row) * dy))
    modality = "CW" if data_type == 4 else ("TDI" if span < 40.0 else "PW")
    sweep = np.zeros(W, bool)
    clicks, strength = S.find_clicks(img, sweep, dt, CFG.render)
    spec = S.Spectrogram(level=img, valid=np.ones_like(img, bool), zero_row=float(zero_row), dy=dy, dt=dt,
                         sweep=sweep, order=np.arange(W), colormap="grey", modality=modality, region=region,
                         baseline_rows=np.array([], int), info={"display_zero_row": float(zero_row)},
                         clicks=clicks, click_strength=strength, click_mode=CFG.render.click_mode)
    ns = S.noise_stats(spec, CFG.likelihood)
    env = S.dp_envelope(spec, ns, CFG.likelihood, CFG.dp)
    alias = False
    if modality != "CW" and S.aliasing_suspected(spec, env["k_pos"], env["k_neg"], CFG.aliasing):
        spec_u, info = S.unwrap_aliasing(spec, env["density"], CFG.aliasing)
        if info["corrected"]:
            alias = True
            spec = spec_u
            ns = S.noise_stats(spec, CFG.likelihood)
            env = S.dp_envelope(spec, ns, CFG.likelihood, CFG.dp)
    s = abs(spec.dy)
    return dict(v_max=env["k_pos"] * s, v_min=-env["k_neg"] * s, clicks=clicks, modality=modality, alias=alias)


def annotation_offset_mps(sm) -> float:
    """Spectral annotations are stored relative to the display baseline (baseline_frac), the matrix axis uses its own
    baseline (spectral_row_baseline_frac). Verified on the CW 'Manual' traces: their onset/end points sit exactly at
    -(row_frac - display_frac) * 2 * Nyquist (25/27 to 3 decimals); for PW both fractions are equal (offset 0).
    Returns the value to ADD to annotation velocities (m/s)."""
    raw = sm.raw or {}
    bf, sbf = raw.get("baseline_frac"), raw.get("spectral_row_baseline_frac")
    if bf is None or sbf is None or sm.nyquist_limit_mps is None:
        return 0.0
    return float((sbf - bf) * 2.0 * sm.nyquist_limit_mps)


def dp_peak(t, env, t0, sign):
    m = (np.abs(t - t0) <= WIN) & ~ndimage.binary_dilation(env["clicks"], iterations=3)
    seg = (env["v_max"] if sign > 0 else env["v_min"])[m]
    seg = seg[~np.isnan(seg)]
    return np.nan if seg.size == 0 else float(seg.max() if sign > 0 else seg.min())


ann = pd.read_csv(ROOT.parent / "echoxflow_annotations.csv")
ann["short"] = ann["label"].fillna("").str.split("/").str[-1]
have = {p.name for p in (ROOT / "exams").iterdir()}
ann = ann[ann["exam"].isin(have)]
cat = load_croissant(ROOT / "croissant.json", root=ROOT)
recs = {r.recording_id: r for r in cat.recordings}
def main():
    trace_rows, point_rows = [], []
    for (rid, kind), g in ann.groupby(["recording", "kind"]):
        use = g[(g["short"] == "Manual") | g["short"].isin(POINT_LABELS)]
        if use.empty:
            continue
        st = open_recording(recs[rid], root=ROOT)
        name = kind.removeprefix("data/")
        lv = st.load_array(f"data/{name}").astype(np.float32)
        t = st.load_array(f"timestamps/{name}").astype(float)
        sm = st.spectral_metadata(name)
        img, dy, z, dt, tc = render(lv, t, np.asarray(sm.row_velocity_mps, float))
        env = envelope(img, dy, z, dt, 4 if "continuous" in kind else 3)
        off = annotation_offset_mps(sm) if FIX_OFFSET else 0.0
        for _, a in use.iterrows():
            pts = st.load_array(a["path"]).astype(float)
            pts[:, 1] += off
            if a["short"] == "Manual":
                pts = pts[np.argsort(pts[:, 0])]
                if np.max(np.abs(pts[:, 1])) < 0.10:
                    continue  # degenerate trace (no excursion from the baseline after correction)
                m = (tc >= pts[0, 0]) & (tc <= pts[-1, 0])
                ref = np.interp(tc[m], pts[:, 0], pts[:, 1] * 100.0)
                sign = -1 if np.median(ref) < 0 else 1
                est = (env["v_max"] if sign > 0 else env["v_min"])[m]
                ok = ~np.isnan(est) & ~ndimage.binary_dilation(env["clicks"], iterations=3)[m]
                err = np.abs(est[ok]) - np.abs(ref[ok])
                peak_ref = float(np.max(np.abs(ref)))
                peak_est = float(np.max(np.abs(est[ok]))) if ok.any() else np.nan
                tm = tc[m]
                t_pk_ref = float(tm[np.argmax(np.abs(ref))])
                t_pk_est = float(tm[ok][np.argmax(np.abs(est[ok]))]) if ok.any() else np.nan
                trace_rows.append(dict(exam=a["exam"], recording=rid, kind=env["modality"], n=int(ok.sum()),
                                       peak_t_err_ms=1000 * (t_pk_est - t_pk_ref), duration_ms=1000 * (tm[-1] - tm[0]),
                                       mae=float(np.mean(np.abs(err))), bias=float(np.mean(err)),
                                       rel_mae=float(np.mean(np.abs(err)) / peak_ref), peak_ref=peak_ref,
                                       peak_est=peak_est, nyquist_cm_s=float(a["nyquist"]) * 100, alias=env["alias"],
                                       corr=float(np.corrcoef(np.abs(est[ok]), np.abs(ref[ok]))[0, 1]) if ok.sum() > 3 else np.nan))
            else:
                k = int(np.argmax(np.abs(pts[:, 1])))  # the peak point (tools have onset + peak points)
                t0, v0 = pts[k, 0], pts[k, 1] * 100.0
                est = dp_peak(tc, env, t0, np.sign(v0))
                point_rows.append(dict(exam=a["exam"], recording=rid, kind=env["modality"], label=POINT_LABELS[a["short"]],
                                       ref=v0, dp=est, err=abs(est) - abs(v0), nyquist_cm_s=float(a["nyquist"]) * 100,
                                       alias=env["alias"]))
    T, P = pd.DataFrame(trace_rows), pd.DataFrame(point_rows)
    tag = ("native" if NATIVE else "rendered") + f"_black{int(BLACK_PCT)}" + ("" if FIX_OFFSET else "_nooffsetfix") + ("_noalias" if not CFG.aliasing.correct else "")
    T.to_csv(OUT / f"traces_{tag}.csv", index=False)
    P.to_csv(OUT / f"points_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"[{tag}] exams {pd.concat([T['exam'], P['exam']]).nunique()}, recordings "
          f"{pd.concat([T['recording'], P['recording']]).nunique()}")
    print("\nManual traces (envelope vs hand-drawn curve, cm/s):")
    T["peak_err"] = T["peak_est"] - T["peak_ref"]
    T["peak_ratio"] = T["peak_ref"] / T["peak_est"]
    print(T.groupby("kind").agg(traces=("mae", "size"), samples=("n", "sum"), MAE_median=("mae", "median"),
                                MAE_mean=("mae", "mean"), bias_mean=("bias", "mean"), rel_MAE_median=("rel_mae", "median"),
                                r_median=("corr", "median"), peak_err_median=("peak_err", "median"),
                                peak_t_err_ms_median=("peak_t_err_ms", lambda x: x.abs().median()),
                                ratio_median=("peak_ratio", "median"),
                                ratio_cv=("peak_ratio", lambda x: x.std() / x.mean())).round(3).to_string())
    print("\nPeak marks (DP peak at the marked beat vs clinician, cm/s):")
    P["rel"] = P["err"].abs() / P["ref"].abs()
    print(P.groupby(["kind", "label"]).agg(n=("err", "size"), ref_mean=("ref", lambda x: x.abs().mean()),
                                             MAE=("err", lambda x: x.abs().mean()), bias=("err", "mean"),
                                             rel_MAE_median=("rel", "median")).round(2).to_string())
    print(P.groupby("kind").agg(n=("err", "size"), MAE=("err", lambda x: x.abs().mean()), bias=("err", "mean"),
                                 rel_MAE_median=("rel", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
