"""Inventory of spectral annotations in the downloaded EchoXFlow exams: label, modality, geometry (point/trace), counts,
plus spectral geometry (rows, Nyquist, dt). Writes echoxflow_annotations.csv (one row per annotation)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from echoxflow import load_croissant, open_recording
from echoxflow.plotting.annotations import _annotation_metadata_by_value_path

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Claude\Doppler\external\echoxflow\data")
cat = load_croissant(ROOT / "croissant.json", root=ROOT)
have = {p.name for p in (ROOT / "exams").iterdir()}
rows = []
for rec in cat.recordings:
    if rec.exam_id not in have:
        continue
    for kind in ("1d_pulsed_wave_doppler", "1d_continuous_wave_doppler"):
        if f"data/{kind}" not in rec.array_paths:
            continue
        ann = [p for p in rec.array_paths if p.startswith(f"data/{kind}_annotation_")]
        if not ann:
            continue
        try:
            st = open_recording(rec, root=ROOT)
            sm = st.spectral_metadata(kind)
            shape = st.group[f"data/{kind}"].shape
            meta = _annotation_metadata_by_value_path(st)
        except Exception as e:  # noqa: BLE001
            print("skip", rec.zarr_path, e)
            continue
        for p in ann:
            v = st.load_array(p)
            m = meta.get(p) or {}
            rows.append(dict(exam=rec.exam_id, recording=rec.recording_id, zarr=rec.zarr_path, kind=kind, path=p,
                             label=m.get("label", ""), n_points=len(v), t0=float(v[:, 0].min()), t1=float(v[:, 0].max()),
                             v_min=float(v[:, 1].min()), v_max=float(v[:, 1].max()), rows=shape[1], frames=shape[0],
                             nyquist=sm.nyquist_limit_mps, baseline_row=sm.baseline_row))
df = pd.DataFrame(rows)
df.to_csv(ROOT.parent / "echoxflow_annotations.csv", index=False)
print(len(df), "annotations in", df["recording"].nunique(), "recordings,", df["exam"].nunique(), "exams")
df["short"] = df["label"].str.split("/").str[-1]
df["geom"] = np.where(df["n_points"] > 1, "trace", "point")
pd.set_option("display.width", 200)
print(df.groupby(["kind", "geom", "short"]).agg(n=("path", "size"), recs=("recording", "nunique"),
                                                 npts=("n_points", "median"), vmin=("v_min", "median"),
                                                 vmax=("v_max", "median"), rows=("rows", "median"),
                                                 nyq=("nyquist", "median")).sort_values("n", ascending=False).to_string())
