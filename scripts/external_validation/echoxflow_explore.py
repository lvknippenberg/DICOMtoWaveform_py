"""Explore EchoXFlow spectral Doppler recordings with the official loader (records from the Croissant catalogue)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from echoxflow import load_croissant, open_recording
from echoxflow.plotting.annotations import _annotation_metadata_by_value_path

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Claude\Doppler\external\echoxflow\data")
kind = sys.argv[1] if len(sys.argv) > 1 else "1d_pulsed_wave_doppler"
nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cat = load_croissant(ROOT / "croissant.json", root=ROOT)
have = {p.name for p in (ROOT / "exams").iterdir()}
shown = 0
for rec in cat.recordings:
    if rec.exam_id not in have or f"data/{kind}" not in rec.array_paths:
        continue
    ann = [p for p in rec.array_paths if p.startswith(f"data/{kind}_annotation_")]
    if not ann:
        continue
    st = open_recording(rec, root=ROOT)
    print("=" * 100, "\n", rec.zarr_path)
    print("paths:", rec.array_paths)
    a = st.load_array(f"data/{kind}")
    t = st.load_array(f"timestamps/{kind}")
    sm = st.spectral_metadata(kind)
    print("spectral", a.shape, a.dtype, "min/max", float(np.nanmin(a)), float(np.nanmax(a)),
          "dt", float(np.median(np.diff(t))), "T", float(t[-1] - t[0]))
    print("baseline_row", sm.baseline_row, "nyquist", sm.nyquist_limit_mps, "axis",
          None if sm.row_velocity_mps is None else (sm.row_velocity_mps[:3], sm.row_velocity_mps[-3:]))
    print("raw spectral doc:", json.dumps(sm.raw, default=str)[:2000])
    meta = _annotation_metadata_by_value_path(st)
    for p in ann:
        v = st.load_array(p)
        m = meta.get(p) or meta.get(p.removeprefix("data/")) or {}
        print(p, v.shape, "meta:", json.dumps(m, default=str)[:700])
        print("   first pts", np.round(v[:4], 4).tolist(), " t-range", float(v[:, 0].min()), float(v[:, 0].max()))
    if "data/ecg" in rec.array_paths:
        e = st.load_array("data/ecg")
        te = st.load_array("timestamps/ecg")
        print("ecg", e.shape, "t", float(te[0]), float(te[-1]), "spectral t", float(t[0]), float(t[-1]))
    shown += 1
    if shown >= nshow:
        break
