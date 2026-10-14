"""Test whether CW annotation velocities are offset by (spectral_row_baseline_frac - baseline_frac) * 2 * Nyquist:
the first and last point of a hand-drawn CW trace lie at jet onset/end, where the true velocity is ~0."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from echoxflow import load_croissant, open_recording

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Claude\Doppler\external\echoxflow\data")
ann = pd.read_csv(ROOT.parent / "echoxflow_annotations.csv")
ann["short"] = ann["label"].fillna("").str.split("/").str[-1]
cat = load_croissant(ROOT / "croissant.json", root=ROOT)
recs = {r.recording_id: r for r in cat.recordings}
rows = []
for kind in ("1d_continuous_wave_doppler", "1d_pulsed_wave_doppler"):
    for _, a in ann[(ann["short"] == "Manual") & (ann["kind"] == kind)].iterrows():
        st = open_recording(recs[a["recording"]], root=ROOT)
        sm = st.spectral_metadata(kind)
        p = st.load_array(a["path"])
        p = p[np.argsort(p[:, 0])]
        raw = sm.raw or {}
        bf, sbf = raw.get("baseline_frac"), raw.get("spectral_row_baseline_frac")
        pred = -(sbf - bf) * 2 * sm.nyquist_limit_mps if bf is not None and sbf is not None else np.nan
        rows.append(dict(kind=kind[3:5], nyq=sm.nyquist_limit_mps, baseline_frac=bf, row_baseline_frac=sbf,
                         first_v=p[0, 1], last_v=p[-1, 1], predicted_offset=pred, peak=p[:, 1].min()))
D = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(D.round(3).to_string())
