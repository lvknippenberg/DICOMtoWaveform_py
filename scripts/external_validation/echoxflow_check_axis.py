"""Check the velocity axis of EchoXFlow spectra: overlay clinician annotations on the raw matrix (axis from the
official spectral_metadata) and print the raw spectral document (baseline, Nyquist keys)."""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from echoxflow import load_croissant, open_recording  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Claude\Doppler\external\echoxflow\data")
OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_out\axis_check")
OUT.mkdir(parents=True, exist_ok=True)
ann = pd.read_csv(ROOT.parent / "echoxflow_annotations.csv")
ann["short"] = ann["label"].fillna("").str.split("/").str[-1]
cat = load_croissant(ROOT / "croissant.json", root=ROOT)
recs = {r.recording_id: r for r in cat.recordings}
want = sys.argv[1:] or ["TR Vmax", "Manual", "AV Vmax", "MV Eprime Lateral Velocity", "A Velocity"]
done = set()
for label in want:
    for kind in ("1d_continuous_wave_doppler", "1d_pulsed_wave_doppler"):
        g = ann[(ann["short"] == label) & (ann["kind"] == kind)]
        for rid in g["recording"].unique()[:2]:
            if (rid, kind) in done:
                continue
            done.add((rid, kind))
            st = open_recording(recs[rid], root=ROOT)
            lv = st.load_array(f"data/{kind}")
            t = st.load_array(f"timestamps/{kind}")
            sm = st.spectral_metadata(kind)
            raw = {k: v for k, v in (sm.raw or {}).items() if k not in ("spectral_annotations",)}
            print("=" * 80, "\n", kind, rid, lv.shape, "\n", json.dumps(raw, default=str)[:1200])
            ax_v = np.asarray(sm.row_velocity_mps)
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.imshow(lv.T, aspect="auto", cmap="gray", extent=(t[0], t[-1], ax_v[-1], ax_v[0]), origin="upper")
            for _, a in ann[(ann["recording"] == rid) & (ann["kind"] == kind)].iterrows():
                p = st.load_array(a["path"])
                ax.plot(p[:, 0], p[:, 1], "o-" if len(p) > 1 else "o", ms=4, lw=1, label=a["short"])
                print("   ", a["short"], np.round(p[:3], 3).tolist())
            ax.set_ylim(ax_v[-1], ax_v[0])
            ax.legend(fontsize=7)
            ax.set_title(f"{kind} {rid} axis +-{sm.nyquist_limit_mps} m/s, baseline row {sm.baseline_row}")
            fig.savefig(OUT / f"{kind[:5]}_{rid}.png", dpi=80)
            plt.close(fig)
