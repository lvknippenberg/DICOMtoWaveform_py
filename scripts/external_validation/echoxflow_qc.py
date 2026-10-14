"""QC plots for EchoXFlow recordings: rendered spectrum + DP envelope (with and without aliasing correction) +
clinician annotations. Usage: python echoxflow_qc.py recording_id [recording_id ...]"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, r"C:\Claude\Doppler\validation")
warnings.filterwarnings("ignore")
import score_echoxflow as SX  # noqa: E402  (module-level scoring runs once; fine for QC)

OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_out\qc")
OUT.mkdir(parents=True, exist_ok=True)
ann = pd.read_csv(SX.ROOT.parent / "echoxflow_annotations.csv")
ann["short"] = ann["label"].fillna("").str.split("/").str[-1]
for rid in sys.argv[1:]:
    for kind, g in ann[ann["recording"] == rid].groupby("kind"):
        st = SX.open_recording(SX.recs[rid], root=SX.ROOT)
        name = kind
        lv = st.load_array(f"data/{name}").astype(np.float32)
        t = st.load_array(f"timestamps/{name}").astype(float)
        sm = st.spectral_metadata(name)
        img, dy, z, dt, tc = SX.render(lv, t, np.asarray(sm.row_velocity_mps, float))
        dtype = 4 if "continuous" in kind else 3
        env = SX.envelope(img, dy, z, dt, dtype)
        SX.CFG.aliasing.correct = False
        env0 = SX.envelope(img, dy, z, dt, dtype)
        SX.CFG.aliasing.correct = True
        H = img.shape[0]
        v_top, v_bot = (0 - z) * dy, (H - 1 - z) * dy
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.imshow(img, aspect="auto", cmap="gray", extent=(tc[0], tc[-1], v_bot, v_top))
        ax.plot(tc, env0["v_max"], "y", lw=0.7, label="DP no alias corr")
        ax.plot(tc, env0["v_min"], "y", lw=0.7)
        ax.plot(tc, env["v_max"], "r", lw=1, label="DP (default)")
        ax.plot(tc, env["v_min"], "c", lw=1)
        for _, a in g.iterrows():
            p = st.load_array(a["path"]).astype(float)
            p[:, 1] += SX.annotation_offset_mps(sm)
            ax.plot(p[:, 0], p[:, 1] * 100, "o-" if len(p) > 1 else "o", ms=4, lw=1.5, label=a["short"][:20])
        ax.set_ylim(min(v_bot, v_top) * 1.6 if env["alias"] else min(v_bot, v_top),
                    max(v_bot, v_top) * (1.6 if env["alias"] else 1))
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"{rid} {kind} {env['modality']} alias corrected={env['alias']}")
        fig.savefig(OUT / f"{rid}_{kind[:5]}.png", dpi=80)
        plt.close(fig)
        print("wrote", rid, kind)
