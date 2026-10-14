"""QC: synthetic EchoXFlow DICOM with EchoNet's point, clinician references and the DP envelope (both time axes)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, r"C:\Github\DICOMtoWaveform_py\standalone")
import dicom2waveform_standalone as S  # noqa: E402

IN = Path(r"C:\Claude\Doppler\validation\echoxflow_dicom")
OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_out\qc_echonet")
OUT.mkdir(parents=True, exist_ok=True)
R = pd.read_csv(r"C:\Claude\Doppler\validation\echoxflow_out\echonet_vs_dp_echoxflow.csv")
for task, rid in [a.split(":") for a in sys.argv[1:]]:
    side = json.loads((IN / task / f"{rid}.json").read_text())
    res = S.extract_waveforms(IN / task / f"{rid}.dcm")
    sp = res.spectrogram
    print(rid, "sweep cols", int(sp.sweep.sum()), "order[0]", int(sp.order[0]), "W", sp.level.shape[1])
    t0 = side["t0"]
    rgb = sp.rgb
    H = rgb.shape[0]
    v_top, v_bot = (0 - sp.zero_row) * sp.dy, (H - 1 - sp.zero_row) * sp.dy
    fig, ax = plt.subplots(figsize=(12, 4.5))
    W = rgb.shape[1]
    ax.imshow(rgb, aspect="auto", extent=(t0, t0 + W * sp.dt, v_bot, v_top))  # raw column order
    tt = res.t + t0
    ax.plot(tt, res.v_max, "r", lw=0.8, label="DP v_max (res.t)")
    ax.plot(tt, res.v_min, "c", lw=0.8)
    for x in side["refs"]:
        ax.plot(x["t"], x["v"], "o", mfc="none", mec="#2f6bff", ms=9, mew=2)
    g = R[(R.recording == rid) & (R.task == task)]
    for _, r in g.iterrows():
        ax.plot(r["echonet_t"], -r["echonet"] if np.median([x["v"] for x in side["refs"]]) < 0 else r["echonet"],
                "x", color="yellow", ms=12, mew=3, label=f"EchoNet {r['measure']}")
    ax.legend(fontsize=7)
    ax.set_title(f"{task} {rid}")
    fig.savefig(OUT / f"{task}_{rid}.png", dpi=80)
    plt.close(fig)
