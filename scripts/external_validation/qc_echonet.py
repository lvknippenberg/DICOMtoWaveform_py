"""QC overlays (patient data; stays in C:\\Claude\\Doppler): display + DP envelope + manual marks (labelled) + EchoNet points
+ R-peaks, for the annotated files used in score_echonet.py."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, r"C:\Claude\Doppler\validation")
import score_echonet as SE  # noqa: E402  (runs the scoring once; uses its cache and helpers)

OUT = Path(r"C:\Claude\Doppler\validation\figures")
OUT.mkdir(exist_ok=True)
R = pd.read_csv(SE.OUT / "scores_vs_marks.csv")
for name, g in R.groupby("file"):
    res = SE.result(name)
    spec = res.spectrogram
    rgb = spec.rgb[:, spec.order]
    v_top = (0 - spec.zero_row) * spec.dy
    v_bot = (rgb.shape[0] - 1 - spec.zero_row) * spec.dy
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].imshow(rgb, aspect="auto", extent=(res.t[0], res.t[-1] + spec.dt, v_bot, v_top), interpolation="nearest")
    if v_top < v_bot:
        ax[0].invert_yaxis()
    ax[0].plot(res.t, res.v_max, "r", lw=0.8)
    ax[0].plot(res.t, res.v_min, "c", lw=0.8)
    for _, r in g.drop_duplicates(["ref_t", "measure"]).iterrows():
        ax[0].plot(r["ref_t"], r["ref_v"], "o", mfc="none", mec="#2f6bff", ms=10, mew=2)
        ax[0].annotate(r["measure"], (r["ref_t"], r["ref_v"]), color="#2f6bff", xytext=(6, 6),
                       textcoords="offset points", fontsize=10, weight="bold")
    for (model, t, v), c in zip(g.drop_duplicates(["model", "echonet_t"])[["model", "echonet_t", "echonet_v"]]
                                 .itertuples(index=False), ["yellow", "lime", "orange", "magenta", "white"] * 3):
        ax[0].plot(t, v, "x", color=c, ms=11, mew=2.5, label=f"EchoNet {model}")
    ax[0].legend(loc="upper right", fontsize=7)
    ax[0].set_title(f"{name}: blue circles = manual marks, x = EchoNet official point, red/cyan = DP")
    ax[1].plot(res.t, res.ecg, "g", lw=0.8)
    for tr in res.t_rpeaks:
        ax[1].axvline(tr, color="k", ls=":")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / f"echonet_{name}.png", dpi=80)
    plt.close(fig)
    print("wrote", name, "R-peaks", np.round(res.t_rpeaks, 2))
