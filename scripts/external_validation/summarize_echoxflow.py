"""Summary tables (markdown) of the EchoXFlow scoring runs: primary (rendered, black 70, offset fix) and sensitivity runs."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_out")
RUNS = [("rendered_black70", "primary: display mapping (black P70), rendered to Philips geometry"),
        ("rendered_black60", "black point P60"), ("rendered_black80", "black point P80"),
        ("rendered_black0", "no display mapping (raw matrix values)"),
        ("native_black70", "native EchoXFlow resolution (no re-rendering)"),
        ("rendered_black70_nooffsetfix", "without the annotation baseline-offset correction")]


def trace_table(T):
    T = T.copy()
    T["peak_err"] = T["peak_est"] - T["peak_ref"]
    g = T.groupby("kind")
    return pd.DataFrame({
        "traces": g.size(), "exams": g["exam"].nunique(),
        "MAE median (cm/s)": g["mae"].median(), "MAE mean": g["mae"].mean(),
        "bias mean": g["bias"].mean(), "rel. MAE median": g["rel_mae"].median(),
        "r median": g["corr"].median(), "peak error median": g["peak_err"].median(),
        "peak timing |err| median (ms)": g["peak_t_err_ms"].apply(lambda x: x.abs().median()),
    }).round(2)


def point_table(P):
    P = P.copy()
    P["abs"] = P["err"].abs()
    P["rel"] = P["abs"] / P["ref"].abs()
    g = P.groupby(["kind", "label"])
    return pd.DataFrame({"n": g.size(), "ref mean |v|": g["ref"].apply(lambda x: x.abs().mean()),
                         "MAE (cm/s)": g["abs"].mean(), "bias": g["err"].mean(),
                         "rel. MAE median": g["rel"].median()}).round(2)


lines = ["# EchoXFlow scoring of the DP envelope\n"]
for tag, desc in RUNS:
    tf, pf = OUT / f"traces_{tag}.csv", OUT / f"points_{tag}.csv"
    if not tf.exists():
        continue
    T, P = pd.read_csv(tf), pd.read_csv(pf)
    lines.append(f"## {tag}: {desc}\n")
    lines.append(f"{pd.concat([T['exam'], P['exam']]).nunique()} exams, "
                 f"{pd.concat([T['recording'], P['recording']]).nunique()} recordings\n")
    lines.append("Hand-drawn traces (error = |DP| - |trace|):\n")
    lines.append(trace_table(T).to_markdown() + "\n")
    lines.append("Peak marks (DP peak within +-60 ms of the mark):\n")
    lines.append(point_table(P).to_markdown() + "\n")
(OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf8")
print("\n".join(lines))
