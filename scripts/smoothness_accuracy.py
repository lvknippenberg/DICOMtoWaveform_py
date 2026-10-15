"""Accuracy of the DP envelope at several smoothness levels (dp.reference_accel_cm_s2) on the synthetic
suite and the 5 vendor-traced vascular images (explainer Table tab:smooth).

    python scripts/smoothness_accuracy.py <out_dir>

Writes accuracy_by_a0.csv (aggregates only). Vendor image location: development machine (HANDOFF.md).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dicom2waveform.evaluation.benchmark import Variant, run_synthetic, run_vendor
from dicom2waveform.synthetic import benchmark_suite

warnings.filterwarnings("ignore")
A0 = [40.0, 80.0, 160.0, 320.0, 640.0, 1280.0]
out = Path(sys.argv[1])

if __name__ == "__main__":
    var = [Variant(f"a0={a:g}", "dp", overrides={"dp": {"reference_accel_cm_s2": a}}) for a in A0]
    syn = pd.DataFrame(run_synthetic(benchmark_suite(), var, workers=12, progress=False))
    syn["pct"] = 100 * syn["env_nmae"]
    g = syn.groupby("variant")
    s = pd.DataFrame({
        "synthetic_median_pct": g["pct"].median(),
        "synthetic_lowsnr_pct": syn[syn.snr_db <= 3].groupby("variant")["pct"].median(),
        "synthetic_highsnr_pct": syn[syn.snr_db >= 20].groupby("variant")["pct"].median(),
        "synthetic_PW_pct": syn[syn.modality == "PW"].groupby("variant")["pct"].median(),
        "synthetic_CW_pct": syn[syn.modality == "CW"].groupby("variant")["pct"].median(),
        "synthetic_TDI_pct": syn[syn.modality == "TDI"].groupby("variant")["pct"].median(),
    })
    for c in ("v_max.peak_abs_err", "v_min.peak_abs_err", "v_max.peak_err", "v_min.peak_err"):
        if c in syn:
            s[c] = g[c].median()
    files = sorted(f for f in Path(r"C:\Github\DICOMtoWaveform\Annotated examples").iterdir() if f.is_file())
    ven = pd.DataFrame(run_vendor(files, var))
    s["vendor_mae_cm_s"] = ven.groupby("variant")["trace.mae"].mean()
    s["vendor_bias_cm_s"] = ven.groupby("variant")["trace.bias"].mean()
    s["vendor_p95_cm_s"] = ven.groupby("variant")["trace.p95ae"].mean()
    s = s.reindex([f"a0={a:g}" for a in A0])
    s.to_csv(out / "accuracy_by_a0.csv")
    print(s.round(2).to_string())
