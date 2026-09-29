"""Run the synthetic benchmark: all variants x the synthetic suite.

Usage: python scripts/benchmark_synthetic.py <out_dir> [--workers 10] [--quick]
Writes <out_dir>/synthetic_raw.csv (one row per case x variant).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from dicom2waveform.evaluation.benchmark import (ablation_variants, experiment_2d_variants, main_variants,
                                                 run_synthetic)
from dicom2waveform.synthetic import benchmark_suite


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="2 SNRs, 1 seed")
    ap.add_argument("--groups", default="main,2d,ablation")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = benchmark_suite(snrs=(0, 20), seeds=(0,)) if a.quick else benchmark_suite()
    groups = a.groups.split(",")
    variants = []
    if "main" in groups:
        variants += main_variants()
    if "2d" in groups:
        variants += experiment_2d_variants()
    if "ablation" in groups:
        variants += ablation_variants()
    print(f"{len(specs)} cases x {len(variants)} variants", flush=True)
    t0 = time.perf_counter()
    rows = run_synthetic(specs, variants, workers=a.workers)
    df = pd.DataFrame(rows)
    df.to_csv(out / "synthetic_raw.csv", index=False)
    n_err = df["error"].notna().sum() if "error" in df else 0
    print(f"done in {time.perf_counter() - t0:.0f} s, {len(df)} rows, {n_err} errors", flush=True)


if __name__ == "__main__":
    sys.exit(main())
