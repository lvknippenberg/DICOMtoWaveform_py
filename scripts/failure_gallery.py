"""Failure gallery (Plan section 49): the worst synthetic cases per method, with truth overlaid.

Usage: python scripts/failure_gallery.py <results_dir> [--n 20] [--metric env_nmae]
Reads synthetic_raw.csv; writes <results_dir>/report/failures/<method>/*.png (synthetic images only).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.envelope import BENCHMARK_ORDER, make_estimator
from dicom2waveform.plot import plot_result
from dicom2waveform.synthetic import benchmark_suite, generate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--metric", default="env_nmae")
    ap.add_argument("--methods", default=",".join(BENCHMARK_ORDER))
    a = ap.parse_args(argv)
    res = Path(a.results_dir)
    df = pd.read_csv(res / "synthetic_raw.csv")
    specs = {f"synthetic:{s.waveform}:{s.modality}:{s.snr_db}:{s.seed}": s for s in benchmark_suite()}
    cfg = load_config()
    cfg.keep_diagnostics = True
    for m in a.methods.split(","):
        d = df[df.variant == m].sort_values(a.metric, ascending=False).head(a.n)
        out = res / "report" / "failures" / m
        out.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(d.iterrows(), 1):
            sp = specs.get(row["case"])
            if sp is None:
                continue
            case = generate(sp)
            r = extract_waveforms(case.image, cfg, estimator=make_estimator(m))
            title = f"#{rank} {m} {row['case']} {a.metric}={row[a.metric]:.3f}"
            plot_result(r, out / f"{rank:02d}.png", title=title,
                        extra={"truth": (case.truth["v_max"], case.truth["v_min"])})
        print(m, len(d))


if __name__ == "__main__":
    sys.exit(main())
