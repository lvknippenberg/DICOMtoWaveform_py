"""Score all variants against vendor envelope traces drawn on real images.

Usage: python scripts/benchmark_vendor.py <annotated_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dicom2waveform.evaluation.benchmark import (ablation_variants, experiment_2d_variants, main_variants,
                                                 run_vendor)


def main(argv=None):
    argv = argv or sys.argv[1:]
    src, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file())
    variants = main_variants() + experiment_2d_variants() + ablation_variants()
    rows = run_vendor(files, variants)
    series = {(Path(r["case"]).name, r["variant"]): (r.pop("_est"), r.pop("_gt")) for r in rows if "_est" in r}
    df = pd.DataFrame(rows)
    df["case"] = df["case"].map(lambda s: Path(s).name)
    df.to_csv(out / "vendor_raw.csv", index=False)
    np.savez_compressed(out / "vendor_series.npz", **{f"{c}|{v}|est": e for (c, v), (e, g) in series.items()},
                        **{f"{c}|gt": g for (c, v), (e, g) in series.items()})
    tab = df.pivot_table(index="variant", columns="case", values="trace.mae")
    tab["mean"] = tab.mean(axis=1)
    print(tab.sort_values("mean").round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
