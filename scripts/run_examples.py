"""Run the pipeline on a folder of DICOMs and write QC figures + CSVs.

Usage: python scripts/run_examples.py <dicom_dir> <out_dir> [--method dp] [--config cfg.yaml]
Output stays outside the repository (figures show patient images).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.plot import plot_result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dicom_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--method", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--csv-dir", default=None, help="write CSVs here instead of out_dir")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg.keep_diagnostics = not a.no_figures
    if a.method:
        cfg.estimator.method = a.method
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.dicom_dir).iterdir() if p.is_file())
    summary = []
    for f in files:
        try:
            res = extract_waveforms(f, cfg)
        except Exception as e:  # keep going over a batch
            print(f"{f.name:30s} FAILED: {type(e).__name__}: {e}")
            summary.append({"file": f.name, "error": str(e)})
            continue
        csv_dir = Path(a.csv_dir) if a.csv_dir else out
        csv_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(csv_dir / f"{f.name}.csv")
        if not a.no_figures:
            plot_result(res, out / f"{f.name}.png", title=f.name)
        hr = 60 / np.median(np.diff(res.t_rpeaks)) if len(res.t_rpeaks) > 1 else float("nan")
        row = {"file": f.name, **{k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item())
                                  for k, v in res.flags.items()},
               "n_rpeaks": len(res.t_rpeaks), "hr_bpm": hr, "hr_header": res.meta["dicom"].get("HeartRate"),
               "vmax_peak": float(np.nanmax(res.v_max)), "vmin_peak": float(np.nanmin(res.v_min)),
               "runtime_s": res.meta["runtime_s"]}
        summary.append(row)
        print(f"{f.name:30s} {row['modality']:3s} R={row['n_rpeaks']} HR={hr:5.1f}/{row['hr_header']} "
              f"vmax={row['vmax_peak']:7.1f} vmin={row['vmin_peak']:7.1f} SNR={row['snr_proxy_db']:5.1f} "
              f"alias={row['aliasing_suspected']} {row['runtime_s']*1000:.0f} ms")
    (Path(a.csv_dir) if a.csv_dir else out).joinpath("summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
