"""Valve clicks and aortic valve closure (AVC) on real DICOMs.

Without a reference, AVC is checked for plausibility and consistency: detection
rate per beat (beats whose 0.2-0.5 s window lies inside the record), the R->AVC
delay distribution (MATLAB reference on the EMW set: 38/38 beats, 363 +/- 38 ms),
and the within-file spread of the delay.

Usage: python scripts/benchmark_avc.py <out_dir> <dicom_dir> [<dicom_dir> ...]
Writes <out_dir>/avc_raw.csv (one row per detected AVC or undetected beat).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dicom2waveform import extract_waveforms


def main(argv=None):
    argv = argv or sys.argv[1:]
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in argv[1:]:
        for f in sorted(p for p in Path(d).iterdir() if p.is_file()):
            res = extract_waveforms(f)
            r = res.t_rpeaks
            full = r[r + 0.5 <= res.t[-1]]
            for a in res.t_avc:
                prev = r[r < a]
                rows.append({"set": Path(d).name, "case": f.name, "modality": res.flags["modality"],
                             "t_avc": a, "r_to_avc_ms": 1000 * (a - prev[-1]) if len(prev) else np.nan})
            rows.append({"set": Path(d).name, "case": f.name, "modality": res.flags["modality"],
                         "n_beats_full_window": len(full), "n_avc": len(res.t_avc),
                         "n_clicks": len(res.t_clicks)})
            print(f"{f.name}: beats {len(full)}, AVC {len(res.t_avc)}, clicks {len(res.t_clicks)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "avc_raw.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
