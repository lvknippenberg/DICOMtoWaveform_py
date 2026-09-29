"""All variants on real cardiac DICOMs without reference traces.

Records per method: agreement with every other method, smoothness, fraction
of samples at the display limit, SNR proxy and runtime. Optionally writes a
DP QC figure per file.

Usage: python scripts/benchmark_real.py <out_dir> <dicom_dir> [<dicom_dir> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dicom2waveform.evaluation.benchmark import experiment_2d_variants, main_variants, run_variants_on_image
from dicom2waveform.evaluation.metrics import smoothness
from dicom2waveform.io import load_dicom


def main(argv=None):
    argv = argv or sys.argv[1:]
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in argv[1:]:
        files += sorted(p for p in Path(d).iterdir() if p.is_file())
    variants = main_variants() + experiment_2d_variants()
    rows, pairs = [], []
    for f in files:
        try:
            img = load_dicom(f)
        except Exception as e:
            print(f"{f.name}: cannot read ({e})")
            continue
        res = run_variants_on_image(img, None, variants, keep_series=True,
                                    case_info={"case": f.name, "set": f.parent.name})
        ser = {r["variant"]: r.pop("_series") for r in res if "_series" in r}
        for r in res:
            s = ser.get(r["variant"])
            if s is not None:
                for sig in ("v_max", "v_min"):
                    r.update({f"{sig}.{k}": v for k, v in smoothness(s[sig]).items()})
                    r[f"{sig}.peak"] = float(np.nanmax(np.abs(s[sig])))
            rows.append(r)
        names = list(ser)
        for a in names:
            for b in names:
                if a >= b:
                    continue
                d = [np.nanmean(np.abs(ser[a][s] - ser[b][s])) for s in ("v_max", "v_min")]
                pairs.append({"case": f.name, "a": a, "b": b, "env_mad": float(np.nanmean(d))})
        print(f"{f.name}: {len(res)} variants", flush=True)
    pd.DataFrame(rows).to_csv(out / "real_raw.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "real_pairs.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
