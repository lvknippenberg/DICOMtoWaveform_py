"""Command line: extract waveforms from one or more DICOMs.

    python -m dicom2waveform file.dcm [more.dcm ...] [--config cfg.yaml] [--out-dir DIR] [--plot]
    python -m dicom2waveform --write-config my_config.yaml

(the standalone script takes the same arguments). Per file it writes <name>.csv (t, ecg, v_mean,
v_mean_pos, v_mean_neg, v_max, v_min), <name>_events.json (R-peaks, valve clicks, aortic valve
closure, flags) and with --plot a QC figure <name>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, write_config
from .pipeline import extract_waveforms
from .plot import plot_result


def main(argv: list[str] | None = None) -> int:
    prog = "python -m dicom2waveform" if Path(sys.argv[0]).name == "__main__.py" else None
    p = argparse.ArgumentParser(prog=prog, description="ECG and Doppler envelopes from spectral / tissue Doppler DICOMs.")
    p.add_argument("dicoms", nargs="*", help="DICOM file(s)")
    p.add_argument("--config", help="YAML file with parameter overrides (see dicom2waveform_config.yaml)")
    p.add_argument("--out-dir", default=".", help="output folder (default: current folder)")
    p.add_argument("--plot", action="store_true", help="also write a QC figure per file (needs matplotlib)")
    p.add_argument("--write-config", metavar="PATH",
                   help="write the documented configuration (defaults, or --config applied) and exit")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.write_config:
        write_config(cfg, a.write_config)
        print(f"wrote {a.write_config}")
        return 0
    if not a.dicoms:
        p.error("no DICOM files given")
    cfg.keep_diagnostics = cfg.keep_diagnostics or a.plot
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for f in a.dicoms:
        name = Path(f).stem
        try:
            res = extract_waveforms(f, cfg)
        except Exception as e:  # keep going over a folder of files
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
            n_fail += 1
            continue
        res.to_csv(out / f"{name}.csv")
        events = {"t_rpeaks": res.t_rpeaks.tolist(), "t_clicks": res.t_clicks.tolist(),
                  "t_avc": res.t_avc.tolist(), "flags": res.flags}
        with open(out / f"{name}_events.json", "w", encoding="utf8") as fh:
            json.dump(events, fh, indent=2, default=str)
        if a.plot:
            plot_result(res, out / f"{name}.png", title=name)
        print(f"{name}: {res.flags['modality']}, {len(res.t_rpeaks)} R-peaks, {len(res.t_avc)} AVC, "
              f"aliasing {'corrected' if res.flags['aliasing_corrected'] else 'no'}, "
              f"{res.meta['runtime_s']:.2f} s")
    return 1 if n_fail else 0
