"""Full-frame review images: each DICOM with v_max / v_min, the extracted ECG,
R-peaks, valve clicks and AVC drawn on top (dicom2waveform.plot.plot_overlay).

Usage: python scripts/overlay_review.py <dicom_dir> <out_dir> [--config cfg.yaml]
The images include burned-in patient details: keep them inside the
institution and outside the repository.
"""
import argparse
import sys
from pathlib import Path

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.io import load_dicom
from dicom2waveform.plot import plot_overlay


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dicom_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg.keep_diagnostics = True
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(p for p in Path(a.dicom_dir).iterdir() if p.is_file()):
        try:
            img = load_dicom(f, frame=cfg.frame)
            res = extract_waveforms(img, cfg)
        except Exception as e:  # keep going over a batch
            print(f"{f.name:30s} skipped: {type(e).__name__}: {e}")
            continue
        title = f"{f.name}  [{res.flags['modality']}, {res.meta['method']}, click_mode={cfg.render.click_mode}]"
        plot_overlay(res, img, out / f"{f.name}.png", title=title)
        print(f"{f.name:30s} {res.flags['modality']:3s} R-peaks {len(res.t_rpeaks)}  clicks {len(res.t_clicks)}  "
              f"AVC {len(res.t_avc)}")


if __name__ == "__main__":
    sys.exit(main())
