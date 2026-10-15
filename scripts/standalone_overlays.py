"""Run the generated standalone script on a folder of DICOMs: overlay image, CSV and events per file.

usage: python scripts/standalone_overlays.py standalone/dicom2waveform_standalone.py <dicom_dir> <out_dir>

Outputs show the full frame with burned-in patient details: write them outside the repository
(e.g. C:/Claude/Doppler/results/stand-alone) and keep them within the institution.
"""
import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
script, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("dicom2waveform_standalone", script)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)
out.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(p for p in src.iterdir() if p.is_file()):
    name = f.name
    try:
        img = S.load_dicom(f)
        res = S.extract_waveforms(img, {"keep_diagnostics": True})
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAILED ({type(e).__name__}: {e})")
        rows.append({"file": name, "error": f"{type(e).__name__}: {e}"})
        continue
    res.to_csv(out / f"{name}.csv")
    with open(out / f"{name}_events.json", "w", encoding="utf8") as fh:
        json.dump({"t_rpeaks": res.t_rpeaks.tolist(), "t_clicks": res.t_clicks.tolist(), "t_avc": res.t_avc.tolist(),
                   "flags": res.flags}, fh, indent=2, default=str)
    S.plot_overlay(res, img, out / f"{name}_overlay.png", title=name)
    fl = res.flags
    lo, hi = res.meta["velocity_limits_cm_s"]
    rows.append({"file": name, "modality": fl["modality"], "ecg_found": fl["ecg_found"], "n_rpeaks": len(res.t_rpeaks),
                 "n_clicks": fl["n_clicks"], "n_avc": len(res.t_avc), "aliasing_suspected": fl["aliasing_suspected"],
                 "aliasing_corrected": fl["aliasing_corrected"], "snr_proxy_db": round(fl["snr_proxy_db"], 1),
                 "v_max_peak_cm_s": round(float(S.np.nanmax(res.v_max)), 1),
                 "v_min_peak_cm_s": round(float(S.np.nanmin(res.v_min)), 1),
                 "display_limits_cm_s": f"{lo:.1f} / {hi:.1f}", "runtime_s": round(res.meta["runtime_s"], 2)})
    print(f"{name}: {fl['modality']}, {len(res.t_rpeaks)} R-peaks, {len(res.t_avc)} AVC, "
          f"aliasing {'corrected' if fl['aliasing_corrected'] else 'no'}, {res.meta['runtime_s']:.2f} s", flush=True)

keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(out / "summary.csv", "w", newline="", encoding="utf8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} files -> {out}")
