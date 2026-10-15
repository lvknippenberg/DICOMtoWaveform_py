# dicom2waveform

Extract the **ECG** and the **Doppler velocity envelopes** from single-frame spectral / tissue Doppler
ultrasound DICOM images (rendered displays, not raw Doppler data).

```python
from dicom2waveform import extract_waveforms

res = extract_waveforms("path/to/dicom")
res.t            # s, shared time axis, oldest sample first
res.ecg          # a.u., isoelectric ~ 0, max |deflection| = 1
res.v_max        # cm/s, outer edge of the positive spectrum (>= 0)
res.v_min        # cm/s, outer edge of the negative spectrum (<= 0)
res.v_mean       # cm/s, intensity-weighted mean velocity inside the flow region (signed)
res.v_mean_pos   # cm/s, same, positive half only
res.v_mean_neg   # cm/s, same, negative half only
res.t_rpeaks     # s, R-peak times
res.t_clicks     # s, valve-click times (narrow broadband transients; not part of the envelopes)
res.t_avc        # s, aortic valve closure per beat (strongest click 0.2-0.5 s after R)
res.flags        # modality (PW/CW/TDI), aliasing_suspected/_corrected, ecg_clipped, snr_proxy_db, ...
res.to_csv("out.csv")
```

NaN marks samples without data (the sweep, i.e. the write position of the scanner's circular display
buffer). Velocity signs follow the DICOM header (`PhysicalDeltaY`): on an operator-inverted display this is
opposite to the on-screen label.

Command line (CSV + events JSON per file, `--plot` adds a QC figure):

```bash
python -m dicom2waveform file.dcm [more.dcm ...] [--config my.yaml] [--out-dir DIR] [--plot]
python -m dicom2waveform --write-config my.yaml        # documented configuration to edit
python scripts/run_examples.py <dicom_dir> <out_dir>    # QC figures + CSVs for a folder
```

## How it works

1. **Calibration from the header.** The spectral region, time per column, velocity per row and the
   zero-velocity row come from the DICOM Ultrasound Region Calibration module (selected by unit codes,
   never by item order).
2. **Rendering to intensity.** Overlays (drawn baseline, time markers, ECG trace, text, scale ticks,
   vendor traces) are masked and filled; colour maps (TDI copper map) are converted to a scalar level;
   valve clicks are detected and reported as events (kept in the envelope by default; `render.click_mode`); the sweep is located and the output rotated
   to time order.
3. **Aliasing (PW/TDI)**: when the flow wraps around the display, an automatic per-column baseline shift
   unwraps it, so `v_max` / `v_min` can exceed the display limits (`flags["aliasing_corrected"]`).
4. **ECG** is traced from its rendered colour, per column; R-peaks from QRS-band energy (port of the MATLAB
   `DetectRpeaks.m`, with the DICOM `HeartRate` tag as a prior for search-back); aortic valve closure
   from the strongest valve click in the 0.2-0.5 s window after each R-peak (port of the MATLAB method).
5. **Envelopes (default: DP).** Noise-referenced soft signal membership -> local density ->
   background-corrected and normalised -> per side, the globally optimal outer edge by dynamic
   programming with a physiological acceleration prior. The edge sits at half the image's typical
   in-flow density (calibrated on scanner auto-traces); on TDI at half of each column's own maximum,
   so faint but real tissue motion is kept.
6. **Mean envelopes** are intensity-weighted (de-compressed display power by default) over the pixels
   between the baseline and the outer edge.

All parameters are in `dicom2waveform/config.py` (dataclasses, physical units, each documented in the
comment above it), overridable from YAML (`load_config("my.yaml")`) or a dict; only the keys you give
change. The documented reference file `standalone/dicom2waveform_config.yaml` is generated from
`config.py`, comments included, so the defaults and their documentation live in one place.

The method is explained from first principles, with a plain-language version for clinicians, in
`docs/explainer/explainer.pdf` (all figures synthetic, shareable).

## Standalone script (for sharing)

`standalone/dicom2waveform_standalone.py` is a single-file copy of the production path (DP envelope,
aliasing correction, ECG, R-peaks, clicks / AVC, plots, command line) that needs only numpy, scipy and
pydicom (pyyaml for YAML configs, matplotlib for plots). Give colleagues that file and
`standalone/dicom2waveform_config.yaml`.

Both files are **generated** from the package by `scripts/build_standalone.py`: never edit them by hand.
After any change to the package run

```bash
python scripts/build_standalone.py          # rewrites standalone/
python scripts/build_standalone.py --check  # exit 1 if standalone/ is out of date
```

`tests/test_standalone.py` fails when they are out of date, and checks that the script gives exactly the
package's results and imports without the package, scikit-image or matplotlib. The benchmark estimators
are left out (`estimator.method` other than `dp` raises an error there).

Notes for users of the earlier hand-written standalone script (before 2026-10-15): the configuration
class is `PipelineConfig` (was `Config`); `plot_result(res)` needs a result computed with
`{"keep_diagnostics": True}` (the command line's `--plot` does this), and the spectrogram is in
`res.diagnostics["spectrogram"]` (was `res.spectrogram`); the internal function names are the package's.

## Benchmark

Ten envelope estimators are implemented behind one interface (`dicom2waveform.envelope`):
fixed threshold, Otsu + morphology, integrated-power percentile, GM, MGM, SNSI, adaptive SNSI
(Kathpalia 2016), QIA (Li 2019), OMM (Latham 2021) and the DP method, plus a configurable
segmentation pipeline for ablations. See `docs/BENCHMARK.md` for the design and results, and
`REFERENCES.md` for the literature.

```bash
python scripts/benchmark_synthetic.py <out_dir>                   # synthetic suite, known truth
python scripts/benchmark_vendor.py "<annotated_dir>" <out_dir>    # real images with scanner traces
python scripts/benchmark_real.py <out_dir> <dicom_dir> [...]      # real images, unsupervised
python scripts/benchmark_avc.py <out_dir> <dicom_dir> [...]       # valve clicks / aortic valve closure
python scripts/make_report.py <out_dir>                           # REPORT.md + figures
```

## Target features

`docs/FEATURES.md` lists, per recording type (TDI, mitral inflow, CW), the features to be extracted
(e′/a′, E/A, jet peak velocity, aortic valve closure, P onset, T end, EMW, ...) with definitions and
clinical context. Not all are implemented yet.

## Review images

```bash
python scripts/overlay_review.py <dicom_dir> <out_dir>
```

One PNG per DICOM: the full frame as displayed with `v_max` / `v_min` (solid red / blue), the side means
`v_mean_pos` / `v_mean_neg` (dashed), the extracted ECG (magenta), R-peaks, valve clicks and aortic valve
closure drawn on top with black-outlined lines, and below it a column-aligned copy of the ECG and
spectrum strip with brightness normalised for visibility (display only; TDI redrawn with a copper map)
and the same overlay (`dicom2waveform.plot.plot_overlay`; colours in `OVERLAY_STYLE`). These images show
patient details: keep them inside the institution.

The same with the generated standalone script, plus CSV / events / a summary table per folder, the
user's manual marks, and a smoothness sweep (see HANDOFF.md, "How to reproduce everything"):

```bash
python scripts/standalone_overlays.py standalone/dicom2waveform_standalone.py <dicom_dir> <out_dir>
python scripts/standalone_poi_overlay.py standalone/dicom2waveform_standalone.py <dicom_dir> <poi_dir> <out_dir>
python scripts/smoothness_sweep.py standalone/dicom2waveform_standalone.py <dicom_dir> <out_dir>
```

## Development

```bash
pip install -e .[dev]     # core needs numpy, scipy, pydicom, pyyaml; extras: plot, bench, dev
pytest
```

Tests use synthetic images (`dicom2waveform.synthetic`), which render complete fake DICOM displays with
known envelopes, ECG and artefacts. Tests on real example DICOMs run when the folder in
`tests/conftest.py` (or `$D2W_EXAMPLES`) exists.

**Patient data:** the example images contain burned-in identifiers. Never commit DICOMs, full-frame
renderings or QC figures (`.gitignore` excludes images and results).

`standalone/` is generated from the package (`python scripts/build_standalone.py`); a test fails
when it is out of date.
