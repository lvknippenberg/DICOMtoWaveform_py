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
res.flags        # modality (PW/CW/TDI), aliasing_suspected, ecg_clipped, snr_proxy_db, ...
res.to_csv("out.csv")
```

NaN marks samples without data (the sweep, i.e. the write position of the scanner's circular display
buffer). Velocity signs follow the DICOM header (`PhysicalDeltaY`): on an operator-inverted display this is
opposite to the on-screen label.

Command line (QC figures + CSVs for a folder):

```bash
python scripts/run_examples.py <dicom_dir> <out_dir>
```

## How it works

1. **Calibration from the header.** The spectral region, time per column, velocity per row and the
   zero-velocity row come from the DICOM Ultrasound Region Calibration module (selected by unit codes,
   never by item order).
2. **Rendering to intensity.** Overlays (drawn baseline, time markers, ECG trace, text, scale ticks,
   vendor traces) are masked and filled; colour maps (TDI copper map) are converted to a scalar level;
   valve clicks are bridged; the sweep is located and the output rotated to time order.
3. **ECG** is traced from its rendered colour, per column; R-peaks from QRS-band energy (port of the MATLAB
   `DetectRpeaks.m`, with the DICOM `HeartRate` tag as a prior for search-back).
4. **Envelopes (default: DP).** Noise-referenced soft signal membership -> local density ->
   background-corrected and normalised -> per side, the globally optimal outer edge by dynamic
   programming with a physiological acceleration prior. The edge sits at half the image's typical
   in-flow density (calibrated on scanner auto-traces).
5. **Mean envelopes** are intensity-weighted (de-compressed display power by default) over the pixels
   between the baseline and the outer edge.

All parameters are in `dicom2waveform/config.py` (dataclasses, physical units), overridable from YAML
(`load_config("my.yaml")`) or a dict.

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
python scripts/make_report.py <out_dir>                           # REPORT.md + figures
```

## Development

```bash
pip install -e .[dev]
pytest
```

Tests use synthetic images (`dicom2waveform.synthetic`), which render complete fake DICOM displays with
known envelopes, ECG and artefacts. Tests on real example DICOMs run when the folder in
`tests/conftest.py` (or `$D2W_EXAMPLES`) exists.

**Patient data:** the example images contain burned-in identifiers. Never commit DICOMs, full-frame
renderings or QC figures (`.gitignore` excludes images and results).
