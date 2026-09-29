# Review of `Plan.txt` — Doppler/ECG waveform extraction from DICOM

Reviewed against: the stated goal (a function returning ECG + five Doppler envelopes),
the 17 example DICOMs in `C:\DICOM waveforms\DICOM doppler examples\...`, and the
learnings recorded in the MATLAB repo (`C:\Github\DICOMtoWaveform\HANDOFF.md`).

## 0. Decisions (2026-09-29)

- **Mean envelopes**: intensity-weighted mean velocity *inside the detected flow region*.
  `v_mean_pos` / `v_mean_neg` use only the positive / negative half.
- **Max/min**: `v_max` = outer boundary of the positive spectrum, `v_min` = outer boundary
  of the negative spectrum. Both are defined in every column (≈0 when there is no flow on that side).
- **Scope**: the **full benchmark** as in `Plan.txt` (all 10 estimators, ablations, SNR
  experiments). This overrides the "defer" advice in §3.5/§4 below.
- **Reference data**: synthetic images + visual QC only (for now). With no reference
  traces on real images, quantitative ranking comes from **synthetic data only**. The
  synthetic generator must reproduce the real rendering effects (log compression, colour maps,
  overlays, sweep gap, clutter, speckle) closely enough for that ranking to mean something.
  The real images give qualitative QC and an unsupervised comparison: agreement between
  methods, smoothness, and failure rate.

## 1. Verdict

`Plan.txt` is a good *research benchmark* plan for one sub-problem: finding the outer
(max-velocity) boundary of a spectrogram. It is not yet a plan for the deliverable you
asked for. Four problems:

1. **Half of the outputs are missing.** It does not cover the ECG, the mean envelopes
   (overall, positive-only, negative-only), or the relation between max/min and upper/lower.
2. **It re-solves problems the DICOM header already solves.** It proposes automatic ROI,
   baseline and axis detection and normalized [0,1]×[−1,1] coordinates. All 17 files
   have the US Region Calibration module, so ROI, time scale, velocity scale and zero line
   can be read directly, in s and cm/s.
3. **It misses the practical problems in these images:** colour maps, burned-in overlays
   (ECG, baseline, tick marks, text), the sweep gap, aliasing, mirror/clutter bands, and
   burned-in patient data (PHI).
4. **The benchmark is too large for the data.** It has 10 estimators, ablations, a
   2×2 SNR×modality factorial and a 60/20/20 split, but there are 17 unannotated images.
   That is about 4 images per factorial cell, with no reference trace to score against.

Recommendation: work **production-first**. Build and validate the extraction pipeline
with one primary envelope method, the DP tracker, which the MATLAB work already showed to be
best (MAE 2.3 vs 18.1 cm/s against vendor traces). Compare it against a small set of
baselines. Keep the rest of the literature benchmark as an optional later phase.

## 2. What the example data actually look like

| Property | Observation | Consequence |
|---|---|---|
| Vendor/systems | Philips iE33, EPIQ CVx, Affiniti 70C | Single-vendor rendering. Overlay colours and layout are consistent |
| Frames | All single-frame | As the plan assumes |
| Size | 16 × 1024×768, 1 × 1920×1080 | Kernel sizes must be in physical units (the plan's §41 is right) |
| Photometric | 15 RGB, 2 MONOCHROME2 | Need both colour and greyscale paths for ECG and Doppler |
| ECG | **No `WaveformSequence`**. ECG is only burned into the pixels (green in RGB, grey/white in MONOCHROME2) | ECG must be extracted from the image. Amplitude is uncalibrated (a.u.) |
| Regions | `SequenceOfUltrasoundRegions` present in all. Doppler region = units X 4 (s) / Y 7 (cm/s). **Region order varies** | Select the region by unit codes, never by index |
| Time scale | `PhysicalDeltaX` = 3.52 ms/px (75 mm/s) or 2.64 ms/px (100 mm/s) | Native sampling ≈ 284 / 379 Hz |
| Velocity scale | `PhysicalDeltaY` 0.056 (TDI) to 1.67 (CW) cm/s/px | Ranges of about 30× |
| Zero line | `ReferencePixelY0` is present | Use it as the baseline. The plan's §4 methods 1–4 are then only a sanity check |
| Region type | Data type 3 (PW, including TDI) ×10, 4 (CW) ×7 | TDI cannot be told apart by data type alone. Use the velocity range and colour map (or the "TDI" label) |
| TDI rendering | Copper/orange colour map on black | Convert colour to intensity with a LUT, not RGB→grey |
| Overlays inside the Doppler box | Orange baseline, orange dotted time markers, white scale ticks, ECG trace crossing the top rows (e.g. `01_10529`, `04_14018`, `04_12485`), bright vertical sweep/marker lines | Mask these and treat them as *missing*, not signal |
| Sweep gap | Visible in e.g. `E3Q0C6H6`, `04_12485` | Return NaN there, never interpolate across it. Reorder oldest→newest (MATLAB learning #5) |
| Aliasing | `01_10529` (mitral inflow PW) wraps around the bottom | Flag it. Do not report the wrapped boundary as the minimum velocity |
| Low SNR / clutter | `04_14018` CW is very noisy. Strong bands near the baseline in several CW images | This is where the DP prior matters. It is also the mirror-artefact risk for positive-only/negative-only means |
| PHI | At least one image shows a **patient name** in the header | Keep DICOMs and full-frame figures out of Git. Crop QC figures to the Doppler region |

## 3. Specific modifications

### 3.1 Scope and outputs (add)
Define the public contract first:

```python
res = extract_waveforms(path, config=None)
res.t              # s, shared time axis for ECG and Doppler (native column rate)
res.ecg            # a.u. (normalized), NaN where not present
res.v_mean         # cm/s, intensity-weighted mean velocity over the whole flow region (signed)
res.v_mean_pos     # cm/s, weighted mean over the positive half only
res.v_mean_neg     # cm/s, weighted mean over the negative half only
res.v_max          # cm/s, outer boundary on the positive side (peak positive velocity)
res.v_min          # cm/s, outer boundary on the negative side (peak negative velocity)
res.confidence     # per-sample 0..1, per signal
res.flags          # aliasing_suspected, sweep_gap, ecg_missing, mirror_suspected, modality (PW/CW/TDI)
res.meta           # calibration, region box, config, version
res.diagnostics    # optional intermediate images for QC plots
```
The definitions of the mean and max/min need confirming (see questions).

### 3.2 DICOM stage (replaces plan §2–4)
- Read calibration from the US Region Calibration module: Doppler region by units (s, cm/s),
  `PhysicalDeltaX/Y`, `ReferencePixelY0`. Handle both signs of `PhysicalDeltaY` (MATLAB #4).
  Mind the 0-based DICOM coordinates (MATLAB #2).
- Keep the plan's automatic ROI/baseline detection **only as a fallback and a
  consistency check**. The header wins when they disagree (MATLAB #3).
- Work in physical units from the start. Keep pixel coordinates for annotation and debugging only.

### 3.3 Rendering → intensity (new)
- **Colour → scalar:** for TDI/colour maps, sample the colour bar in the image or fit the
  map once per system, then invert it with a nearest-LUT lookup. For grey, use the value directly.
- **Overlay masks:** hue-based masks for the green ECG and orange markers. Masks for the
  baseline line, text and scale ticks. Masked pixels become *missing* in the likelihood map.
- **Log compression:** display grey is log-compressed. The integrated-power methods
  (percentile, GM, MGM, SNSI, QIA) assume linear power, so they need an explicit, configurable
  de-compression step, and the choice must be documented. This is one more reason not to
  put much weight on them (the plan's §40 is right, but should draw this conclusion).

### 3.4 ECG (new)
- RGB: per-column centroid of the green mask. Mono: search a band at and above the Doppler
  top edge and track the thin bright line with a small DP (same machinery as the envelope).
- Put it on the **same time axis** as the Doppler (same columns, same sweep-gap NaNs).
- Output in a.u., baseline-corrected. R-peak detection is optional; `DetectRpeaks.m`
  logic can be ported later. Timing accuracy matters more than amplitude here.

### 3.5 Envelopes (restructure plan §6–22)
- **Primary:** noise-referenced soft membership → DP shortest path per side with an
  L1 slope prior in *physical* units (cm/s²) → two-state Viterbi for flow direction.
  This is a port of the validated MATLAB DP method, reusing its learnings (soft membership
  is essential; edge-cost and flow-presence gates were tried and rejected).
- **Drop the second-order term** from plan §20 as a DP state. It squares the state space
  (~400² states × ~900 columns) and is the likely reason for slow runtimes. Use first-order
  L1 with O(W·H) min-convolution (distance-transform trick), then optional light
  Savitzky–Golay smoothing.
- **Mean envelopes:** compute them *inside the detected flow region* (between the baseline
  and v_max, or v_min). Over the whole column, noise pulls the mean toward zero. Make the
  weighting (display grey vs de-compressed power) a config option. Document that this is
  an image-domain estimate of mean velocity, not the scanner's Vmean.
- **Clutter band:** exclude ±N cm/s around the baseline (config, physical units) from
  positive/negative-only means. Mirror artefacts are flagged, not "fixed" (MATLAB open issue 1).
- **Baselines for comparison (keep a few):** fixed/percentile threshold, Otsu + morphology +
  outermost pixel, and one integrated-power method (MGM *or* SNSI). **Defer:** GM, adaptive
  SNSI, QIA, the Latham-style method, Kalman, and the full ablation grid. Latham in
  particular works on audio-derived spectrograms, which the plan itself notes.

### 3.6 Validation (right-size plan §24–54)
- **Unit tests on synthetic spectrograms** (plan §57 is good). Add the rendering effects:
  colour map, overlays, sweep gap, aliasing wrap, clutter band, log compression.
- **Visual QC gallery** for all 17 images, cropped to the Doppler region (no PHI).
- **Ground truth:** the cheapest high-quality reference is **vendor auto-trace** images,
  as in `Annotated examples/`. Failing that, an annotation GUI (plan §25) on a subset.
  A handful of annotated images is enough for a sanity benchmark. It is not enough for
  the SNR factorial or train/val/test splits, so drop those until there is more data.
- **Metrics:** primary = MAE in **cm/s**, since calibration is known. Secondary = bias,
  peak velocity error, VTI error, and **timing error** of onset/peak/end relative to the ECG,
  because downstream use depends on ECG–Doppler timing. Keep an image-domain SNR proxy
  as a descriptor, not as a factorial axis.

### 3.7 Engineering
- Drop `fit()` from the estimator interface. Nothing is learned; `detect(img, calib, cfg)` is enough.
- Config: dataclasses with defaults, plus optional YAML override (plan §56). Store parameters
  in physical units (ms, cm/s, cm/s²) and convert to pixels per image.
- Performance target: **< 0.5 s per image** in pure NumPy/SciPy. Add numba only if profiling
  demands it.
- Package with `pyproject.toml`, `pytest`, `.gitignore` for DICOMs, outputs and figures.
- Reproducibility: store config and version in `res.meta`. The plan's `results/` tree is
  fine for the benchmark scripts.

## 4. Revised phases

| Phase | Content | Exit criterion |
|---|---|---|
| 0 | Repo skeleton, env, packaging, synthetic-image generator | `pytest` green |
| 1 | DICOM IO + calibration + colour→intensity + overlay masks + sweep gap | QC plot of calibrated Doppler intensity for all 17 files |
| 2 | ECG extraction (RGB + mono) | ECG visually correct on all 17. R-peaks align with the image |
| 3 | DP max/min envelopes + flow-direction model + confidence | Synthetic MAE < 1 px. Visual QC on 17 files |
| 4 | Mean envelopes (overall / pos / neg), flags (aliasing, mirror) | Synthetic tests with known mean. Visual QC |
| 5 | Validation vs ground truth + 3 baselines + runtime | Report table, failure gallery |
| 6 (optional) | Remaining literature methods, SNR study, aliasing unwrapping | Only if phase 5 shows it matters |

Proposed layout (`C:\Github\DICOMtoWaveform_py`):
```
dicom2waveform/
  __init__.py          # extract_waveforms()
  io.py  calibration.py  render.py  ecg.py  quality.py  plot.py  config.py
  envelope/ likelihood.py  dp.py  mean.py  baselines.py
tests/     synthetic.py  test_*.py
scripts/   run_examples.py  benchmark.py
```

---

## 5. Implementation status (2026-09-29)

Code: `C:\Github\DICOMtoWaveform_py` (package `dicom2waveform`; committed 2026-09-29, see HANDOFF.md).
Design and tuning protocol: `docs/BENCHMARK.md`. Results: `C:\Claude\Doppler\results\report\REPORT.md`.

Implemented against the plan:
- `extract_waveforms()` returns t, ECG, v_mean, v_mean_pos, v_mean_neg, v_max, v_min, R-peaks, confidence and flags.
  It runs in ~0.2 s per image.
- Header calibration, overlay masking (baseline, markers, ECG, text, ticks, vendor traces), TDI colour map,
  sweep handling, and valve-click bridging.
- ECG extraction for RGB and mono. R-peak detection is ported from the MATLAB code, with the DICOM HeartRate as a prior.
  The heart rate matches the header within 1 bpm in all 11 files that have one.
- All 10 estimators of plan section 23, the configurable segmentation pipeline (P0-P4, B1-B5, morphology,
  components, E1-E4), and post-processing (median, Savitzky-Golay, Kalman CV/CA with RTS).
- Synthetic DICOM generator with known truth. Metrics (sections 26-31), paired bootstrap, Bland-Altman helpers,
  factorial interaction, the 2-D experiment (section 54), ablations (section 45), and the failure gallery (section 49).
- 39 pytest tests.

Deviations from Plan.txt (and why):
- The annotation GUI (section 25) was not built. The user chose synthetic data + visual QC, plus the 5 vendor-traced images.
- There are no YAML-per-experiment files. Every parameter is a documented dataclass default, overridable by YAML
  (`configs/default.yaml`).
- The DP has no second-order term. It would square the state space.
- The DP data term was changed from the MATLAB two-region split to an outer-edge model. The split collapses on hollow spectra
  (PW spectral window, TDI bands).
- The spectrum methods use a common per-side flow-presence gate. Without it they report noise on every no-flow side.

Open issues:
- Edge convention on CW jets. The DP edge level (theta = 0.5 of the typical in-flow density) was calibrated on
  vascular PW vendor traces. On CW jets with a faint high-velocity tail (e.g. AFTQAAFO) the DP follows the dense core.
  Cardiac vendor traces (PW, CW and TDI) are needed to settle this.
- Mirror artefacts. A mirrored band on the weak side is reported as flow (not flagged). This is also an open issue from the MATLAB work.
- Aliasing is flagged (wrap-around test) but not unwrapped.
