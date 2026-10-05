# DICOMtoWaveform_py — working notes / handoff

State as of **2026-10-05** (session 1 on 2026-09-29, plus a second round on user feedback, "Round 2"
in `docs/BENCHMARK.md`; session 2 on 2026-10-05: target feature list, click mode, review images; see
"Target features" below). Read this first in a new session, then `docs/BENCHMARK.md`.

## Purpose

Python replacement for the MATLAB tool in `C:\Github\DICOMtoWaveform` (slow; see its `HANDOFF.md`).
Given a single-frame spectral / tissue Doppler ultrasound DICOM (a rendered display, not raw Doppler
data), return:

| Output | Definition (agreed with the user) |
|---|---|
| `t` | s, shared time axis, oldest sample first (sweep-aligned) |
| `ecg` | a.u., traced from the burned-in trace; isoelectric ~0, max abs deflection 1 |
| `v_max` | cm/s, outer boundary of the **positive** spectrum (>= 0; 0 when no flow on that side) |
| `v_min` | cm/s, outer boundary of the **negative** spectrum (<= 0; 0 when no flow) |
| `v_mean` | cm/s, intensity-weighted mean velocity **inside the detected flow region** (signed, both halves) |
| `v_mean_pos` / `v_mean_neg` | same, positive / negative half only |

Plus `t_rpeaks`, `t_clicks` (valve clicks), `t_avc` (aortic valve closure), per-column confidence,
and flags (modality PW/CW/TDI, `aliasing_suspected`, `aliasing_corrected`, `ecg_clipped`,
`n_clicks`, `snr_proxy_db`, ...). After aliasing correction `v_max`/`v_min` may exceed the display
limits (that is the point). Timing between ECG and Doppler events is what downstream
analysis (a separate repo) uses, as for the MATLAB tool.

## Status

Everything in `Plan.txt` (the user's plan, `docs/plan/Plan.txt`) plus the review
(`docs/plan/Plan_review.md`) is implemented, except the annotation GUI (the user chose synthetic data
+ visual QC instead). The user chose the **full benchmark** scope. 47 pytest tests pass.

- Package `dicom2waveform` (editable install in the env), entry point `extract_waveforms(path)`.
  ~0.2 s per 1024x768 image, ~0.6 s for 1920x1080.
- All 10 estimators of Plan section 23 plus a configurable segmentation pipeline, post-processing,
  synthetic DICOM generator, metrics, benchmark runners, report generator, failure gallery.
- Benchmark run completed; results in `docs/results/REPORT.md` (copy) and
  `C:\Claude\Doppler\results\report\REPORT.md` (original, with raw CSVs).
- Clinician review images for the 17 cardiac examples: `C:\Claude\Doppler\review\overlay\`
  (`scripts/overlay_review.py`; full frame with overlay + a brightness-normalised copy of the
  spectrum strip with the same overlay). The user plans to have clinicians review them.

## Target features (user, 2026-10-05)

The user annotated what must eventually be extracted from each DICOM:
`C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples\Points of interest\` (one PNG per
example, blue marks, plus `Readme.txt`). Blue dot = a velocity to measure, blue line = a timing
(start/end of a wave, for temporal windows). The full list will be implemented later.
Definitions and clinical context per recording type: `docs/FEATURES.md`.

| Feature | Where | Derivable from | State (checked against the marks) |
|---|---|---|---|
| e' and a' peak velocity | TDI (10292, 10364, 10638, 277, 376, 03_10673) | `v_min` | not built; envelope within ~0.2-1 cm/s of most dots |
| a' onset (for iLAD: P wave -> a') | TDI (line on the baseline) | `v_min`: last return to 0 before the a' peak | not built; +-7 ms in 5/5 |
| E and A peak velocity | PW mitral (E3Q0C6H6, 10529, L3VHDBZ5) | `v_max` | not built; envelope 2.5 cm/s inside the dots on average (narrow tips cut, open point 3) |
| Jet peak velocity | CW (E3Q0C6HS, L3VHDS0Z) | `v_min` | not built; envelope ~10 cm/s outside the dots (user marks the dense edge) |
| Aortic valve closure | CW (L3VHCNLD, 232, 11747, 12485, 14018) | clicks, `t_avc` | built; 4/5 within +-3 ms, 14018 missed (open point 4) |
| T-wave end (for EMW, with AVC) | ECG, 01_232 | ECG trace | not built |
| P-wave onset (for iLAD) | ECG | ECG trace | not built |

Notes: labelling E vs A and e' vs a' needs ECG timing (A/a' after P, before R). 03_10673 is in AF
(e' only; detect AF from RR irregularity, skip a' and P). 01_5012 is very dark (unannotated). Open
question to the user: iLAD = P onset -> a' onset (as drawn) or -> a' peak (classic PA-TDI)? In
01_10638 a' onset nearly coincides with P onset.

## Environment

- Conda env: `C:\Users\p70096100\AppData\Local\miniconda3\envs\DopplerUS` (Python 3.12). The user
  allows modifying it. Installed: numpy 2.5, scipy 1.18, pydicom 3.0, scikit-image 0.26, matplotlib
  3.11, pandas 3.0, pyyaml, pytest, pypdf, tabulate, pylibjpeg; package installed with `pip install -e .`.
  Run with the full path: `.../envs/DopplerUS/python.exe`.
- Working folder for plans, QC figures and results (outside the repo, contains patient images):
  `C:\Claude\Doppler` — `Plan.txt`, `Plan_review.md`, `qc/`, `results/` (one results PNG per example
  DICOM + `csv/`, benchmark CSVs, `report/`), `Kathpalia/` and `Latham/` (paper PDFs the user supplied).
- Not installed: node, poppler (the Read tool cannot render PDFs; use `pypdf` for text, and
  `page.images` to extract embedded figures, e.g. Latham Fig. 5 pseudocode).
- Do not name a script `inspect.py` (it shadows the stdlib module and breaks numpy import).

## Data

| Set | Location | Use |
|---|---|---|
| 17 new cardiac examples | `C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples` | QC, unsupervised comparison, `tests/test_real_examples.py` |
| 15 cardiac (3 patients) | `C:\Github\DICOMtoWaveform\DICOM examples` | QC, unsupervised comparison |
| 12 cardiac (EMW) | `C:\Github\DICOMtoWaveform\EMW examples` | QC, unsupervised comparison |
| 5 vascular with vendor trace | `C:\Github\DICOMtoWaveform\Annotated examples` | **only real ground truth** (blue trace RGB 98,164,255) |

All Philips (iE33, EPIQ CVx, Affiniti 70C; iU22 for the vascular set), single frame, no
`WaveformSequence` (the ECG exists only in the pixels). 15 RGB + 2 MONOCHROME2; one 1920x1080. Spectral
region data type 3 (PW, incl. TDI) or 4 (CW); region order varies. **Images contain burned-in patient
identifiers: never commit DICOMs, full-frame renderings or QC figures** (`.gitignore` excludes `*.png`
except `docs/results/*.png`, which are synthetic/aggregate only).

## Code map

```
dicom2waveform/
  config.py        all parameters (dataclasses, physical units), YAML override; configs/default.yaml
  io.py            DICOM read, US region calibration (region by unit codes), metadata
  render.py        crop, overlay masks (baseline, markers, ECG, text, ticks, vendor trace), colour map ->
                   level, sweep detection + time order, click detection/bridging, level_to_power
  ecg.py           ECG trace extraction, R-peak detection (port of DetectRpeaks.m)
  envelope/
    base.py        SideImage (baseline-outward view of one side), EnvelopeEstimator, ColumnwiseEstimator
    likelihood.py  noise stats, soft membership, local density, Otsu (+ effectiveness metric)
    dp.py          DP estimator (primary): edge_cost, split_cost (MATLAB), O(n) L1 min-convolution
    spectral.py    percentile, GM, MGM, SNSI, adaptive SNSI (Kathpalia), QIA (Li) + presence gate
    segmentation.py threshold, otsu, configurable segmentation pipeline, Latham OMM
    postprocess.py median / Savitzky-Golay / Kalman CV-CA (+RTS)
    mean.py        mean envelopes inside the flow region
  quality.py       flow mask, SNR proxy, aliasing (wrap-around) test
  aliasing.py      aliasing correction: least-signal cut path through the circular display (DP),
                   rows beyond the cut moved by one Nyquist span -> taller, unwrapped Spectrogram
  pipeline.py      extract_waveforms(), WaveformResult, envelopes_from_spectrogram()
  plot.py          QC figure (spectral region + ECG strip only); plot_overlay (full-frame review image
                   + normalised spectrum panel), normalise_brightness (display only)
  synthetic.py     synthetic DICOM displays with known truth; benchmark_suite()
  evaluation/      metrics.py (pointwise, per-beat, bootstrap, Bland-Altman), benchmark.py (variants,
                   runners for synthetic / vendor-trace / real)
scripts/           run_examples, benchmark_synthetic, benchmark_vendor, benchmark_real, benchmark_avc,
                   make_report, failure_gallery, overlay_review, write_default_config
tests/             47 tests (DP vs brute force, rendering, pipeline accuracy per modality, R-peaks,
                   every estimator runs, aliasing unwrap both directions, click/AVC timing, click
                   modes, real examples incl. HR vs DICOM HeartRate tag, default.yaml in sync with
                   config.py, brightness normalisation)
```

## Key decisions and learnings (with the reasons)

MATLAB learnings carried over (see the MATLAB `HANDOFF.md`): region by unit codes; zero row =
`ReferencePixelY0` (header wins over pixel detection); both signs of `PhysicalDeltaY` (positive =
inverted display; output sign follows the header); sweep = columns without drawn baseline, NaN, rotate
to time order; missing ECG allowed; R-peaks on QRS-band energy 8-30 Hz, upper-half-median threshold,
RR-relative T-wave guard.

New in this port:
1. **DP data term = outer-edge model**, not the MATLAB two-region split. The split assumes a flow
   region filled from the baseline; it collapses to v = 0 on hollow spectra (PW spectral window, TDI
   bands, a dark wall-filter gap in CW). Edge cost: excess density beyond the edge + shortfall in a band
   of h rows just inside + tiny height prior; the interior is free. Test: `tests/test_dp.py`.
2. **Background subtraction of the density** (50th percentile). Without it, uniform noise attracts the
   edge outward and it runs to the display limit at low SNR.
3. **Density normalised by its 99th percentile, edge at theta = 0.5** ("half the typical in-flow
   density"). theta chosen by leave-one-image-out on the vendor traces (0.5 in 3/5 folds, held-out MAE
   2.16 cm/s); normalisation chosen on the synthetic dev split. Without normalisation a single theta
   cannot serve dense (vascular/CW) and sparse (speckled TDI) renderings: theta 0.5 cut faint TDI,
   theta 0.2 overshot the vendor traces by +3..11 cm/s.
4. **Wall band** (3% of display height next to the baseline) is nearly free to include; edges inside
   it are reported as 0. Fixed CW images where a dark band next to the baseline pulled v_max to 0.
5. **Acceleration prior scaled with the display range** (a_ref = 640 cm/s^2 at 100 cm/s range).
   MATLAB open issue: a fixed a_ref clipped systolic upstrokes on high-range displays.
6. **Path decoupled across the sweep** (not adjacent in time).
7. **Overlays masked, not zeroed**: chromatic pixels on grey displays (ECG, markers, vendor traces),
   grey pixels on colour-map (TDI) displays, baseline colour everywhere if chromatic, **text/ticks** =
   small bright components with a dark surround (saturated flow has a bright surround). Masked pixels
   are filled by vertical interpolation.
8. **Valve clicks** (broadband transients <= 35 ms in the column mean) are reported as events (see
   Round 2 for the detector). Default `click_mode` changed from 'bridge' to **'keep'** on 2026-10-05 at the
   user's request: the envelope should trace the spectrum as displayed, clicks included. 'bridge'
   (interpolate over click columns on PW/CW) and 'follow' remain available.
9. **TDI copper colour map** -> luminance x 255/160 (brightest copper colour has luminance ~150); noise
   margins are in grey levels.
10. **ECG**: colour searched in a band of rows (not one row: one EPIQ trace did not cross the B-mode
    lower edge); the baseline colour is excluded explicitly (three iE33 CW images picked the orange
    baseline); per column the stroke top/bottom at local extremes, centre elsewhere; `ecg_clipped` flag
    (the scanner clips tall R-waves to its display box); DICOM `HeartRate` tag used as RR prior for
    search-back (fixed a missed clipped beat), search-back adds one peak per gap.
11. **Aliasing flag = wrap-around test** (both edges at their display limits in the same columns). One
    edge alone at its limit is common (mirror/clutter texture on a small side) and is not aliasing.
    When it fires, the display is unwrapped (Round 2).
12. **Spectrum methods get a common flow-presence gate** (vertical run of supra-Otsu pixels >= 1.5% of
    the side depth). The published methods assume a known flow direction; without the gate they report
    noise on every no-flow side (ungated variants are kept in the 2-D experiment).
13. **Synthetic suite seeding uses zlib.crc32**, not `hash()` (randomised per process).

## Tried and rejected

- Fixed 2% supra-Otsu gate, and a median-referenced gate for spectrum methods: too permissive on TDI
  speckle, resp. too strict when most column sides carry flow.
- DP with a free interior (no height prior / no background correction): runaway to the display limit.
- DP interior weight 0.1 (instead of the edge model): still collapsed on sparse TDI bands.
- DP theta 0.2 as default: best on synthetic truth before normalisation, but +3..11 cm/s against
  vendor traces.
- Second-order (curvature) DP term (Plan section 20): squares the state space; not implemented.

## Benchmark results (2026-09-29; details in docs/results/REPORT.md)

- **Vendor traces (real, 5 vascular):** DP MAE 1.92 cm/s (corr 0.99; held-out 2.16); Otsu 3.68;
  OMM (Latham) 4.67; GM 8.5; percentile 10.1; MGM 11.4; fixed threshold 11.8; adaptive SNSI 13.3;
  SNSI 17.9; QIA 44.8. Plain MATLAB-DP port 2.23 (MATLAB reported 2.31: port validated).
- **Synthetic, held-out split:** normalised envelope MAE (median) DP 1.2%, OMM 1.6%, Otsu 1.7%,
  adaptive SNSI 3.1%, threshold 3.2%, MGM 4.5%, GM 5.1%, SNSI 5.4%, QIA 5.8%, percentile 10.4%.
  Low SNR (<= 3 dB): DP 2.2% spectral / 4.0% TDI (best). High SNR: image methods equivalent.
- 2-D experiment at <= 3 dB: column-wise 8.6% > +smoothing 6.4% > 2-D segmentation 5.1% > DP 2.8%.
- One DP parameter set works for spectral and TDI (interaction Delta 0.013 vs 0.07-0.16 for most others).
- Real images have SNR proxies 15-43 dB, i.e. the high-SNR end of the synthetic range: on real data
  the rendering effects (clutter, mirror, clicks, text) separate methods more than noise does.
- Real cardiac images (44): ECG + R-peaks found in all; HR equals the DICOM `HeartRate` tag within
  1 bpm in all 11 files that have it.

## Open points

1. **Edge convention per modality (highest priority).** theta was calibrated on vascular PW vendor
   traces. The user's marks (2026-10-05) pull in opposite directions: on CW the user marks the peak at
   the dense edge and DP reads ~10 cm/s further out (4 dots); on PW mitral DP sits 2.5 cm/s inside
   (6 dots). Every global setting that takes in more PW signal pushes CW further out (sweep: edge
   density 0.35/0.4, noise margin 16/8, density window 5x3, accel x2; density window 5x3 was the best
   PW candidate: PW MAE 2.8 -> 1.9, vendor 1.92 -> 1.97, CW +10 -> +14). The user accepts separate
   PW/TDI(/CW) settings but wants more examples first. Cardiac vendor auto-traces would also help.
2. **Mirror artefacts** (bright mirrored band on the weak side, e.g. PW `L3VHDBZ5`, `E3Q0C6H6` with a
   small negative range) are reported as flow; not flagged. Also open in MATLAB. Need to confirm what
   those bands are before "fixing" (genuine bidirectional flow exists, e.g. FI77VVJW).
3. **Narrow, faint peak tips are cut off** (user, `01_10529`: tops of E and A, thin diastolic peaks;
   also faint TDI tails in `01_5012`). Not a threshold effect (edge density 0.3 barely moves the
   line): the density window (9 x 5 px) and the acceleration prior remove thin structures.
4. **AVC threshold**: one marked beat missed (04_14018, strength 5.7 vs `ecg.avc_min_snr` 6.0, at
   +13 ms from the mark); consider a threshold relative to the other beats of the image. AVC is also
   computed on TDI, where it is meaningless (hidden in the review images).
5. **Latham OMM adaptation**: the per-window STFT Otsu (EM selection) is emulated by time-averaging the
   display; published areas are applied after resampling to the paper's 5 ms x 31.25 Hz grid with 512
   rows over the display span (assumption: display span <-> the paper's +-8 kHz).
6. **Spectrum methods** use an assumed 40 dB display dynamic range to de-compress the level (unknown
   in the DICOM); `dynamic_range_db` is a parameter, not benchmarked.
7. **Mean envelopes** are validated only on synthetic data (power-weighted truth). No real reference.
8. **ECG amplitude** is uncalibrated (pixels); normalised to max |deflection| = 1, so it is not
   comparable between images.
9. Runtimes in the benchmark tables were measured with 12 parallel workers (inflated).
10. The synthetic ECG is always green; greyscale ECG extraction is only tested on the 2 mono examples.

## Next steps (suggested order)

1. Get cardiac images with the scanner's auto-trace (PW, CW, TDI); run `scripts/benchmark_vendor.py`
   on them (trace colour may differ: `VENDOR_TRACE_RGB` in `evaluation/benchmark.py`); re-tune
   `dp.edge_density` (possibly per modality) by leave-one-image-out; update the report.
2. Decide on the mirror question with the user (visual review of `L3VHDBZ5`, `E3Q0C6H6`, `AFTQAAFO`),
   then add a mirror flag (e.g. correlation of the weak side's texture with the strong side's
   envelope, reflected about the baseline).
3. Implement the target feature list (section above): per-beat peak picking and labelling (E/A, e'/a',
   CW jet peak) using ECG timing, a' onset, P-wave onset and T-wave end on the ECG, AF detection.
   Validate against the blue marks (mark pixels = annotated PNG minus the DICOM pixels, blue
   components; dot vs line by aspect ratio).
4. Collect clinician feedback on the review images; then address open points 1 and 3 (per-modality
   settings, narrow tips) with more PW examples.
5. Optional: annotation GUI (Plan section 25) if manual cardiac references are wanted.
6. Packaging: pin dependency versions once the API settles; CI running `pytest` (synthetic tests need
   no data).

## How to reproduce everything

```bash
PY=/c/Users/p70096100/AppData/Local/miniconda3/envs/DopplerUS/python.exe
$PY -m pytest -q
$PY scripts/run_examples.py "<dicom_dir>" "C:/Claude/Doppler/results" --csv-dir "C:/Claude/Doppler/results/csv"
$PY scripts/benchmark_synthetic.py C:/Claude/Doppler/results --workers 12          # ~4 min
$PY scripts/benchmark_vendor.py "C:/Github/DICOMtoWaveform/Annotated examples" C:/Claude/Doppler/results
$PY scripts/benchmark_real.py C:/Claude/Doppler/results "<dir1>" "<dir2>" ...
$PY scripts/make_report.py C:/Claude/Doppler/results   # regenerates REPORT.md (conclusions section is hand-written; re-add)
$PY scripts/failure_gallery.py C:/Claude/Doppler/results --n 20
$PY scripts/overlay_review.py "<dicom_dir>" C:/Claude/Doppler/review/overlay   # full-frame review images + brightness-normalised spectrum panel (PHI: institution only)
$PY scripts/write_default_config.py   # regenerate configs/default.yaml after changing config.py defaults (tests/test_config.py checks it)
```
