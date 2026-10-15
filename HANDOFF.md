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
+ visual QC instead). The user chose the **full benchmark** scope. 55 pytest tests pass.

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
  `page.images` to extract embedded figures, e.g. Latham Fig. 5 pseudocode; `pymupdf` is installed in
  DopplerUS to render PDF pages to PNG).
- LaTeX: Tectonic in its own conda env `envs\tectonic` (conda-forge only; installing it into DopplerUS
  mixed channels and failed with a DLL entry-point error, so that was reverted). Run
  `envs\tectonic\Library\bin\tectonic.exe report.tex` with that env's `Library\bin` on PATH.
- Methods report for co-workers (2026-10-13): `docs/report/report.tex` (+ `make_figures.py`, which runs the
  package). Tutorial-style explainer with a clinician part (2026-10-15): `docs/explainer/explainer.tex` (+
  `make_figures.py`, all figures synthetic, so that PDF is shareable). Figures and PDF include crops of clinical images and are gitignored (`*.pdf`).
- External-validation envs (2026-10-13): `envs\echonet` (Python 3.10, PyTorch 2.3.1 CPU; 2.4 fails on the missing
  `libomp140.x86_64.dll`) for EchoNet-Measurements, `envs\echoxflow` (Python 3.13) for the EchoXFlow loader.
  External code/data in `C:\Claude\Doppler\external` (EchoNet repo + Doppler weights 1.3 GB; EchoXFlow subset ~20 GB).
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
  config.py        all parameters (dataclasses, physical units), documented in the comment above each
                   field; YAML override; config_yaml() writes the commented YAML from those comments
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
  cli.py           command line (python -m dicom2waveform ...; --write-config); __main__.py
  plot.py          QC figure (spectral region + ECG strip only); plot_overlay (full-frame review image
                   + normalised spectrum panel), normalise_brightness (display only)
  synthetic.py     synthetic DICOM displays with known truth; benchmark_suite()
  evaluation/      metrics.py (pointwise, per-beat, bootstrap, Bland-Altman), benchmark.py (variants,
                   runners for synthetic / vendor-trace / real)
scripts/           run_examples, benchmark_synthetic, benchmark_vendor, benchmark_real, benchmark_avc,
                   make_report, failure_gallery, overlay_review,
                   build_standalone (writes standalone/ from the package; --check)
scripts/external_validation/  EchoNet-Measurements and EchoXFlow validation (docs/VALIDATION.md); absolute paths
                   to the working folder and external data
standalone/        GENERATED (scripts/build_standalone.py, 2026-10-15): single-file script = the package's
                   production modules concatenated (config, io, render, ecg, envelope/likelihood, base, dp,
                   mean, postprocess, quality, aliasing, pipeline, plot, cli; no benchmark estimators) +
                   dicom2waveform_config.yaml (commented, from config.py). Never edit by hand; rebuild
                   after package changes. Identical output to the package on all 49 example DICOMs
                   (defaults and overrides) and in tests/test_standalone.py.
tests/             68 tests (DP vs brute force, rendering, pipeline accuracy per modality, R-peaks,
                   every estimator runs, aliasing unwrap both directions and per beat, click/AVC timing, click
                   modes, real examples incl. HR vs DICOM HeartRate tag, generated YAML in sync with
                   config.py, standalone up to date and identical to the package, brightness normalisation)
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
14. **Aliasing cut may not jump through signal** (`aliasing.cut_cross_weight` = 0.25, 2026-10-15). With a
    pure L1 transition the cut could jump through a flow band between two columns (~3 cost for 60 rows)
    and take the shorter route through the hollow PW spectral window near the baseline, pinned at the
    guard: the band was unwrapped the wrong way (systolic alias case: 3/8 seeds at 20 dB, 30/30 at 10 dB
    had one beat with v_min ~75 cm/s off and a ~120 cm/s v_max spike). Now a transition also costs
    `cut_cross_weight` per unit of density present in both columns between the two cut rows (an L1
    distance in a warped coordinate, still O(rows) per column via `dp_solve(..., coords)`); following a
    growing band is free. 0.1 and 0.25 clean at 10-30 dB; 0.05 and 0.5-2 leave 1/30 failing at 10 dB. Real images: only 01_10529
    changes (1 column, 2.8 cm/s). `cut_cross_weight: 0` restores the old behaviour exactly. Test:
    `test_aliasing_unwrap_every_beat`.

15. **Standalone generated from the package; one documented config** (user's choice, 2026-10-15). The
    hand-copied standalone script had drifted (aliasing keys under `quality:` vs `aliasing:`, a stale
    `render.click_background_s`, no Kalman). Now `config.py` is the only place where defaults and their
    documentation live (comments above each field; `config_yaml()` turns them into the commented YAML), and
    `scripts/build_standalone.py` concatenates the production modules into the single file. Defaults stay in
    code, the YAML stays optional (rejected: a mandatory config file, since results would then depend on which
    copy of the YAML a colleague has). Config changes: `quality:` section removed, `alias_edge_fraction` /
    `alias_min_fraction` moved to `aliasing:`; `click_background_s` removed; the estimator registry is loaded
    lazily so the production path needs no scikit-image; scikit-image and matplotlib are optional extras.
    A relative import inside a function must sit in `try: ... except ImportError` (the build script checks
    this), because the standalone script has no package to import from.

16. **`dp.normalise_floor` 0.2 -> 0.1; `dp.wall_band_rel` stays 0.03** (2026-10-15, after the one-at-a-time sweep,
    `scripts/parameter_sensitivity.py`, explainer Section 12). Floor: binds only on 17/224 synthetic displays (all
    <= 3 dB); dev split <= 3 dB 4.35 -> 4.01 %, held-out 4.07 -> 3.40 %; no change on real, vendor or pure-noise
    displays. Wall band 6 % rejected: its synthetic gain is only simulated PW/CW wall clutter; on real images it
    zeroed genuine slow flow (TDI 1.3-2.5 cm/s in 10-22 % of columns, CW up to 45 cm/s between jets, PW tails) and
    moved the TDI a' onset 4-21 ms later (error vs the user's 5 marks 6 -> 14 ms). A band in cm/s is worse (TDI
    needs <= ~1 cm/s, PW clutter ~12). Sweep with the new default: `C:\Claude\Doppler\results\sensitivity_floor01`.

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

## External validation (2026-10-13/14; details in docs/VALIDATION.md, scripts in scripts/external_validation/)

- **EchoNet-Measurements** (public weights, `github.com/echonet/measurements`, env `envs\echonet`, CPU torch 2.3.1)
  on our 1024x768 DICOMs vs the user's marks: e' MAE ~0.9-1.2 cm/s for both EchoNet and DP; E/A 4-7 cm/s both;
  CW (TR) jets: DP +15, EchoNet `trvmax` +33 cm/s outside the user's marks. EchoNet gives one point per image.
  Aortic CW of the AVC sets: DP - EchoNet AV Vmax -15 +- 32 cm/s (n = 22, r 0.85).
- **EchoXFlow** (GE Vivid E95, 46-exam 19.5 GB subset in `C:\Claude\Doppler\external\echoxflow`, env
  `envs\echoxflow`): spectral matrices rendered to Philips geometry + display mapping (black P70); 126 clinician
  traces: PW median MAE 8.7 cm/s (9 %), CW 12.6 cm/s (7 %), r 0.97-0.98; TDI e' DP +2 cm/s; TR Vmax DP -77 cm/s
  (GE clinicians mark the faint tail). **Dataset bug found**: CW annotations are relative to the display baseline
  (`baseline_frac`), the matrix axis to `spectral_row_baseline_frac`; add (row_frac - display_frac) x 2 x Nyquist.
- **Head-to-head on EchoXFlow** (both read the same synthetic 1024x768 DICOM; per-recording vs the clinician
  measurement; EchoNet automatic, DP read at the marked beats): DP better or equal for e' (lat 2.1 vs 1.9, sept
  1.8 vs 2.5 cm/s, n.s.), A (10.9 vs 15.2, p 0.06), LVOT Vmax (11.3 vs 17.5, p 0.11), AV Vmax (26.4 vs 38.1,
  p 0.03); EchoNet better for TR Vmax (39 vs 81, p < 0.001). EchoNet picks a marked beat in only 14-48 %.
- Conclusion for open point 1: the edge convention differs per measurement and per reader (user: CW dense edge;
  GE clinicians: TR faint tail, e' inside the outer edge). EchoXFlow traces/marks are a usable cardiac reference
  for per-measurement tuning (leave-one-exam-out).

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

1. Re-tune `dp.edge_density` per measurement (e', s', E/A, LVOT, AV, TR) by leave-one-exam-out on the EchoXFlow
   traces and marks (`scripts/external_validation/score_echoxflow.py`; 46 exams already downloaded), then check
   against the user's own marks (they follow a different convention on CW). Cardiac Philips images with the
   scanner's auto-trace would still help (`scripts/benchmark_vendor.py`).
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
$PY scripts/build_standalone.py   # regenerate standalone/ (script + commented YAML) after ANY package change (tests check it)
```
