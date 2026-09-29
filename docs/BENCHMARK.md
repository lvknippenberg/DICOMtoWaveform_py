# Envelope benchmark: design

This document describes how the envelope estimators are compared. Numbers are produced by
`scripts/make_report.py` into `REPORT.md` next to the benchmark CSVs (kept outside the repository).

## Question

Which envelope estimator is most accurate on single-frame rendered Doppler DICOMs; how does the
ranking change with SNR; and does one detector work for spectral (PW/CW) and tissue (TDI) Doppler?

## Hypotheses (Plan.txt section 53)

Written down before the full benchmark was run, but after exploratory development on a few
synthetic cases and the vendor-traced images, so they are not strictly pre-registered.

- **H1** At high SNR, spectrum-based methods (adaptive SNSI, QIA) are competitive with image methods.
- **H2** With decreasing SNR, image-based methods degrade more gracefully.
- **H3** A global curve-tracking formulation (DP) beats independent-column estimators at low SNR.
- **H4** One architecture works for spectral Doppler and TDI; parameters may differ.
- **H5** The best practical solution is a hybrid image-domain method (background estimation,
  normalisation, segmentation/likelihood, DP tracking).

## Estimators

| Label | Method | Class | Source of the definition |
|---|---|---|---|
| threshold | fixed threshold, outermost pixel | image | Plan section 6 |
| otsu | Gaussian + Otsu + close/open/fill/remove + outermost pixel | image | Plan section 7 |
| percentile | integrated power spectrum, p = 95% | spectrum | Plan section 8 |
| gm | geometric method (line from peak-power point to f_H) | spectrum | Kathpalia 2016 II-A |
| mgm | modified GM (line from first to last IPS point) | spectrum | Kathpalia 2016 II-A |
| snsi | signal-noise slope intersection, x = 0.1 | spectrum | Kathpalia 2016 II-A |
| asnsi | adaptive SNSI with altered GM, one-third condition, adaptive power threshold | spectrum | Kathpalia 2016 II-A, II-B |
| qia | quadratic iteration algorithm | spectrum | Li 2019 section 2.1 |
| latham | OMM: Otsu with EM-selected window, hierarchical morphology, biggest gap | image | Latham 2021 II-B..D, Fig. 5 |
| dp | soft membership, density, background correction + normalisation, DP outer edge | image | this work (extends the MATLAB DP) |

Common framework steps applied to all methods: header calibration, overlay masking, colour-map
conversion, click bridging, sweep handling. For the spectrum methods, a common **flow-presence gate**
per side (a vertical run of supra-Otsu pixels of at least 1.5% of the side depth) is on by default. The
published methods assume a spectrum with known flow direction; without the gate every no-flow side of
a bidirectional display would be reported as noise. Ungated variants are reported in the 2-D
experiment. Spectrum methods de-compress the display level with an assumed 40 dB dynamic range.

## Data

1. **Synthetic suite** (`dicom2waveform.synthetic.benchmark_suite`). 12 waveform/modality plans
   (PW: 6, CW: 4, TDI: 2) x 8 SNRs (-3 to 30 dB) x 2 seeds = 192 complete rendered DICOM displays, with
   clutter, mirror, clicks, holes, speckles, overlays, sweep and ECG. Seed 0 = **dev split**, seed 1 =
   **held-out test split**; different seeds also use the two sweep speeds (3.52 / 2.64 ms per column).
   Truth: physical outer band edge (v_max, v_min) and power-weighted mean velocities.
2. **Vendor-traced real images**: 5 vascular Philips iU22 images with the scanner's own envelope trace
   (`Annotated examples` of the MATLAB repository). This is the only pixel-accurate reference on real data.
3. **Real cardiac images without reference**: 17 new examples + 27 from the MATLAB repository (Philips
   iE33 / EPIQ / Affiniti; PW, CW, TDI). Unsupervised: agreement between methods, jitter, runtime,
   visual QC.

## Tuning (what was fitted on what)

Parameters of the published methods are the published values. For the DP, two parameters were
chosen with data:
- `edge_density` (theta): the density level at which the edge is placed, by **leave-one-image-out
  on the 5 vendor-traced images** (theta = 0.5 selected in 3 of 5 folds, 0.6 in 2; held-out
  MAE 2.16 cm/s).
- `normalise_percentile` = 99: normalising the density by its 99th percentile, chosen on the
  **synthetic dev split** together with the vendor images. It leaves the vendor result unchanged and
  removes the failure on sparse (speckled) TDI bands.

The synthetic test split was not used for any choice. Development itself (e.g. finding that the MATLAB
split cost fails on hollow PW/TDI spectra) used individual synthetic cases outside the suite and the
real example images.

## Metrics

Primary endpoint: **normalised envelope MAE** (mean |error| of v_max and v_min over samples with data,
divided by the display velocity span). Secondary: MAE and RMSE in cm/s, bias, over- and
under-estimation fractions, per-beat peak velocity error, VTI error and timing errors (onset, peak,
end), correlation, jitter (S1, S2), failure rate (normalised MAE > 5%), mean-envelope MAE, runtime.
Paired bootstrap CIs of the per-image difference between DP and each method, and Bland-Altman
helpers, are in `dicom2waveform.evaluation.metrics`.

## Main design decisions and why

- **DP outer-edge cost instead of the MATLAB two-region split.** The split assumes a flow region
  filled from the baseline. It collapses on hollow spectra (PW spectral window, TDI bands, a dark
  wall-filter gap in CW). Tested in `tests/test_dp.py`; ablation `dp_matlab_split`.
- **Background subtraction + normalisation of the density**, so that noise is cost-neutral (no
  runaway to the display limit at low SNR) and theta means the same on dense and sparse renderings.
- **No second-order DP term** (Plan section 20): it squares the state space. Curvature is left to
  optional post-smoothing (`post.method`).
- **Header calibration over pixel detection** (MATLAB learnings 2-5): region by unit codes,
  zero row from `ReferencePixelY0`, 0-based coordinates, both signs of `PhysicalDeltaY`.

## Results

See REPORT.md (generated). Summary of the 2026-09-29 run:



**Which estimator is most accurate?** The DP method, on both references:
- **Real images, scanner auto-traces (5 vascular):** MAE 1.92 cm/s, correlation 0.99. The next best are Otsu (3.68) and OMM (4.67). The spectrum methods reach 8-45 cm/s. The DP edge level was tuned on these 5 images; the held-out estimate is 2.16 cm/s, still the best.
- **Synthetic, held-out split:** median normalised error 1.2% of the display span, with the lowest failure rate (12.5%). In the paired comparison it is better than every spectrum method (bootstrap CI excludes 0). Against OMM the advantage is small (CI up to 0.0000), and it is equivalent to Otsu (CI includes 0).

**How does the ranking change with SNR?**
- **High SNR (>= 20 dB):** the image methods (Otsu, OMM, DP) are equivalent (0.4-0.6% of span). GM and MGM are close on synthetic data (0.6-0.8%).
- **Low SNR (<= 3 dB):** DP has the lowest error: 2.2% spectral and 4.0% TDI, against OMM 3.2% / 4.3%, Otsu 6.1% / 21.6% and the spectrum methods 5.6-33%.

The real images have SNR proxies of 15-43 dB. That is at or above the synthetic 20-30 dB range (proxy 15-23 dB). So on these data the practical differences come less from noise than from rendering effects (clutter, mirror bands, clicks, text, sweep, colour maps). The vendor traces separate the methods much more strongly than synthetic high-SNR cases do.

**Does one detector work for spectral Doppler and TDI?** Yes, for DP and OMM. With the same parameters, the extra degradation of TDI at low SNR (Delta) is 0.013 for DP and -0.002 for OMM. It is 0.02-0.16 for Otsu and the spectrum methods (adaptive SNSI 0.02, the others 0.07-0.16). Density normalisation was the component that made one DP parameter set work for dense (vascular/CW) and sparse (TDI) renderings.

**Hypotheses:**
- H1 (spectrum methods competitive at high SNR): supported on synthetic data (GM/MGM within 0.1-0.3% of span of DP). Not supported on real images, where they are 4-23x worse than DP against the vendor traces.
- H2 (image methods degrade more gracefully): supported.
- H3 (global curve tracking beats column-wise at low SNR): supported. At <= 3 dB: A 8.6% > B 6.4% > C 5.1% > D 2.8%. At high SNR, C (segmentation) is marginally better than D.
- H4 (one architecture for spectral and TDI): supported for DP (small Delta, one parameter set).
- H5 (hybrid image-domain method best): supported.

**Which DP components matter?** In order, on synthetic median / low-SNR / vendor MAE:
- the outer-edge cost instead of the MATLAB split: split 4.1% vs 1.1%, vendor 2.23 vs 1.92;
- density normalisation: 2.9% vs 1.1%;
- local density: 3.2% vs 1.1%;
- the calibrated edge level: theta 0.2 gives 4.7 cm/s on vendor traces;
- the wall band, background correction, click bridging and the range-scaled acceleration prior each add 0.4-0.6 percentage points at low SNR.

Soft vs hard membership barely matters once the density is normalised. The weighting choice for the mean envelopes (power, level or membership) changes the mean-envelope error only marginally (0.8-1.1% of span).

**Limitations:**
- The synthetic truth is my own rendering model.
- The only real ground truth is 5 vascular PW images. The edge convention on cardiac CW jets (dense core vs faint tail) is not validated.
- Mirror artefacts are reported as flow.
- Aliasing is flagged but not corrected.
