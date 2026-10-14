# External validation (2026-10-13/14)

Three external checks of `standalone/dicom2waveform_standalone.py` (default configuration, unchanged). Scripts:
`scripts/external_validation/` (paths point to the working folder `C:\Claude\Doppler\validation` and the external
data in `C:\Claude\Doppler\external`; neither is in the repository). Full per-case CSVs and QC figures stay in the
working folder (they contain patient-derived data).

Error convention: |estimate| - reference, in cm/s (positive = further from the baseline than the reference).

## Resources

| Resource | What | Licence |
|---|---|---|
| **EchoNet-Measurements** (Sahashi et al., JACC 2025) | DeepLabv3 heatmap models; Doppler: AV/TR/MR/LVOT Vmax, lateral/septal e', E/A. One point per image. `github.com/echonet/measurements` (the paper's URL `.../measurement` is dead). | none stated in the repository |
| **EchoXFlow** (Ahus, arXiv 2605.05447, 2026) | GE Vivid E95, 666 exams; spectral PW/CW Doppler as matrices with EchoPAC clinician annotations (hand-drawn traces, peak marks). 46-exam subset (19.5 GB) used, chosen as the most annotated PW/CW recordings per GB. | CC BY-NC-SA 4.0 |

Environments (conda): `echonet` (Python 3.10, PyTorch 2.3.1 CPU; 2.4 needs `libomp140.x86_64.dll` from a newer VC++
redistributable), `echoxflow` (Python 3.13, the official `echoxflow` loader).

## 1. EchoNet-Measurements on our DICOMs (vs the user's manual marks)

Official folder-mode scripts, only change `device = "cpu"`. Our images match its training geometry (1024x768,
Doppler region rows 342-348). DP is read at the marked beat (peak within +-60 ms, valve-click columns excluded).

| Measure | n marks | EchoNet MAE / bias | DP MAE / bias |
|---|---|---|---|
| TDI e' (lateral model) | 6 | 0.9 / -0.7 | 0.9 / +0.9 |
| TDI e' (septal model) | 6 | 1.2 / -0.5 | 0.9 / +0.9 |
| PW E | 3 | 6.5 / -6.5 | 6.1 / -5.5 |
| PW A | 3 | 4.0 / -3.1 | 4.2 / -4.2 |
| CW jet (TR), `trvmax` | 4 | 33.2 / +33.2 | 15.1 / +15.1 |

Aortic CW of the AVC sets (no reference, 22 unique recordings): DP - EchoNet AV Vmax = -15 +- 32 cm/s, r = 0.85.
Small n; EchoNet chose the user's marked beat in 5 of 22 comparisons.

## 2. EchoXFlow: making it compatible

1. **Format.** Spectral Doppler is stored as a (time x velocity-bin) matrix (PW 96/128 bins, CW 256 bins,
   ~1.9 ms per line) with a velocity axis, not as a rendered screen. It is rendered to the Philips display geometry
   the method was developed on (390 rows, 3.5 ms per column).
2. **Display mapping.** The matrix is stored before the scanner's grey map (bright noise background, the wall
   clutter brightest). A reject/gain emulation is applied: black point = 70th percentile of the image, white point =
   99.5th percentile, fixed a priori (60/80 only as sensitivity).
3. **Annotation baseline offset (dataset bug).** Spectral annotations are stored relative to the *display*
   baseline (`baseline_frac`), the matrix axis uses `spectral_row_baseline_frac`. For CW they differ (0.1 vs 0.5):
   every hand-drawn CW trace starts and ends at exactly -(0.5 - 0.1) x 2 x Nyquist (25/27 traces to 3 decimals)
   instead of 0. Correction: `v = v_annotation + (spectral_row_baseline_frac - baseline_frac) x 2 x Nyquist`
   (zero for PW, where both fractions are equal). Afterwards the CW traces lie on the jet edge. The official
   EchoXFlow renderer uses the same axis and shows the same mismatch.
4. **Loader.** Recordings opened through the Croissant catalogue (the Zarr v2 stores lack intermediate `.zgroup`
   files, so zarr 3 cannot walk them).

## 3. EchoXFlow: DP envelope vs clinician annotations

44 exams, 262 recordings; 126 hand-drawn traces (6,806 samples), 311 peak marks.

| Hand-drawn traces | n | median MAE | mean bias | median rel. MAE | median r | median peak error | peak timing |
|---|---|---|---|---|---|---|---|
| PW (LVOT-type) | 64 | **8.7** | +4.5 | 9.0 % | 0.97 | +7.8 | 14 ms |
| CW (AV-type jets) | 62 | **12.6** | -2.5 | 7.2 % | 0.98 | -15.7 | 18 ms |

| Peak marks | n | mean reference | MAE | bias | median rel. error |
|---|---|---|---|---|---|
| TDI e' lateral / septal / unspecified | 71 / 69 / 16 | 7.8 / 5.9 / 6.2 | 2.5 / 1.9 / 1.7 | +2.0 / +1.9 / +1.2 | 25-30 % |
| TDI RV s' | 8 | 13.7 | 4.0 | -0.2 | 28 % |
| PW A (A-velocity tool peak) | 85 | 79.7 | 11.8 | +1.1 | 9 % |
| PW pulmonary-vein peak (tool) | 8 | 69.9 | 11.9 | -4.4 | 9 % |
| CW AV Vmax | 7 | 111 | 11.5 | -11.5 | 6 % |
| CW TR Vmax | 42 | 267 | 79.9 | -77.1 | 31 % |

Sensitivity (PW / CW trace median MAE): black point P60 10.8 / 12.6, P80 8.4 / 14.5, no display mapping
8.6 / 15.5, native resolution 8.2 / 13.4, aliasing correction off: identical; without the offset correction CW
228.7. The black point mainly shifts the edge bias (it is a reject level), not the shape.

## 4. EchoXFlow: head-to-head with EchoNet-Measurements

Each recording written as a synthetic 1024x768 DICOM in EchoNet's expected geometry (Doppler region rows 342-731,
Ultrasound Region Calibration tags, drawn baseline, same rendering and display mapping); **both methods read the
same file**. Per recording the reference is the clinician's measurement (mean of the marked beats; for traces the
trace peak). EchoNet = its single automatic value. DP @ marks = DP read at the marked beats. Wilcoxon signed-rank
on the per-recording absolute errors.

| Measure | recordings (exams) | reference mean | EchoNet MAE / bias | DP @ marks MAE / bias | DP better | p |
|---|---|---|---|---|---|---|
| e' lateral | 48 (40) | 7.6 | **1.9** / +0.7 | 2.1 / +1.7 | 48 % | 0.84 |
| e' septal | 46 (39) | 6.1 | 2.5 / +2.1 | **1.8** / +1.8 | 52 % | 0.13 |
| PW A | 43 (41) | 79.7 | 15.2 / -15.0 | **10.9** / +0.5 | 58 % | 0.06 |
| LVOT Vmax (PW trace peak) | 36 (33) | 95.3 | 17.5 / +4.6 | **11.3** / +5.4 | 56 % | 0.11 |
| AV Vmax (CW) | 44 (33) | 213 | 38.1 / -21.7 | **26.4** / -18.0 | 66 % | **0.03** |
| TR Vmax (CW) | 39 (27) | 264 | **39.3** / -26.4 | 80.7 / -77.6 | 18 % | **<0.001** |

EchoNet chose a marked beat in 14-48 % of recordings and the wrong side of the baseline in 16 % of AV images.
On the subset where it did choose a marked beat (66 recordings) the picture is the same (e.g. A 21.1 vs 8.8,
AV 31.0 vs 16.5, TR 41.0 vs 81.6). Fully automatic DP (EchoNet's beat, DP's peak within +-150 ms) is worse than
both, because EchoNet's beat choice and side are unreliable on these images.

**Interpretation.** DP is at least as accurate as the published learned model for e', A, LVOT and AV peaks
(significantly better for AV), and clearly worse for TR Vmax, where GE clinicians measure in the faint outer tail
of the jet that the DP edge level does not reach. EchoNet reads TR closer to that convention. The comparison is
not fully symmetric: DP is given the timing of the marked beats (no per-beat peak picking yet), EchoNet chooses its
own beat; and EchoNet sees a re-rendered GE spectrum without the text and B-mode of a real screen (domain shift).

## Conclusions

- The DP envelope transfers to a second vendor and to clinician-drawn envelopes: shape r 0.97-0.98, ~9 % (PW) and
  ~7 % (CW) mean absolute error relative to the peak.
- The edge convention depends on the measurement and the reader: the user marks CW jets at the dense edge (DP
  +15 cm/s), GE clinicians mark TR in the faint tail (DP -77 cm/s) and e' ~2 cm/s inside the outer edge. A single
  global edge level cannot match both; per-measurement settings are needed (HANDOFF open point 1), and the
  EchoXFlow traces and marks are a usable cardiac reference to tune them on (leave-one-exam-out).
- Per-beat peak picking and labelling would remove the asymmetry in the head-to-head and make DP fully automatic.

## Reproduce

```bash
# working folder C:\Claude\Doppler\validation (copies of scripts/external_validation/)
envs/DopplerUS/python.exe echonet_prepare.py            # inputs + CPU copies of the EchoNet scripts
(echonet env, cwd = external/echonet_measurements/Measurement) inference_*_cpu.py --folders ...
envs/DopplerUS/python.exe score_echonet.py               # section 1
envs/DopplerUS/python.exe echoxflow_select.py 19.5 && envs/DopplerUS/python.exe echoxflow_download.py
envs/echoxflow/python.exe echoxflow_inventory.py
bash run_echoxflow_all.sh                                # section 3 (+ sensitivity runs)
envs/echoxflow/python.exe echoxflow_to_dicom.py          # section 4: synthetic DICOMs
(echonet env) inference_*_cpu.py on echoxflow_dicom/<task>
envs/DopplerUS/python.exe score_echoxflow_vs_echonet.py
```
