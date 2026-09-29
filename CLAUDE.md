# CLAUDE.md

Python package `dicom2waveform`: ECG + Doppler envelopes (v_max, v_min, v_mean, v_mean_pos, v_mean_neg)
from rendered spectral / tissue Doppler DICOMs. Port and extension of the MATLAB tool in
`C:\Github\DICOMtoWaveform`.

- **Start with `HANDOFF.md`** (state, decisions and reasons, open points, next steps), then
  `docs/BENCHMARK.md`.
- Python: `C:\Users\p70096100\AppData\Local\miniconda3\envs\DopplerUS\python.exe` (package installed
  editable). Tests: `python -m pytest -q`.
- All parameters live in `dicom2waveform/config.py` (physical units, documented). Do not bury
  parameters in code.
- Working folder outside the repo: `C:\Claude\Doppler` (plans, QC figures, benchmark results).
- **Patient data:** example DICOMs have burned-in identifiers. Never commit DICOMs, full-frame
  renderings, QC figures or result CSVs of real images. Only synthetic/aggregate figures go in
  `docs/results/`.
- Commit when asked; do not push unless asked.
