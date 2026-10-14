# External validation scripts

Copies of the scripts in the working folder `C:\Claude\Doppler\validation` used for `docs/VALIDATION.md`
(EchoNet-Measurements on our DICOMs; EchoXFlow subset scoring; head-to-head on EchoXFlow). They use absolute paths
to the working folder, the example DICOMs and the external downloads (`C:\Claude\Doppler\external`), and run in the
conda envs `DopplerUS`, `echonet` (EchoNet, CPU PyTorch 2.3.1) and `echoxflow` (Python 3.13); see the docstring of
each script and the "Reproduce" section of `docs/VALIDATION.md`. Outputs (CSVs, QC figures) stay in the working
folder: they are derived from patient images.
