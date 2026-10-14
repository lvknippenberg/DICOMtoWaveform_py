"""Prepare EchoNet-Measurements runs: copy DICOMs (as .dcm) into per-task input folders and write CPU copies of the
official inference scripts (only change: device 'cuda:0' -> 'cpu')."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Github\DICOMtoWaveform_py\standalone")
import dicom2waveform_standalone as S  # noqa: E402

NEW = Path(r"C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples")
OLD = [Path(r"C:\Github\DICOMtoWaveform\DICOM examples"), Path(r"C:\Github\DICOMtoWaveform\EMW examples")]
OUT = Path(r"C:\Claude\Doppler\validation\echonet_in")
MEAS = Path(r"C:\Claude\Doppler\external\echonet_measurements\Measurement")

TR_CW = {"E3Q0C6HS", "L3VHDS0Z"}  # holosystolic ~2.2 m/s jets: TR
MITRAL_PW = {"E3Q0C6H6", "01_10529_", "L3VHDBZ5"}  # matched by substring of the file name

tasks = {"tdi": [], "mitral": [], "tr": [], "aortic_cw": []}
for f in sorted(p for d in [NEW, *OLD] for p in d.iterdir() if p.is_file() and p.suffix.lower() not in (".png", ".txt")):
    img = S.load_dicom(f)
    if img.pixels.shape[:2] != (768, 1024):
        continue  # EchoNet's Doppler models assume 1024x768 frames
    r = S.spectral_region(img)
    res = S.extract_waveforms(img)
    mod = res.flags["modality"]
    if mod == "TDI":
        tasks["tdi"].append(f)
    elif any(k in f.name for k in MITRAL_PW):
        tasks["mitral"].append(f)
    elif any(k in f.name for k in TR_CW):
        tasks["tr"].append(f)
    elif mod == "CW" and r.data_type == 4:
        tasks["aortic_cw"].append(f)  # CW recordings used for aortic valve closure (aortic outflow)
for k, files in tasks.items():
    (OUT / k).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, OUT / k / f"{f.name}.dcm")
    print(k, len(files), [f.name for f in files])

for name in ("inference_Doppler_image.py", "inference_MV_EperA.py"):
    src = (MEAS / name).read_text(encoding="utf8")
    assert 'device       = "cuda:0"' in src
    (MEAS / name.replace(".py", "_cpu.py")).write_text(src.replace('device       = "cuda:0"', 'device       = "cpu"'),
                                                         encoding="utf8")
print("CPU scripts written")
