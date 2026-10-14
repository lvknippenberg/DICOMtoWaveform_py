"""Manual points of interest (blue marks) drawn by the user on the 17 cardiac examples.

Mark pixels = pixels where the annotated PNG differs from the DICOM and are blue. Connected
components are classified as dots (compact) or lines (elongated, vertical: timing marks).
Each dot is converted to (t, v) with the time axis of dicom2waveform_standalone (sweep-aligned).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, r"C:\Github\DICOMtoWaveform_py\standalone")
import dicom2waveform_standalone as S  # noqa: E402

DICOM_DIR = Path(r"C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples")
POI_DIR = DICOM_DIR / "Points of interest"


def read_marks(name: str, res: S.WaveformResult | None = None):
    """Return (dots, lines, img). dots: list of dict(x, y, t, v, area); lines: list of dict(x, t, y0, y1)."""
    img = S.load_dicom(DICOM_DIR / name)
    d = img.pixels if img.pixels.ndim == 3 else np.repeat(img.pixels[..., None], 3, 2)
    a = np.asarray(Image.open(POI_DIR / f"{name}.png").convert("RGB")).astype(int)
    if a.shape != d.shape:
        raise ValueError(f"{name}: annotated PNG {a.shape} != DICOM {d.shape}")
    changed = np.abs(a - d.astype(int)).max(axis=2) > 30
    blue = (a[..., 2] > 150) & (a[..., 2] - a[..., 0] > 60) & (a[..., 2] - a[..., 1] > 20)
    m = ndimage.binary_closing(changed & blue, iterations=1)
    lab, n = ndimage.label(m)
    res = res or S.extract_waveforms(img)
    region = res.spectrogram.region
    inv = np.argsort(res.spectrogram.order)  # raw column -> time index
    dots, lines = [], []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = lab[sl] == i + 1
        area = int(comp.sum())
        if area < 6:
            continue
        h, w = comp.shape
        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
        col = int(round(cx)) - region.x0
        if not (0 <= col < region.width):
            continue
        t = float(res.t[inv[col]])
        if h > 3 * w and h > 12:  # vertical line: timing mark
            lines.append(dict(x=cx, t=t, y0=sl[0].start, y1=sl[0].stop - 1))
        else:
            v = float((cy - region.y0 - region.ref_y0) * region.dy)
            dots.append(dict(x=cx, y=cy, t=t, v=v, area=area))
    return dots, lines, img


if __name__ == "__main__":
    for f in sorted(POI_DIR.glob("*.png")):
        try:
            dots, lines, _ = read_marks(f.stem)
        except Exception as e:  # noqa: BLE001
            print(f.stem, "ERROR", e)
            continue
        print(f"{f.stem}: {len(dots)} dots {[round(d['v'], 1) for d in dots]}, {len(lines)} lines")
