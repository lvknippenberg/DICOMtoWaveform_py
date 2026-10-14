"""Write EchoXFlow spectral recordings as synthetic single-frame ultrasound DICOMs in the screen geometry EchoNet-
Measurements expects (1024x768 RGB, Doppler region rows 342..731, x from 42, Ultrasound Region Calibration tags),
so that EchoNet and dicom2waveform_standalone read the *same* image.

The spectrum is rendered exactly as in score_echoxflow.py (display mapping black P70, 390 rows, 3.5 ms/column);
a thin grey baseline is drawn at zero velocity, as scanners do. At most 902 columns (3.16 s) fit the region; longer
recordings are windowed around their annotations.

Per recording a JSON sidecar stores the time of column 0, dt, and the clinician references (offset-corrected,
cm/s): points (label, t, v) and trace peaks (label 'trace', t of the peak, peak v, start/end time).
Folders by EchoNet task: tdi_lat, tdi_med (lateral / septal e'; unspecified e' in both), mitral (A tool, E, A),
lvot (PW hand-drawn traces), tr (TR Vmax), av (AV Vmax and CW hand-drawn traces).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, r"C:\Claude\Doppler\validation")
import score_echoxflow as SX  # noqa: E402

OUT = Path(r"C:\Claude\Doppler\validation\echoxflow_dicom")
X0, Y0, NCOLS = 42, 342, 902
US_IMAGE = "1.2.840.10008.5.1.4.1.1.6.1"


def write_dicom(path, img, dy, zero_row, dt, data_type):
    H, W = img.shape
    frame = np.zeros((768, 1024, 3), np.uint8)
    g = np.clip(np.round(img), 0, 255).astype(np.uint8)
    frame[Y0:Y0 + H, X0:X0 + W] = g[..., None]
    z = int(round(zero_row))
    frame[Y0 + z, X0:X0 + W] = 150  # drawn baseline
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = US_IMAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = US_IMAGE, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Manufacturer = "US", "EchoXFlow-render"
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PlanarConfiguration = 768, 1024, 3, 0
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    r = Dataset()
    r.RegionSpatialFormat, r.RegionDataType, r.RegionFlags = 3, data_type, 0
    r.RegionLocationMinX0, r.RegionLocationMinY0 = X0, Y0
    r.RegionLocationMaxX1, r.RegionLocationMaxY1 = X0 + W - 1, Y0 + H - 1
    r.ReferencePixelX0, r.ReferencePixelY0 = 0, z
    r.PhysicalUnitsXDirection, r.PhysicalUnitsYDirection = 4, 7
    r.PhysicalDeltaX, r.PhysicalDeltaY = float(dt), float(dy)
    ds.SequenceOfUltrasoundRegions = [r]
    ds.PixelData = frame.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def tasks_for(labels, kind):
    t = set()
    cw = "continuous" in kind
    for lab in labels:
        if lab == "MV Eprime Lateral Velocity":
            t.add("tdi_lat")
        elif lab == "MV Eprime Septal Velocity":
            t.add("tdi_med")
        elif lab == "MV Eprime Velocity":
            t.update(("tdi_lat", "tdi_med"))
        elif lab in ("A Velocity", "MV E Velocity", "MV A Velocity"):
            t.add("mitral")
        elif lab == "Manual":
            t.add("av" if cw else "lvot")
        elif lab == "TR Vmax":
            t.add("tr")
        elif lab == "AV Vmax":
            t.add("av")
    return sorted(t)


def main():
    n = 0
    for (rid, kind), g in SX.ann.groupby(["recording", "kind"]):
        tasks = tasks_for(set(g["short"]), kind)
        if not tasks:
            continue
        st = SX.open_recording(SX.recs[rid], root=SX.ROOT)
        lv = st.load_array(f"data/{kind}").astype(np.float32)
        t = st.load_array(f"timestamps/{kind}").astype(float)
        sm = st.spectral_metadata(kind)
        img, dy, z, dt, tc = SX.render(lv, t, np.asarray(sm.row_velocity_mps, float))
        off = SX.annotation_offset_mps(sm)
        refs = []
        for _, a in g.iterrows():
            if a["short"] not in SX.POINT_LABELS and a["short"] != "Manual":
                continue
            p = st.load_array(a["path"]).astype(float)
            p[:, 1] = (p[:, 1] + off) * 100.0
            k = int(np.argmax(np.abs(p[:, 1])))
            if a["short"] == "Manual":
                if np.abs(p[:, 1]).max() < 10:
                    continue
                refs.append(dict(label="trace", t=p[k, 0], v=p[k, 1], t_start=p[:, 0].min(), t_end=p[:, 0].max()))
            else:
                refs.append(dict(label=SX.POINT_LABELS[a["short"]], t=p[k, 0], v=p[k, 1]))
        if not refs:
            continue
        # window of NCOLS columns around the annotations
        W = img.shape[1]
        c0 = 0
        if W > NCOLS:
            mid = np.mean([r["t"] for r in refs])
            c0 = int(np.clip(np.searchsorted(tc, mid) - NCOLS // 2, 0, W - NCOLS))
        win = img[:, c0:c0 + NCOLS]
        t0 = float(tc[c0])
        refs = [r for r in refs if t0 <= r["t"] <= t0 + win.shape[1] * dt]
        if not refs:
            continue
        dtype = 4 if "continuous" in kind else 3
        side = dict(recording=rid, exam=g["exam"].iloc[0], kind=kind, t0=t0, dt=dt, tasks=tasks, refs=refs,
                    nyquist_cm_s=float(sm.nyquist_limit_mps) * 100)
        for task in tasks:
            d = OUT / task
            d.mkdir(parents=True, exist_ok=True)
            write_dicom(d / f"{rid}.dcm", win, dy, z, dt, dtype)
            (d / f"{rid}.json").write_text(json.dumps(side, default=float), encoding="utf8")
        n += 1
    for d in sorted(OUT.iterdir()):
        print(d.name, len(list(d.glob("*.dcm"))))
    print(n, "recordings exported")


if __name__ == "__main__":
    main()
