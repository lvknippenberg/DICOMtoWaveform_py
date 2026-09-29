"""DICOM reading and ultrasound region calibration.

The spectral display is located and calibrated from the DICOM Ultrasound
Region Calibration module (SequenceOfUltrasoundRegions) rather than from the
pixels: the header states the region box, the time and velocity increment per
pixel and the zero-velocity row exactly.

Region coordinates in DICOM are 0-based and inclusive, which matches NumPy
indexing directly (MATLAB needed +1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

UNIT_CM = 3
UNIT_SECONDS = 4
UNIT_CM_PER_S = 7

# RegionDataType codes (DICOM PS3.3 C.8.5.5.1.2)
DATA_TYPE_NAMES = {
    1: "tissue",
    2: "color_flow",
    3: "pw_spectral",
    4: "cw_spectral",
    5: "doppler_mean_trace",
    6: "doppler_mode_trace",
    7: "doppler_max_trace",
    8: "volume_trace",
    10: "ecg_trace",
    11: "pulse_trace",
    12: "phonocardiogram_trace",
}


@dataclass
class Region:
    """One item of SequenceOfUltrasoundRegions."""

    x0: int
    y0: int
    x1: int  # inclusive
    y1: int  # inclusive
    dx: float  # physical units per pixel in x
    dy: float  # physical units per pixel in y (downwards)
    units_x: int
    units_y: int
    data_type: int
    spatial_format: int
    ref_x0: int | None = None
    ref_y0: int | None = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def is_spectral(self) -> bool:
        return self.units_x == UNIT_SECONDS and self.units_y == UNIT_CM_PER_S

    @property
    def is_bmode(self) -> bool:
        return self.units_x == UNIT_CM and self.units_y == UNIT_CM

    @property
    def data_type_name(self) -> str:
        return DATA_TYPE_NAMES.get(self.data_type, f"type_{self.data_type}")


@dataclass
class DicomImage:
    path: str
    pixels: np.ndarray  # (H, W) uint8 or (H, W, 3) uint8, RGB
    photometric: str
    regions: list[Region]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rgb(self) -> bool:
        return self.pixels.ndim == 3


def _get(ds: Any, name: str, default: Any = None) -> Any:
    try:
        val = getattr(ds, name)
    except AttributeError:
        return default
    return default if val is None or val == "" else val


def _parse_regions(ds: Any) -> list[Region]:
    regions = []
    for r in _get(ds, "SequenceOfUltrasoundRegions", []) or []:
        regions.append(
            Region(
                x0=int(r.RegionLocationMinX0),
                y0=int(r.RegionLocationMinY0),
                x1=int(r.RegionLocationMaxX1),
                y1=int(r.RegionLocationMaxY1),
                dx=float(_get(r, "PhysicalDeltaX", np.nan)),
                dy=float(_get(r, "PhysicalDeltaY", np.nan)),
                units_x=int(_get(r, "PhysicalUnitsXDirection", 0)),
                units_y=int(_get(r, "PhysicalUnitsYDirection", 0)),
                data_type=int(_get(r, "RegionDataType", 0)),
                spatial_format=int(_get(r, "RegionSpatialFormat", 0)),
                ref_x0=None if _get(r, "ReferencePixelX0") is None else int(r.ReferencePixelX0),
                ref_y0=None if _get(r, "ReferencePixelY0") is None else int(r.ReferencePixelY0),
            )
        )
    return regions


def _to_uint8_rgb_or_grey(ds: Any, arr: np.ndarray) -> tuple[np.ndarray, str]:
    photometric = str(_get(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric.startswith("YBR"):
        from pydicom.pixels import convert_color_space

        arr = convert_color_space(arr, photometric, "RGB")
        photometric = "RGB"
    elif photometric == "PALETTE COLOR":
        from pydicom.pixels import apply_color_lut

        arr = apply_color_lut(arr, ds)
        if arr.dtype != np.uint8:
            arr = (arr / 257).astype(np.uint8)
        photometric = "RGB"
    elif photometric == "MONOCHROME1":
        arr = arr.max() - arr
        photometric = "MONOCHROME2"
    if arr.dtype != np.uint8:
        bits = int(_get(ds, "BitsStored", 8))
        arr = np.clip(arr.astype(np.float64) / (2**bits - 1) * 255, 0, 255).astype(np.uint8)
    # RGB images whose three channels are identical are still handled as RGB,
    # so that overlay colour logic is uniform; nothing is lost.
    return arr, photometric


META_TAGS = [
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "Modality",
    "Rows", "Columns", "PhotometricInterpretation", "BitsAllocated", "BitsStored",
    "NumberOfFrames", "FrameTime", "CineRate", "RescaleSlope", "RescaleIntercept",
    "WindowCenter", "WindowWidth", "PixelSpacing", "HeartRate",
    "TransducerData", "ProcessingFunction", "SeriesDescription", "StudyDescription",
    "SOPInstanceUID",
]


def load_dicom(path: str | Path, frame: int = 0) -> DicomImage:
    """Read a DICOM and return its pixel array, regions and metadata.

    Only non-identifying technical metadata is recorded in `meta`.
    """
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    n_frames = int(_get(ds, "NumberOfFrames", 1) or 1)
    photometric = str(_get(ds, "PhotometricInterpretation", ""))
    if n_frames > 1:
        arr = arr[frame]
    elif arr.ndim == 3 and photometric == "MONOCHROME2" and arr.shape[-1] != 3:
        arr = arr[frame]
    arr, photometric = _to_uint8_rgb_or_grey(ds, arr)

    meta: dict[str, Any] = {}
    for tag in META_TAGS:
        val = _get(ds, tag)
        if val is not None:
            meta[tag] = val if isinstance(val, (int, float, str)) else str(val)
    meta["TransferSyntax"] = str(ds.file_meta.TransferSyntaxUID.name) if hasattr(ds, "file_meta") else ""
    meta["HasWaveformSequence"] = "WaveformSequence" in ds
    return DicomImage(path=str(path), pixels=arr, photometric=photometric,
                      regions=_parse_regions(ds), meta=meta)


class RegionError(ValueError):
    pass


def spectral_region(img: DicomImage) -> Region:
    """The spectral Doppler region: seconds horizontally, cm/s vertically.

    Selected by physical unit codes, never by item order (the order varies).
    """
    cands = [r for r in img.regions if r.is_spectral]
    if not cands:
        raise RegionError("No spectral Doppler region (s vs cm/s) in SequenceOfUltrasoundRegions")
    if len(cands) > 1:
        # Prefer the largest one (e.g. a duplicated/trace region)
        cands.sort(key=lambda r: r.width * r.height, reverse=True)
    r = cands[0]
    if not np.isfinite(r.dy) or r.dy == 0:
        raise RegionError(f"PhysicalDeltaY must be finite and non-zero, got {r.dy}")
    if not np.isfinite(r.dx) or r.dx <= 0:
        raise RegionError(f"PhysicalDeltaX must be finite and positive, got {r.dx}")
    return r


def bmode_region_above(img: DicomImage, spec: Region) -> Region | None:
    """The B-mode region directly above the spectral region (ECG runs along its lower edge)."""
    cands = [r for r in img.regions if r.is_bmode and r.y1 <= spec.y0]
    if not cands:
        return None
    return max(cands, key=lambda r: r.y1)
