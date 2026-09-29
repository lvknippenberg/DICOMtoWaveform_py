"""Smoke tests on the real example DICOMs (skipped when the folder is absent)."""
import numpy as np
import pytest

from dicom2waveform import extract_waveforms

from .conftest import example_files

FILES = example_files()


@pytest.mark.skipif(not FILES, reason="example DICOMs not available")
@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_example(path):
    res = extract_waveforms(path)
    assert res.flags["ecg_found"]
    assert len(res.t_rpeaks) >= 2
    hr_header = res.meta["dicom"].get("HeartRate")
    if hr_header:
        hr = 60 / np.median(np.diff(res.t_rpeaks))
        assert abs(hr - float(hr_header)) < 3
    assert np.nanmax(res.v_max) > 0 and np.nanmin(res.v_min) < 0
    assert res.meta["runtime_s"] < 5.0  # ~0.2-0.6 s on an idle machine
