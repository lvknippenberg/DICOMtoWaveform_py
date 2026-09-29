import os
from pathlib import Path

import pytest

from dicom2waveform.synthetic import SyntheticSpec, generate

EXAMPLES = Path(os.environ.get("D2W_EXAMPLES", r"C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples"))


@pytest.fixture(scope="session")
def cases():
    """One synthetic case per modality at a comfortable SNR."""
    return {
        "PW": generate(SyntheticSpec(waveform="two_peaks", modality="PW", snr_db=20, seed=1,
                                     peak_velocity=70, v_top=100, v_bottom=-100)),
        "CW": generate(SyntheticSpec(waveform="pos_neg", modality="CW", snr_db=20, seed=2,
                                     peak_velocity=250, v_top=150, v_bottom=-450)),
        "TDI": generate(SyntheticSpec(waveform="tdi", modality="TDI", snr_db=20, seed=3,
                                      peak_velocity=10, v_top=15, v_bottom=-15)),
    }


def example_files():
    if not EXAMPLES.is_dir():
        return []
    return sorted(p for p in EXAMPLES.iterdir() if p.is_file())
