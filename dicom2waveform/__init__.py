"""dicom2waveform: ECG and Doppler envelopes from spectral / tissue Doppler DICOM images.

>>> from dicom2waveform import extract_waveforms
>>> res = extract_waveforms("path/to/dicom")
>>> res.t, res.ecg, res.v_mean, res.v_mean_pos, res.v_mean_neg, res.v_max, res.v_min
"""

__version__ = "0.1.0"

from .config import PipelineConfig, load_config  # noqa: E402
from .pipeline import WaveformResult, extract_waveforms  # noqa: E402

__all__ = ["extract_waveforms", "WaveformResult", "PipelineConfig", "load_config", "__version__"]
