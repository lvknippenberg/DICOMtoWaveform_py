import dataclasses

import numpy as np

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.plot import normalise_brightness
from dicom2waveform.synthetic import SyntheticSpec, generate


def _signal_median(spec, rgb):
    use = spec.valid & ~spec.sweep[None, :] & ~spec.overlay
    raw = spec.rgb.astype(float).mean(axis=2)[use]
    sig = raw > np.percentile(raw, 50) + 0.1 * (np.percentile(raw, 99.9) - np.percentile(raw, 50))
    return float(np.median(rgb.astype(float).mean(axis=2)[use][sig]))


def test_normalise_brightness_dark_and_bright_agree():
    case = generate(SyntheticSpec(waveform="two_peaks", modality="PW", snr_db=20, seed=3))
    cfg = load_config(None)
    cfg.keep_diagnostics = True
    medians = []
    for gain in (0.3, 1.0):
        px = np.clip(case.image.pixels.astype(float) * gain, 0, 255).astype(np.uint8)
        res = extract_waveforms(dataclasses.replace(case.image, pixels=px), cfg)
        spec = res.diagnostics["display_spectrogram"]
        out = normalise_brightness(spec)
        assert out.shape == spec.rgb.shape and out.dtype == np.uint8
        # overlays are left as drawn
        assert np.array_equal(out[spec.overlay], spec.rgb[spec.overlay])
        medians.append(_signal_median(spec, out))
    assert all(abs(m - 128) < 15 for m in medians), medians
