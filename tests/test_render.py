import numpy as np

from dicom2waveform.config import RenderConfig
from dicom2waveform.io import spectral_region
from dicom2waveform.render import build_spectrogram, find_text, sweep_order


def test_sweep_order_circular():
    sweep = np.zeros(20, bool)
    sweep[[18, 19, 0, 1]] = True  # sweep wraps around the right edge
    order = sweep_order(sweep, True)
    assert order[0] == 18


def test_text_is_masked_but_saturated_flow_is_not():
    level = np.zeros((60, 60))
    level[5:10, 5:8] = 255  # small glyph on black
    level[30:60, 20:40] = 200
    level[40:50, 25:35] = 255  # saturated core inside bright flow
    m = find_text(level, np.zeros_like(level, bool), RenderConfig())
    assert m[7, 6] and not m[45, 30]


def test_colour_map_detected_for_tdi(cases):
    img = cases["TDI"].image
    spec = build_spectrogram(img, spectral_region(img), RenderConfig())
    assert spec.colormap == "colour" and spec.modality == "TDI"
    spec_pw = build_spectrogram(cases["PW"].image, spectral_region(cases["PW"].image), RenderConfig())
    assert spec_pw.colormap == "grey" and spec_pw.sweep.sum() >= 8
