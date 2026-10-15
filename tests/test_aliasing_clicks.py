import numpy as np
import pytest

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.synthetic import SyntheticSpec, generate


@pytest.mark.parametrize("wf,vp,top,bot", [("two_peaks", 100.0, 85.0, -34.0),  # positive flow past the top
                                           ("systolic", 75.0, 100.0, -40.0)])  # negative jet past the bottom
def test_aliasing_is_unwrapped(wf, vp, top, bot):
    case = generate(SyntheticSpec(waveform=wf, modality="PW", snr_db=20, seed=4, peak_velocity=vp,
                                  v_top=top, v_bottom=bot, alias=True))
    res = extract_waveforms(case.image)
    assert res.flags["aliasing_suspected"] and res.flags["aliasing_corrected"]
    tr = case.truth
    peak_true = np.nanmax(tr["v_max"]) if wf == "two_peaks" else np.nanmin(tr["v_min"])
    peak_est = np.nanmax(res.v_max) if wf == "two_peaks" else np.nanmin(res.v_min)
    limit = top if wf == "two_peaks" else abs(bot)
    assert abs(peak_true) > limit  # the truth really exceeds the display window
    assert abs(peak_est - peak_true) < 0.1 * abs(peak_true)
    # without correction the envelope cannot exceed the display window
    cfg = load_config({"aliasing": {"correct": False}})
    raw = extract_waveforms(case.image, cfg)
    assert np.nanmax(raw.v_max) <= top + 1 and np.nanmin(raw.v_min) >= bot - 1
    span = top - bot
    err_c = np.nanmean(np.abs(res.v_max - tr["v_max"]) + np.abs(res.v_min - tr["v_min"])) / span
    err_u = np.nanmean(np.abs(raw.v_max - tr["v_max"]) + np.abs(raw.v_min - tr["v_min"])) / span
    assert err_c < 0.5 * err_u


@pytest.mark.parametrize("seed", range(4, 12))
def test_aliasing_unwrap_every_beat(seed):
    # Negative jet past the bottom of a hollow PW spectrum. The cut must follow the wrapped band
    # into the top part; it used to jump through the band into the empty spectral window near the
    # baseline in one beat (seeds 4, 10, 11), leaving v_min ~75 cm/s off and a ~120 cm/s v_max spike.
    case = generate(SyntheticSpec(waveform="systolic", modality="PW", snr_db=20, seed=seed, peak_velocity=75.0,
                                  v_top=100.0, v_bottom=-40.0, alias=True))
    res = extract_waveforms(case.image)
    assert res.flags["aliasing_corrected"]
    assert np.nanmax(np.abs(res.v_min - case.truth["v_min"])) < 15
    assert np.nanmax(res.v_max) < 10  # no positive flow in the truth


def test_no_unwrap_without_aliasing(cases):
    res = extract_waveforms(cases["PW"].image)
    assert not res.flags["aliasing_corrected"]


def test_clicks_and_avc():
    case = generate(SyntheticSpec(waveform="systolic", modality="CW", snr_db=20, seed=5, peak_velocity=250,
                                  v_top=150, v_bottom=-450, clicks=True))
    res = extract_waveforms(case.image)
    truth = case.info["t_clicks"]
    assert len(truth) >= 2
    # every true click is reported, within 10 ms
    for tc in truth:
        assert np.min(np.abs(res.t_clicks - tc)) < 0.01
    # AVC: one per beat where the click falls in the 0.2-0.5 s window after R
    assert len(res.t_avc) >= len(truth) - 1
    for a in res.t_avc:
        assert np.min(np.abs(truth - a)) < 0.01
    # bridged: the envelope does not jump to the (saturated, broadband) click
    bridged = extract_waveforms(case.image, load_config({"render": {"click_mode": "bridge"}}))
    i = np.argmin(np.abs(bridged.t - truth[0]))
    assert bridged.v_max[i] < 0.5 * case.params.v_top


def test_click_modes():
    case = generate(SyntheticSpec(waveform="systolic", modality="CW", snr_db=20, seed=5, peak_velocity=250,
                                  v_top=150, v_bottom=-450, clicks=True))
    i = int(np.argmin(np.abs(np.arange(len(case.truth["t"])) * case.params.dt - case.info["t_clicks"][0])))
    follow = extract_waveforms(case.image, load_config({"render": {"click_mode": "follow"}}))
    bridge = extract_waveforms(case.image, load_config({"render": {"click_mode": "bridge"}}))
    assert follow.v_max[i] > bridge.v_max[i] + 20  # 'follow' lets the envelope reach up the click
    off = extract_waveforms(case.image, load_config({"render": {"click_mode": "off"}}))
    assert len(off.t_clicks) == 0
