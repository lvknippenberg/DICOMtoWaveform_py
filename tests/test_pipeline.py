import numpy as np
import pytest

from dicom2waveform import extract_waveforms, load_config
from dicom2waveform.envelope import BENCHMARK_ORDER, make_estimator
from dicom2waveform.evaluation.metrics import evaluate_signals


@pytest.mark.parametrize("mod", ["PW", "CW", "TDI"])
def test_dp_pipeline_accuracy(cases, mod):
    case = cases[mod]
    res = extract_waveforms(case.image)
    tr = case.truth
    assert len(res.t) == len(tr["t"])
    span = case.params.v_top - case.params.v_bottom
    m = evaluate_signals(res.as_dict(), tr, tr["t"], span, span / case.params.height)
    assert m["env_nmae"] < 0.03, m["env_nmae"]
    assert m["mean_nmae"] < 0.03, m["mean_nmae"]
    # signs
    assert np.nanmin(res.v_max) >= 0 and np.nanmax(res.v_min) <= 0
    assert np.nanmin(res.v_mean_pos) >= 0 and np.nanmax(res.v_mean_neg) <= 0
    # sweep is NaN, and samples are time ordered (truth is NaN at the same samples)
    assert np.array_equal(np.isnan(res.v_max), np.isnan(tr["v_max"]))


def test_rpeaks_from_synthetic_ecg(cases):
    case = cases["PW"]
    res = extract_waveforms(case.image)
    assert len(res.t_rpeaks) == len(case.t_rpeaks)
    assert np.max(np.abs(res.t_rpeaks - case.t_rpeaks)) < 0.01


@pytest.mark.parametrize("name", BENCHMARK_ORDER + ["segmentation"])
def test_every_estimator_runs(cases, name):
    cfg = load_config()
    res = extract_waveforms(cases["PW"].image, cfg, estimator=make_estimator(name))
    assert res.v_max.shape == res.t.shape
    assert np.isnan(res.v_max).sum() == np.isnan(cases["PW"].truth["v_max"]).sum()


def test_config_override_and_unknown_key():
    cfg = load_config({"dp": {"edge_density": 0.4}, "estimator": {"method": "otsu"}})
    assert cfg.dp.edge_density == 0.4 and cfg.estimator.method == "otsu"
    with pytest.raises(KeyError):
        load_config({"dp": {"nonexistent": 1}})
