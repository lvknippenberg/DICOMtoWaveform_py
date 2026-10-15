from pathlib import Path

import pytest

from dicom2waveform.config import PipelineConfig, load_config, write_config

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "standalone" / "dicom2waveform_config.yaml"


def test_default_yaml_matches_code():
    # regenerate with scripts/build_standalone.py when this fails
    assert load_config(DEFAULT_YAML) == PipelineConfig()


def test_write_config_roundtrip(tmp_path):
    cfg = load_config({"render": {"click_mode": "bridge"}, "dp": {"edge_density": 0.4},
                       "estimator": {"params": {"hard_threshold": True}}})
    write_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        load_config({"quality": {"alias_edge_fraction": 0.02}})  # moved to aliasing: (2026-10-15)
