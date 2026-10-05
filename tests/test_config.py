from pathlib import Path

from dicom2waveform.config import PipelineConfig, load_config, write_config

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_code():
    # regenerate with scripts/write_default_config.py when this fails
    assert load_config(DEFAULT_YAML) == PipelineConfig()


def test_write_config_roundtrip(tmp_path):
    cfg = load_config({"render": {"click_mode": "bridge"}, "dp": {"edge_density": 0.4}})
    write_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
