"""The single-file standalone script is generated from the package (scripts/build_standalone.py):
it must be up to date, import without the package or benchmark dependencies, and give exactly the
package's results."""

import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dicom2waveform import extract_waveforms
from dicom2waveform.config import PipelineConfig, load_config
from dicom2waveform.synthetic import SyntheticSpec, generate

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "standalone" / "dicom2waveform_standalone.py"


def _load_build_script():
    spec = importlib.util.spec_from_file_location("build_standalone", ROOT / "scripts" / "build_standalone.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S():
    spec = importlib.util.spec_from_file_location("dicom2waveform_standalone", STANDALONE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look up their module
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _to_standalone(S, img):
    """A package DicomImage as the standalone script's own DicomImage class."""
    kw = {f.name: getattr(img, f.name) for f in dataclasses.fields(img)}
    kw["regions"] = [S.Region(**{f.name: getattr(r, f.name) for f in dataclasses.fields(r)}) for r in img.regions]
    return S.DicomImage(**kw)


def test_standalone_is_up_to_date():
    # regenerate with: python scripts/build_standalone.py
    py, yml = _load_build_script().build()
    assert STANDALONE.read_text(encoding="utf8") == py
    assert (ROOT / "standalone" / "dicom2waveform_config.yaml").read_text(encoding="utf8") == yml


def test_standalone_imports_without_package_or_benchmark_deps():
    code = (f"import sys; sys.path.insert(0, {str(STANDALONE.parent)!r}); import dicom2waveform_standalone; "
            "bad = [m for m in ('dicom2waveform', 'skimage', 'matplotlib', 'pandas') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=STANDALONE.parent)
    assert r.returncode == 0, r.stdout + r.stderr


CASES = {
    "PW": dict(waveform="two_peaks", modality="PW", snr_db=15, seed=1, peak_velocity=70, v_top=100, v_bottom=-100),
    "CW": dict(waveform="systolic", modality="CW", snr_db=15, seed=2, peak_velocity=180, v_top=50, v_bottom=-250,
               clicks=True),
    "TDI": dict(waveform="tdi", modality="TDI", snr_db=15, seed=3, peak_velocity=10, v_top=15, v_bottom=-15),
    "alias": dict(waveform="systolic", modality="PW", snr_db=20, seed=5, peak_velocity=75, v_top=100, v_bottom=-40,
                  alias=True),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("config", [None, {"dp": {"edge_density": 0.4}, "render": {"click_mode": "bridge"}}])
def test_standalone_matches_package(S, name, config):
    case = generate(SyntheticSpec(**CASES[name]))
    a = extract_waveforms(case.image, config)
    b = S.extract_waveforms(_to_standalone(S, case.image), config)
    for key in ("t", "ecg", "v_max", "v_min", "v_mean", "v_mean_pos", "v_mean_neg", "conf_pos", "conf_neg",
                "t_rpeaks", "t_clicks", "t_avc"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)
    assert a.flags == b.flags


def test_standalone_write_config(tmp_path):
    out = tmp_path / "cfg.yaml"
    r = subprocess.run([sys.executable, str(STANDALONE), "--write-config", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert load_config(out) == PipelineConfig()


def test_standalone_has_dp_only(S):
    case = generate(SyntheticSpec(**CASES["PW"]))
    with pytest.raises(ValueError, match="dp"):
        S.extract_waveforms(_to_standalone(S, case.image), {"estimator": {"method": "otsu"}})
