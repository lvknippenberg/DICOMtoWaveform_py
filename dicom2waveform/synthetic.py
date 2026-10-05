"""Synthetic spectral / tissue Doppler DICOM images with known ground truth.

A synthetic case is a complete rendered display wrapped in a `DicomImage`
(with an ultrasound region calibration), so it runs through exactly the same
pipeline as a real file: overlay detection, colour -> level, sweep handling,
ECG extraction and envelope estimation.

Rendering model (per pixel, velocity v, time t):

    P(v, t) = N + sum_c S_c(v, t)                  expected power
    S_c     = A * shape(v) * roll(|v| - u_c) * roll(l_c - |v|)
    P_pix   = P * Gamma(k, 1/k) speckle (optionally spatially correlated)
    level   = 255 * (10 log10 P_pix - floor_dB) / DR,  clipped to 0..255

with u_c(t), l_c(t) the outer and inner edges of flow component c, `roll` an
erfc roll-off modelling intrinsic spectral broadening, and A set by the SNR:
SNR_dB = 10 log10(mean in-band signal power / noise power N).

Truth:
    v_max(t) = u of the positive component (0 if none)
    v_min(t) = -u of the negative component (0 if none)
    v_mean*  = expected-signal-power-weighted mean velocity (no noise, no
               broadening beyond the band), i.e. the physical mean velocity.

These images are for debugging, sensitivity analysis and ranking under
controlled conditions; they do not establish clinical performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import zlib

import numpy as np
from scipy import ndimage, special

from .io import DicomImage, Region

COPPER_LUT = np.array([
    (0, 0, 0), (26, 7, 0), (43, 15, 11), (53, 23, 17), (64, 32, 23), (74, 43, 31), (84, 51, 40),
    (95, 59, 47), (104, 67, 53), (112, 74, 59), (120, 82, 64), (130, 89, 71), (137, 95, 76),
    (145, 101, 81), (152, 108, 86), (158, 112, 90), (163, 118, 95)], np.uint8)
ECG_GREEN = (69, 249, 69)
BASE_ORANGE = (157, 89, 29)

WAVEFORMS = ("gaussian", "systolic", "two_peaks", "pos_neg", "crossing", "continuous", "tdi")


@dataclass
class SyntheticSpec:
    """Parameters of one synthetic case."""

    waveform: str = "systolic"
    modality: str = "PW"  # 'PW', 'CW' or 'TDI'
    snr_db: float = 20.0
    seed: int = 0
    heart_rate: float = 70.0
    duration_s: float = 3.2
    dt: float = 0.003521
    height: int = 385
    v_top: float = 100.0  # velocity at the top row (cm/s)
    v_bottom: float = -100.0  # velocity at the bottom row
    peak_velocity: float = 70.0  # peak |v| of the main component
    fill: str | None = None  # 'filled' (CW), 'hollow' (PW), 'band' (TDI); None -> by modality
    broadening: float = 0.04  # roll-off width as a fraction of the edge velocity
    dynamic_range_db: float = 40.0
    reject_db: float = 3.0  # display floor above the mean noise level (dB)
    speckle_shape: float = 1.0  # Gamma shape (1 = exponential / single-look)
    speckle_corr_px: float = 0.7  # spatial correlation of the speckle (px sigma)
    colormap: str | None = None  # 'grey' or 'copper'; None -> by modality
    # artefacts
    sweep: bool = True
    overlays: bool = True
    ecg: bool = True
    clutter_db: float = 0.0  # extra power near the baseline (dB above noise), 0 = none
    clutter_cm_s: float = 5.0
    mirror_db: float = 0.0  # mirror of the dominant component on the other side (dB below), 0 = none
    alias: bool = False  # wrap velocities beyond the display limit (PW)
    clicks: bool = False  # valve closure clicks (saturated vertical lines)
    salt_pepper: float = 0.0  # fraction of pixels
    gradient_db: float = 0.0  # background gradient across the display (dB)
    holes: int = 0  # number of signal dropouts inside the band
    missing_columns: int = 0  # number of dropped columns (noise only)
    speckles: int = 0  # random bright blobs

    def resolved(self) -> "SyntheticSpec":
        s = SyntheticSpec(**vars(self))
        if s.fill is None:
            s.fill = {"CW": "filled", "PW": "hollow", "TDI": "band"}[s.modality]
        if s.colormap is None:
            s.colormap = "copper" if s.modality == "TDI" else "grey"
        return s


@dataclass
class SyntheticCase:
    image: DicomImage
    params: SyntheticSpec
    truth: dict[str, np.ndarray]  # time order, per column
    t_rpeaks: np.ndarray
    power: np.ndarray | None = None  # expected signal power (H, W), time order
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Waveforms: per component, (sign, outer edge u(t) >= 0, inner edge l(t))
# ---------------------------------------------------------------------------

def _pulse(phase, centre, rise, fall):
    """Asymmetric smooth pulse (0..1) peaking at `centre` (all in cycle fractions)."""
    x = phase - centre
    return np.where(x < 0, np.exp(-0.5 * (x / rise) ** 2), np.exp(-0.5 * (x / fall) ** 2))


def _components(p: SyntheticSpec, t: np.ndarray, rng: np.random.Generator):
    rr = 60.0 / p.heart_rate
    jitter = 1 + 0.03 * rng.standard_normal()
    phase = ((t / (rr * jitter)) + rng.uniform(0, 1)) % 1.0
    Vp = p.peak_velocity
    comps = []
    w = p.waveform
    if w == "gaussian":
        comps.append((+1, Vp * _pulse(phase, 0.25, 0.06, 0.06)))
    elif w == "systolic":
        comps.append((-1, Vp * _pulse(phase, 0.18, 0.035, 0.09)))
    elif w == "two_peaks":
        comps.append((+1, Vp * _pulse(phase, 0.45, 0.04, 0.07) + 0.6 * Vp * _pulse(phase, 0.85, 0.03, 0.04)))
    elif w == "pos_neg":
        comps.append((-1, Vp * _pulse(phase, 0.2, 0.04, 0.08)))
        comps.append((+1, 0.5 * Vp * _pulse(phase, 0.55, 0.05, 0.1)))
    elif w == "crossing":
        v = Vp * np.sin(2 * np.pi * phase) * (0.7 + 0.3 * np.cos(2 * np.pi * phase))
        comps.append((+1, np.clip(v, 0, None)))
        comps.append((-1, np.clip(-v, 0, None)))
    elif w == "continuous":
        comps.append((+1, 0.25 * Vp + 0.75 * Vp * _pulse(phase, 0.2, 0.04, 0.12)))
    elif w == "tdi":
        comps.append((+1, Vp * _pulse(phase, 0.22, 0.04, 0.06)))  # s'
        comps.append((-1, 1.2 * Vp * _pulse(phase, 0.55, 0.03, 0.05) + 0.8 * Vp * _pulse(phase, 0.88, 0.025, 0.04)))  # e', a'
    else:
        raise ValueError(f"unknown waveform '{w}'")
    # tiny envelopes are "no flow"
    out = []
    for sign, u in comps:
        u = np.where(u < 0.03 * Vp, 0.0, u)
        out.append((sign, u))
    return out, phase


def _ecg_template(t: np.ndarray, r_times: np.ndarray) -> np.ndarray:
    e = np.zeros_like(t)
    for r in r_times:
        x = t - r
        e += (1.0 * np.exp(-0.5 * (x / 0.010) ** 2) - 0.25 * np.exp(-0.5 * ((x - 0.025) / 0.012) ** 2)
              - 0.15 * np.exp(-0.5 * ((x + 0.02) / 0.01) ** 2) + 0.3 * np.exp(-0.5 * ((x - 0.28) / 0.05) ** 2)
              + 0.12 * np.exp(-0.5 * ((x + 0.16) / 0.03) ** 2))
    return e


def generate(spec: SyntheticSpec) -> SyntheticCase:
    p = spec.resolved()
    rng = np.random.default_rng(p.seed)
    W = int(round(p.duration_s / p.dt))
    H = p.height
    t = np.arange(W) * p.dt
    dy = -(p.v_top - p.v_bottom) / (H - 1)
    zero_row = int(round(p.v_top / -dy))
    v_rows = (np.arange(H) - zero_row) * dy  # velocity of each display row
    comps, phase = _components(p, t, rng)

    # --- expected signal power --------------------------------------------
    # Computed on an extended grid of three display spans (rows -H..2H-1), so
    # that flow beyond the display limits exists: it is dropped (no aliasing)
    # or folded back modulo the display span (PW aliasing).
    rows_ext = np.arange(-H, 2 * H)
    v_ext = (rows_ext - zero_row) * dy
    Nnoise = 1.0
    A = 10 ** (p.snr_db / 10.0) * Nnoise
    S = np.zeros((3 * H, W))
    truth = {"v_max": np.zeros(W), "v_min": np.zeros(W)}
    sw_pos = np.zeros(W)
    swv_pos = np.zeros(W)
    sw_neg = np.zeros(W)
    swv_neg = np.zeros(W)
    for sign, u in comps:
        if p.fill == "filled":
            l = np.zeros(W)
        elif p.fill == "hollow":
            l = 0.35 * u
        else:  # band (TDI): narrow band just inside the edge
            l = np.clip(u - np.maximum(0.25 * u, 0.6), 0, None)
        speed = sign * v_ext[:, None]  # >0 on this component's side
        sigma_u = np.maximum(p.broadening * u, 0.6 * abs(dy))[None, :]
        roll_out = 0.5 * special.erfc((speed - u[None, :]) / (np.sqrt(2) * sigma_u))
        roll_in = 0.5 * special.erfc((l[None, :] - speed) / (np.sqrt(2) * np.maximum(sigma_u, abs(dy))))
        # power higher towards the band centre / lower velocities (typical spectra)
        rel = np.clip((speed - l[None, :]) / np.maximum(u - l, 1e-6)[None, :], 0, 1)
        shape = 1.0 - 0.5 * rel
        comp = A * shape * roll_out * roll_in * (speed > 0) * (u[None, :] > 0)
        S += comp
        # truth
        if sign > 0:
            truth["v_max"] = np.maximum(truth["v_max"], u)
        else:
            truth["v_min"] = np.minimum(truth["v_min"], -u)
        inband = (speed >= l[None, :]) & (speed <= u[None, :]) & (u[None, :] > 0)
        wgt = np.where(inband, A * shape, 0.0)
        if sign > 0:
            sw_pos += wgt.sum(0)
            swv_pos += (wgt * v_ext[:, None]).sum(0)
        else:
            sw_neg += wgt.sum(0)
            swv_neg += (wgt * v_ext[:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        truth["v_mean_pos"] = np.where(sw_pos > 0, swv_pos / sw_pos, 0.0)
        truth["v_mean_neg"] = np.where(sw_neg > 0, swv_neg / sw_neg, 0.0)
        tot = sw_pos + sw_neg
        truth["v_mean"] = np.where(tot > 0, (swv_pos + swv_neg) / tot, 0.0)

    if p.alias:
        # PW aliasing: the displayed spectrum is periodic with the display span
        # (H rows); flow beyond one limit re-enters at the other
        S = S[:H] + S[H:2 * H] + S[2 * H:]
    else:
        S = S[H:2 * H]
    if p.mirror_db > 0:
        # mirror about the zero row
        mir = np.zeros_like(S)
        for r in range(H):
            r2 = 2 * zero_row - r
            if 0 <= r2 < H:
                mir[r2] += S[r]
        S = S + mir * 10 ** (-p.mirror_db / 10)
    if p.clutter_db > 0:
        prof = np.exp(-0.5 * (v_rows / max(p.clutter_cm_s, abs(dy))) ** 2)
        S = S + (10 ** (p.clutter_db / 10) * Nnoise) * prof[:, None] * (0.6 + 0.4 * rng.random(W))[None, :]

    # --- noise, speckle ------------------------------------------------------
    P = S + Nnoise
    if p.gradient_db:
        g = 10 ** (np.linspace(0, p.gradient_db, H) / 10)[:, None] * np.ones((1, W))
        P = P * g
    k = p.speckle_shape
    X = rng.gamma(k, 1.0 / k, size=(H, W))
    if p.speckle_corr_px > 0:
        # correlated speckle with the same mean: smooth a Gaussian field and map
        # it through the Gamma quantile function
        Z = ndimage.gaussian_filter(rng.standard_normal((H, W)), p.speckle_corr_px)
        Z = (Z - Z.mean()) / (Z.std() + 1e-12)
        from scipy import stats
        X = stats.gamma.ppf(np.clip(stats.norm.cdf(Z), 1e-9, 1 - 1e-9), k, scale=1.0 / k)
    Ppix = P * X
    if p.holes:
        for _ in range(p.holes):
            cy, cx = rng.integers(0, H), rng.integers(0, W)
            ry, rx = rng.integers(3, 12), rng.integers(3, 12)
            yy, xx = np.ogrid[:H, :W]
            m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            Ppix[m] = Nnoise * X[m]
    if p.missing_columns:
        cols = rng.choice(W, size=min(p.missing_columns, W), replace=False)
        Ppix[:, cols] = Nnoise * X[:, cols]
    floor_db = 10 * np.log10(Nnoise) + p.reject_db
    level = np.clip((10 * np.log10(Ppix) - floor_db) / p.dynamic_range_db * 255.0, 0, 255)
    if p.salt_pepper:
        m = rng.random((H, W)) < p.salt_pepper
        level[m] = rng.choice([0.0, 255.0], size=m.sum())
    if p.speckles:
        for _ in range(p.speckles):
            cy, cx = rng.integers(0, H), rng.integers(0, W)
            level[max(0, cy - 2):cy + 2, max(0, cx - 1):cx + 1] = 230
    # --- ECG ----------------------------------------------------------------
    rr = 60.0 / p.heart_rate
    r_times = np.arange(rng.uniform(0.1, rr), p.duration_s, rr)
    ecg = _ecg_template(t, r_times)

    # valve clicks ~20 ms wide at aortic valve closure (R + 0.38 RR, i.e. within
    # the 0.2-0.5 s AVC window at 50-110 bpm)
    t_clicks = np.array([])
    if p.clicks:
        t_clicks = r_times + 0.38 * rr
        t_clicks = t_clicks[t_clicks < p.duration_s - 0.02]
        n = max(2, int(round(0.02 / p.dt)))
        for tc in t_clicks:
            c = int(round(tc / p.dt)) - n // 2
            level[:, max(0, c):c + n] = np.maximum(level[:, max(0, c):c + n], 245)

    # --- sweep: the first n samples (time order) are being overwritten --------
    n_sweep = 8 if p.sweep else 0
    time_valid = np.ones(W, bool)
    time_valid[:n_sweep] = False

    # --- render the display (time order), then place in the circular buffer ---
    if p.colormap == "copper":
        idx = np.clip(np.round(level / 255.0 * (len(COPPER_LUT) - 1)), 0, len(COPPER_LUT) - 1).astype(int)
        rgb = COPPER_LUT[idx]
    else:
        g = np.round(level).astype(np.uint8)
        rgb = np.repeat(g[..., None], 3, axis=2)
    rgb[:, ~time_valid] = 0

    top_pad = 70  # rows above the spectral region: ECG strip / B-mode bottom
    left_pad = 20
    img = np.zeros((H + top_pad + 10, W + left_pad + 30, 3), np.uint8)
    y0, x0 = top_pad, left_pad
    img[y0:y0 + H, x0:x0 + W] = rgb
    ecg_colour = ECG_GREEN
    if p.ecg:
        # ECG polyline around the region top edge, amplitude ~35 px
        e_rows = (y0 - 8 - 35 * ecg / max(ecg.max(), 1e-6)).round().astype(int)
        for c in range(W):
            if not time_valid[c]:
                continue
            r0 = e_rows[c]
            r1 = e_rows[c - 1] if c > 0 and time_valid[c - 1] else r0
            lo, hi = min(r0, r1), max(r0, r1)
            img[lo:hi + 2, x0 + c] = ecg_colour
    if p.overlays:
        base_col = BASE_ORANGE
        img[y0 + zero_row - 2:y0 + zero_row + 1, x0:x0 + W][:, time_valid] = base_col
        # dotted time markers every second
        for tm in np.arange(1.0, p.duration_s, 1.0):
            c = int(tm / p.dt)
            if c < W and time_valid[c]:
                img[y0:y0 + H:25, x0 + c] = base_col
        # a white text block top-left and tick marks at the bottom
        for i in range(4):
            img[y0 + 10 + 18 * i:y0 + 20 + 18 * i, x0 + 2:x0 + 40:2] = 255
        img[y0 + H - 4:y0 + H - 1, x0:x0 + W:28] = 255
    # circular buffer: sweep at a random position p0; time sample i -> column (p0 + i) % W
    p0 = int(rng.integers(0, W)) if p.sweep else 0
    cols = (p0 + np.arange(W)) % W
    disp = img.copy()
    disp[:, x0 + cols] = img[:, x0 + np.arange(W)]

    regions = [
        Region(x0=x0 + 5, y0=0, x1=x0 + W - 5, y1=y0 - 1, dx=0.05, dy=0.05, units_x=3, units_y=3,
               data_type=1, spatial_format=1),
        Region(x0=x0, y0=y0, x1=x0 + W - 1, y1=y0 + H - 1, dx=p.dt, dy=dy, units_x=4, units_y=7,
               data_type=4 if p.modality == "CW" else 3, spatial_format=3, ref_x0=0, ref_y0=zero_row),
    ]
    dimg = DicomImage(path=case_name(p), pixels=disp,
                      photometric="RGB", regions=regions, meta={"synthetic": True})
    truth["t"] = t
    truth["valid"] = time_valid
    truth["ecg"] = ecg
    for key in ("v_max", "v_min", "v_mean", "v_mean_pos", "v_mean_neg"):
        truth[key] = np.where(time_valid, truth[key], np.nan)
    return SyntheticCase(image=dimg, params=p, truth=truth, t_rpeaks=r_times,
                         info={"zero_row": zero_row, "dy": dy, "p0": p0, "t_clicks": t_clicks})


def case_name(p: SyntheticSpec) -> str:
    """Unique name of a synthetic case (also its DicomImage.path)."""
    return f"synthetic:{p.waveform}:{p.modality}{'-alias' if p.alias else ''}:{p.snr_db}:{p.seed}"


def benchmark_suite(snrs=(-3, 0, 3, 6, 10, 15, 20, 30), seeds=(0, 1), artefacts: bool = True) -> list[SyntheticSpec]:
    """The standard synthetic benchmark: modality x waveform x SNR x seed."""
    specs = []
    plans = {
        "PW": [("gaussian", 70), ("systolic", 90), ("two_peaks", 70), ("pos_neg", 70), ("crossing", 60), ("continuous", 60)],
        "CW": [("systolic", 350), ("pos_neg", 250), ("crossing", 200), ("two_peaks", 120)],
        "TDI": [("tdi", 10), ("crossing", 8)],
    }
    ranges = {"PW": (100.0, -100.0), "CW": (150.0, -450.0), "TDI": (15.0, -15.0)}
    for mod, wfs in plans.items():
        for wf, vp in wfs:
            for snr in snrs:
                for seed in seeds:
                    top, bot = ranges[mod]
                    if mod == "CW" and wf in ("crossing", "two_peaks"):
                        top, bot = 250.0, -250.0
                    rng = np.random.default_rng(zlib.crc32(repr((mod, wf, snr, seed)).encode()))
                    s = SyntheticSpec(waveform=wf, modality=mod, snr_db=float(snr), seed=int(seed),
                                      peak_velocity=float(vp), v_top=top, v_bottom=bot,
                                      heart_rate=float(rng.uniform(55, 95)),
                                      dt=0.003521 if seed % 2 == 0 else 0.002641)
                    if artefacts:
                        s.clutter_db = float(rng.choice([0, 0, 10, 20])) if mod != "TDI" else 0.0
                        s.clicks = bool(mod != "TDI" and rng.random() < 0.3)
                        s.mirror_db = float(rng.choice([0, 0, 20])) if mod == "CW" else 0.0
                        s.holes = int(rng.integers(0, 4))
                        s.speckles = int(rng.integers(0, 20))
                        s.broadening = float(rng.uniform(0.02, 0.08))
                        s.reject_db = float(rng.uniform(0, 6))
                    specs.append(s)
    # PW aliasing: flow beyond the Nyquist limit wraps around the display.
    # Positive E/A pattern past the top of a baseline-shifted display (as in
    # 01_10529), and a negative systolic jet past the bottom.
    alias_plans = [("two_peaks", 100.0, 85.0, -34.0), ("systolic", 75.0, 100.0, -40.0)]
    for wf, vp, top, bot in alias_plans:
        for snr in snrs:
            for seed in seeds:
                rng = np.random.default_rng(zlib.crc32(repr(("PW_alias", wf, snr, seed)).encode()))
                s = SyntheticSpec(waveform=wf, modality="PW", snr_db=float(snr), seed=int(seed),
                                  peak_velocity=vp, v_top=top, v_bottom=bot, alias=True,
                                  heart_rate=float(rng.uniform(55, 95)),
                                  dt=0.003521 if seed % 2 == 0 else 0.002641)
                if artefacts:
                    s.holes = int(rng.integers(0, 4))
                    s.speckles = int(rng.integers(0, 20))
                    s.broadening = float(rng.uniform(0.02, 0.08))
                    s.reject_db = float(rng.uniform(0, 6))
                specs.append(s)
    return specs
