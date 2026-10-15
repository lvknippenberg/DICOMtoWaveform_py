"""Figures for docs/explainer/explainer.tex (the tutorial-style explanation of the method).

    python docs/explainer/make_figures.py [--preview DIR]

All figures are synthetic (simulated Doppler signals or synthetic displays from
dicom2waveform.synthetic), so the PDF contains no patient data and can be shared.
The envelope code is the dicom2waveform package (the standalone script is generated from it
and gives identical results).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from dicom2waveform import extract_waveforms as _extract_waveforms  # noqa: E402
from dicom2waveform.config import AliasingConfig, DPConfig, LikelihoodConfig, load_config  # noqa: E402
from dicom2waveform.aliasing import find_cut  # noqa: E402
from dicom2waveform.envelope.base import Context, side_view  # noqa: E402
from dicom2waveform.envelope.dp import DPEstimator, _l1_min_convolution, dp_solve, edge_cost  # noqa: E402
from dicom2waveform.envelope.likelihood import membership, noise_stats  # noqa: E402
from dicom2waveform.render import level_to_power  # noqa: E402


def extract_waveforms(source, cfg=None):
    """extract_waveforms with diagnostics kept (the figures need the spectrogram)."""
    cfg = load_config(cfg)
    cfg.keep_diagnostics = True
    return _extract_waveforms(source, cfg)


def spectrogram(res):
    """The spectrogram the envelope was computed on (unwrapped if aliasing was corrected)."""
    return res.diagnostics["spectrogram"]


def signal_density(spec, ns, lc, dc):
    """Normalised local signal density d, as used by the DP edge cost."""
    return DPEstimator().likelihood(Context(spec, lc, dc, _noise=ns))
from dicom2waveform.synthetic import SyntheticSpec, generate  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
PREVIEW = None

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})
C_MAX, C_MIN, C_MEAN, C_ECG, C_TRUE = "#ff3b30", "#00c8ff", "#ffd60a", "#2e7d32", "white"
C_BLUE = "#1565c0"


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / name.replace(".pdf", ".png"), dpi=110)
    plt.close(fig)


def run_case(cfg=None, **kw):
    case = generate(SyntheticSpec(**kw))
    img = case.image
    res = extract_waveforms(img, cfg)
    return case, img, res


def extent(res, spec=None):
    spec = spec or spectrogram(res)
    v_top = (0 - spec.zero_row) * spec.dy
    v_bot = (spec.level.shape[0] - 1 - spec.zero_row) * spec.dy
    return (res.t[0], res.t[-1] + spec.dt, v_bot, v_top), v_top < v_bot


def show(ax, res, image=None, cmap=None, vmax=None, spec=None):
    spec = spec or spectrogram(res)
    img = spec.rgb if image is None else image
    ext, flip = extent(res, spec)
    ax.imshow(img[:, spec.order], aspect="auto", extent=ext, origin="upper", interpolation="nearest",
              cmap=cmap, vmin=0, vmax=vmax)
    if flip:
        ax.invert_yaxis()


# The main didactic case: PW, flow on both sides, wall clutter, 10 dB
MAIN = dict(waveform="pos_neg", modality="PW", snr_db=10.0, seed=3, clicks=False, mirror_db=0.0, clutter_db=6.0)


# ---------------------------------------------------------------------------------------------
# 1. From echoes to a spectrogram (simulated slow-time signal)
# ---------------------------------------------------------------------------------------------

def fig_spectrogram():
    rng = np.random.default_rng(1)
    f0, c, prf = 2.5e6, 1540.0, 5000.0
    v_nyq = c * prf / (4 * f0) * 100  # cm/s
    t = np.arange(0, 1.75, 1 / prf)
    rr = 0.857
    ph = (t / rr + 0.85) % 1.0
    x = ph - 0.15
    pulse = np.where(x < 0, np.exp(-0.5 * (x / 0.035) ** 2), np.exp(-0.5 * (x / 0.08) ** 2))
    vmax = 15.0 + 52.0 * pulse  # cm/s, carotid-like
    N = 300
    r2 = rng.uniform(0, 1, N)  # (r/R)^2 uniform for scatterers spread uniformly over the lumen
    vfrac = 1 - r2  # parabolic profile -> velocity uniform on [0, vmax]
    amp = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    vcum = np.cumsum(vmax / 100.0) / prf  # integral of vmax (m)
    phase = 2 * np.pi * (2 * f0 / c) * np.outer(vfrac, vcum)
    s = (amp[:, None] * np.exp(1j * phase)).sum(axis=0) / np.sqrt(N)
    noise = (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))) / np.sqrt(2)
    s = s + 10 ** (-24 / 20) * noise * 6
    nper, hop = 128, 18
    f, tt, Z = signal.stft(s, fs=prf, window="hann", nperseg=nper, noverlap=nper - hop, return_onesided=False,
                           boundary=None)
    f = np.fft.fftshift(f)
    P = np.fft.fftshift(np.abs(Z) ** 2, axes=0)
    v = f * c / (2 * f0) * 100
    PdB = 10 * np.log10(P + 1e-12)

    fig = plt.figure(figsize=(6.8, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.55, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    rr_ = np.linspace(-1, 1, 101)
    ax.plot(1 - rr_ ** 2, rr_, color=C_BLUE)
    for yy in np.linspace(-0.9, 0.9, 7):
        ax.annotate("", xy=(1 - yy ** 2, yy), xytext=(0, yy), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
    ax.set_xlabel(r"$v / v_{max}$")
    ax.set_ylabel("position across vessel ($r/R$)")
    ax.set_title("(a) laminar (parabolic) profile:\nevery speed from 0 to $v_{max}$", loc="left")
    ax.set_xlim(0, 1.1)
    ax = fig.add_subplot(gs[0, 1])
    k0 = int(np.argmax(vmax))
    seg = slice(k0 - 100, k0 + 100)
    ax.plot((t[seg] - t[k0]) * 1000, s[seg].real, color="k", lw=0.6, label="I (real)")
    ax.plot((t[seg] - t[k0]) * 1000, s[seg].imag, color="0.6", lw=0.6, label="Q (imag)")
    ax.set_xlabel("slow time (ms)")
    ax.set_yticks([])
    ax.set_title("(b) slow-time signal\n(one sample per pulse)", loc="left")
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    j = int(np.argmin(np.abs(tt - t[k0])))
    pc = PdB[:, j] - np.median(PdB)
    ax.plot(v, pc, color="k", lw=0.7)
    ax.axvline(vmax[k0], color=C_MAX, ls="--", lw=0.9)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.text(vmax[k0] + 3, pc.max() - 2, "true\n$v_{max}$", color=C_MAX, fontsize=7, va="top")
    ax.set_xlabel("velocity (cm/s)")
    ax.set_ylabel("power (dB re noise)")
    ax.set_xlim(-v_nyq, v_nyq)
    ax.set_title("(c) one column = power\nspectrum of one window", loc="left")
    ax = fig.add_subplot(gs[1, :])
    ref = np.median(PdB)
    ax.imshow(PdB - ref, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=40,
              extent=(tt[0], tt[-1], v[0], v[-1]))
    ax.plot(t, vmax, color=C_MAX, ls="--", lw=0.9, label="true $v_{max}(t)$ (fastest blood)")
    ax.axhline(0, color="#d08030", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity (cm/s)")
    ax.set_title(f"(d) spectrogram (STFT, Hann {nper / prf * 1000:.0f} ms, step {hop / prf * 1000:.1f} ms; "
                 f"0-40 dB above noise; Nyquist $\\pm${v_nyq:.0f} cm/s)", loc="left")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlim(tt[0], tt[-1])
    save(fig, "spectrogram.pdf")
    return PdB - ref, v, tt


# ---------------------------------------------------------------------------------------------
# 2. Display rendering: log compression, gain, dynamic range
# ---------------------------------------------------------------------------------------------

def fig_rendering(PdB, v, tt):
    DR, floor = 40.0, 3.0
    gains = (0.0, 8.0)
    fig = plt.figure(figsize=(6.8, 3.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.3, 1.3], hspace=0.6, wspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    p = np.linspace(-5, 55, 300)
    for g, col in zip(gains, ("k", C_BLUE)):
        ax.plot(p, np.clip(255 * (p + g - floor) / DR, 0, 255), color=col, label=f"gain {g:+.0f} dB")
    ax.set_xlabel("Doppler power (dB re noise)")
    ax.set_ylabel("grey level $L$")
    ax.set_title("(a) grey map: log, clip\n(reject + dynamic range)", loc="left")
    ax.legend(loc="lower right")
    Ls = []
    for i, (g, lab) in enumerate(zip(gains, ("(b)", "(c)"))):
        L = np.clip(255 * (PdB + g - floor) / DR, 0, 255)
        Ls.append(L)
        a = fig.add_subplot(gs[i, 1])
        a.imshow(L, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=255, extent=(tt[0], tt[-1], v[0], v[-1]))
        a.set_title(f"{lab} rendered at gain {g:+.0f} dB", loc="left")
        a.set_ylabel("cm/s")
        if i == 1:
            a.set_xlabel("time (s)")
    a = fig.add_subplot(gs[:, 2])
    for L, g, col in zip(Ls, gains, ("k", C_BLUE)):
        a.hist(L.ravel(), bins=64, range=(0, 255), histtype="step", color=col, density=True, label=f"gain {g:+.0f} dB")
        a.axvline(np.median(L), color=col, ls=":", lw=0.9)
    a.set_yscale("log")
    a.set_xlabel("grey level $L$")
    a.set_ylabel("fraction of pixels (log)")
    a.set_title("(d) histograms: gain shifts\nnoise and flow alike", loc="left")
    a.legend(loc="upper right")
    save(fig, "rendering.pdf")


# ---------------------------------------------------------------------------------------------
# 3. Why a per-column threshold is not enough
# ---------------------------------------------------------------------------------------------

def outermost_above(spec, side, T):
    sv = side_view(spec, side)
    above = sv.level >= T
    k = np.where(above.any(axis=0), sv.depth - 1 - np.argmax(above[::-1], axis=0), 0)
    k = k.astype(float)
    k[spec.sweep] = np.nan
    return sv.sign * k * sv.speed_per_row


def fig_threshold(case, res):
    spec = spectrogram(res)
    o = spec.order
    ns = noise_stats(spec, LikelihoodConfig())
    t = res.t
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    show(ax[0], res)
    for T, col, lab in [(20.0, "#ff9100", "$L \\geq$ 20 (Otsu)"), (60.0, "#00e676", "$L \\geq$ 60")]:
        ax[0].plot(t, outermost_above(spec, "pos", T)[o], color=col, lw=0.7, label=lab)
        ax[0].plot(t, outermost_above(spec, "neg", T)[o], color=col, lw=0.7)
    ax[0].set_title("(a) outermost pixel above a threshold", loc="left")
    show(ax[1], res)
    ax[1].plot(t, res.v_max, color=C_MAX, lw=1.0, label="DP $v_{max}$")
    ax[1].plot(t, res.v_min, color=C_MIN, lw=1.0, label="DP $v_{min}$")
    ax[1].set_title("(b) this method, same image", loc="left")
    for a in ax:
        a.plot(t, case.truth["v_max"], color=C_TRUE, ls="--", lw=0.7)
        a.plot(t, case.truth["v_min"], color=C_TRUE, ls="--", lw=0.7, label="truth")
        a.set_xlim(0.2, 1.6)
        a.set_xlabel("time (s)")
        a.legend(loc="upper right", fontsize=6, framealpha=0.85)
    ax[0].set_ylabel("velocity (cm/s)")
    save(fig, "threshold.pdf")


# ---------------------------------------------------------------------------------------------
# 4. Soft membership
# ---------------------------------------------------------------------------------------------

def fig_membership(res):
    spec = spectrogram(res)
    ns = noise_stats(spec, LikelihoodConfig())
    vals = spec.level[spec.valid]
    m = membership(spec.level, ns)
    m[:, spec.sweep] = 0
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.55))
    a = ax[0]
    a.hist(vals, bins=128, range=(0, 255), color="0.6", density=True)
    a.set_yscale("log")
    a.set_xlabel("display level $L$")
    a.set_ylabel("fraction of pixels (log)")
    a.set_xlim(-3, 120)
    labels = [(ns.level, r"$\mu$ = %.0f" % ns.level, "k", 0.2), (ns.noise_top, r"$T_0$ = %.1f" % ns.noise_top, C_BLUE, 0.07),
              (ns.threshold, r"$T$ = min($\mu$+$M$, $T_{Otsu}$) = %.0f" % ns.threshold, C_MAX, 0.024)]
    for x, lab, col, yf in labels:
        a.axvline(x, color=col, lw=0.9)
        # labels right of the ramp, so they do not touch the lines
        a.text(ns.threshold + 4, yf, lab, color=col, fontsize=6.5)
    b = a.twinx()
    L = np.linspace(0, 255, 256)
    b.plot(L, membership(L, ns), color=C_MAX, lw=1.4)
    b.set_ylabel("membership $m(L)$", color=C_MAX)
    b.set_ylim(-0.02, 1.05)
    b.spines["right"].set_visible(True)
    a.set_title("(a) level histogram and the ramp", loc="left")
    show(ax[1], res, image=m, cmap="magma", vmax=1)
    ax[1].set_xlim(0.2, 1.6)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("velocity (cm/s)")
    ax[1].set_title("(b) membership $m$ of every pixel (0 black, 1 bright)", loc="left")
    save(fig, "membership.pdf")
    return ns


# ---------------------------------------------------------------------------------------------
# 5. Local density, background subtraction, normalisation
# ---------------------------------------------------------------------------------------------

def fig_density(res):
    spec = spectrogram(res)
    lc, dc = LikelihoodConfig(), DPConfig()
    ns = noise_stats(spec, lc)
    m = membership(spec.level, ns)
    m[:, spec.sweep] = 0
    mt = ndimage.uniform_filter(m.astype(float), size=(lc.density_window_px_rows, lc.density_window_px_cols),
                                mode="nearest")
    d = signal_density(spec, ns, lc, dc)
    valid = spec.valid & ~spec.sweep[None, :]
    bg = float(np.percentile(mt[valid], dc.background_percentile))
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=True)
    for a, img, tt in zip(ax, (m, mt, d), ("(a) membership $m$", r"(b) local mean $\tilde d$ (9$\times$5 px)",
                                           r"(c) $d$: minus background, $\div$ P99")):
        show(a, res, image=img, cmap="magma", vmax=1)
        a.set_title(tt, loc="left")
        a.set_xlim(0.45, 1.15)
        a.set_xlabel("time (s)")
    ext, flip = extent(res)
    ax[2].contour(d[:, spec.order], levels=[0.5], colors=["#00e676"], linewidths=0.7, origin="upper",
                  extent=ext)
    ax[0].set_ylabel("velocity (cm/s)")
    save(fig, "density.pdf")
    return bg


# ---------------------------------------------------------------------------------------------
# 6. Cost of an edge in one column
# ---------------------------------------------------------------------------------------------

def column_profiles(D=200, dv=0.5):
    rng = np.random.default_rng(5)
    k = np.arange(D)

    def roll(edge, w=4.0):
        return 0.5 * (1 - np.tanh((k - edge) / w))

    blob = 0.65 * np.exp(-0.5 * ((k - 168) / 2.5) ** 2)
    base = 0.04 * rng.random(D)
    filled = np.clip(0.95 * roll(120) + blob + base, 0, 1)
    hollow = np.clip(0.12 + 0.83 * roll(120) * (1 - roll(70, 3.0)) + blob + base, 0, 1)
    return k, k * dv, filled, hollow


def fig_column_cost():
    theta, h, rho, wall = 0.5, 10, 0.01, 6
    k, v, filled, hollow = column_profiles()
    D = len(k)
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.6), gridspec_kw=dict(width_ratios=[1, 1.6], hspace=0.4), sharey=True)
    for row, (d, name) in enumerate(((filled, "filled spectrum (CW-like)"), (hollow, "hollow spectrum (PW window)"))):
        a = ax[row, 0]
        a.fill_betweenx(v, 0, d, color="0.85")
        a.plot(d, v, color="k", lw=0.9)
        a.axvline(theta, color="0.4", ls=":", lw=0.9)
        a.text(theta + 0.02, v[-1] * 0.97, r"$\theta$", fontsize=8, va="top")
        a.set_xlim(0, 1.05)
        a.set_ylabel("velocity (cm/s)")
        a.set_title(f"({'ac'[row]}) {name}", loc="left")
        dd = d[:, None]
        w = np.ones((D, 1))
        w[:wall] = 0.01
        excess = np.clip(dd - theta, 0, None)
        beyond = (excess.sum(0, keepdims=True) - np.cumsum(excess, 0))[:, 0]
        sc = np.cumsum(w * np.clip(theta - dd, 0, None), 0)
        prev = np.zeros_like(sc)
        prev[h:] = sc[:-h]
        inner = (sc - prev)[:, 0]
        prior = rho * k
        total = edge_cost(dd, theta, h, rho, wall)[:, 0]
        split = (dd.sum(0, keepdims=True) - np.cumsum(dd, 0) + np.cumsum(w * (1 - dd), 0))[:, 0]
        b = ax[row, 1]
        b.plot(beyond, v, color="#ef6c00", lw=0.9, label="signal left outside")
        b.plot(inner, v, color="#6a1b9a", lw=0.9, label="no signal just inside")
        b.plot(prior, v, color="0.5", lw=0.9, label=r"prior $\rho k$")
        b.plot(total, v, color=C_BLUE, lw=1.8, label="edge cost $C(k)$")
        b.plot(split, v, color="k", ls="--", lw=0.9, label="two-region split (old)")
        ke, ks = int(np.argmin(total)), int(np.argmin(split))
        b.plot(total[ke], v[ke], "o", color=C_BLUE, ms=5)
        b.plot(split[ks], v[ks], "s", color="k", ms=4, mfc="none")
        a.axhline(v[ke], color=C_BLUE, lw=0.9)
        a.axhline(v[ks], color="k", ls="--", lw=0.8)
        b.set_xlim(0, 80)
        b.set_title(f"({'bd'[row]}) cost per candidate edge: minimum at {v[ke]:.0f} cm/s "
                    f"(old split: {v[ks]:.0f} cm/s)", loc="left")
        if row == 0:
            b.legend(loc="upper right", fontsize=6.5)
    ax[1, 1].set_xlabel("cost")
    ax[1, 0].set_xlabel("density $d$")
    save(fig, "column_cost.pdf")


# ---------------------------------------------------------------------------------------------
# 7. Dynamic programming: toy example and the L1 distance transform
# ---------------------------------------------------------------------------------------------

def fig_dp_toy():
    rng = np.random.default_rng(2)
    D, W = 8, 12
    true = np.array([1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 2, 1])
    k = np.arange(D)[:, None]
    C = 0.9 * np.abs(k - true[None, :]) ** 0.8 + 0.6 * rng.random((D, W))
    C[7, 6] = 0.0  # a bright noise speck far out
    C[5, 6] = 0.8
    C[0, 9] = 0.0  # and a dropout to "no flow"
    C = np.round(C, 1)
    greedy = C.argmin(axis=0)
    lam = 1.0
    path = dp_solve(C, lam)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw=dict(width_ratios=[1.55, 1]))
    a = ax[0]
    a.imshow(C, origin="lower", cmap="Blues_r", aspect="auto", vmin=0, vmax=C.max())
    for x in range(W):
        for y in range(D):
            a.text(x, y, f"{C[y, x]:.1f}", ha="center", va="center", fontsize=5.5,
                   color="k" if C[y, x] > 1.5 else "w")
    a.plot(np.arange(W), greedy, "x", color="#ff9100", ms=7, mew=1.6, label="per-column minimum")
    a.plot(np.arange(W), path, "-o", color=C_MAX, lw=1.4, ms=3, label=rf"DP path ($\lambda$ = {lam:g})")
    a.set_xlabel("column $x$ (time)")
    a.set_ylabel("row $k$ (speed)")
    E0 = lambda p: C[p, np.arange(W)].sum() + lam * np.abs(np.diff(p)).sum()  # noqa: E731
    a.set_title(f"(a) cost table $C_x(k)$; total $E$: per-column {E0(greedy):.1f}, DP {E0(path):.1f}", loc="left")
    a.legend(loc="upper left", fontsize=6)
    # (b) L1 distance transform
    b = ax[1]
    f = np.array([3.0, 2.2, 3.4, 1.2, 2.8, 3.6, 2.0, 2.6, 3.8, 3.2])
    lam2 = 0.6
    i = np.linspace(-0.5, len(f) - 0.5, 400)
    for kk, fk in enumerate(f):
        b.plot(i, fk + lam2 * np.abs(i - kk), color="0.8", lw=0.6)
    hmin, arg = _l1_min_convolution(f, lam2)
    b.plot(np.arange(len(f)), f, "o", color="k", ms=3.5, label=r"$g(k') = F_{x-1}(k')$")
    b.plot(np.arange(len(f)), hmin, "-s", color=C_MAX, ms=3, lw=1.2, label=r"$\psi(k) = \min_{k'} g(k')+\lambda|k-k'|$")
    b.set_ylim(0.8, 5)
    b.set_xlabel("row $k$")
    b.set_title("(b) inner minimum = lower envelope\nof cones (L1 distance transform)", loc="left")
    b.legend(loc="upper left", fontsize=6)
    save(fig, "dp_toy.pdf")


def fig_lambda(case, img):
    t0, t1 = 0.45, 1.25
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 2.6))
    base = extract_waveforms(img)
    show(ax, base)
    for a_ref, col, lab in [(1e7, "#00e676", r"$a_{ref}\to\infty$ ($\lambda\approx 0$): column by column"),
                            (640.0, C_MAX, r"$a_{ref}$ = 640 cm/s$^2$ (default)"),
                            (40.0, "#ff9100", r"$a_{ref}$ = 40 cm/s$^2$: too stiff")]:
        r = extract_waveforms(img, {"dp": {"reference_accel_cm_s2": a_ref}})
        ax.plot(r.t, r.v_max, color=col, lw=0.9, label=lab)
        ax.plot(r.t, r.v_min, color=col, lw=0.9)
    ax.plot(base.t, case.truth["v_max"], color=C_TRUE, ls="--", lw=0.7, label="truth")
    ax.plot(base.t, case.truth["v_min"], color=C_TRUE, ls="--", lw=0.7)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity (cm/s)")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)
    ax.set_title("Effect of the smoothness weight (expressed as a reference acceleration)", loc="left")
    save(fig, "lambda.pdf")


# ---------------------------------------------------------------------------------------------
# 8. Aliasing
# ---------------------------------------------------------------------------------------------

def fig_aliasing():
    case, img, res = run_case(waveform="systolic", modality="PW", snr_db=20, seed=5, peak_velocity=75.0,
                              v_top=100.0, v_bottom=-40.0, alias=True)
    raw = extract_waveforms(img, {"aliasing": {"correct": False}})
    # the cut path that the unwrap step uses
    spec = spectrogram(raw)
    ns = noise_stats(spec, LikelihoodConfig())
    dens = signal_density(spec, ns, LikelihoodConfig(), DPConfig())
    cut = find_cut(dens, spec.zero_row, spec.sweep, AliasingConfig())[0].astype(float)
    cut[spec.sweep] = np.nan
    v_cut = ((cut - spec.zero_row) * spec.dy)[spec.order]

    fig = plt.figure(figsize=(6.8, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.55, wspace=0.25)
    a = fig.add_subplot(gs[0, :])
    vn = 50.0
    v = np.linspace(-3.2 * vn, 3.2 * vn, 2000)
    true = np.exp(-0.5 * ((v - 70) / 14) ** 2)
    rep = sum(np.exp(-0.5 * ((v - 70 - n * 2 * vn) / 14) ** 2) for n in (-2, -1, 1, 2))
    a.fill_between(v, 0, rep, color="0.85", label="replicas every $2v_{Nyq}$ (sampling at the PRF)")
    a.plot(v, true, color="k", ls="--", lw=0.9, label="true velocity spectrum")
    inwin = np.abs(v) <= vn
    a.plot(v[inwin], (true + rep)[inwin], color=C_MAX, lw=1.6, label="what the display shows")
    for x in (-vn, vn):
        a.axvline(x, color=C_BLUE, lw=0.9)
    a.text(vn + 2, 1.05, "$+v_{Nyq}$", color=C_BLUE, fontsize=7)
    a.text(-vn - 2, 1.05, "$-v_{Nyq}$", color=C_BLUE, fontsize=7, ha="right")
    a.set_xlabel("velocity")
    a.set_yticks([])
    a.set_xticks([])
    a.set_ylim(0, 1.15)
    a.legend(loc="upper center", fontsize=6.5, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.18))
    a.set_title("(a) a PW spectrum is periodic in velocity", loc="left")
    b = fig.add_subplot(gs[1, 0])
    show(b, raw)
    b.plot(raw.t, raw.v_max, color=C_MAX, lw=0.9)
    b.plot(raw.t, raw.v_min, color=C_MIN, lw=0.9)
    b.plot(raw.t, v_cut, ".", color="#00e676", ms=1.5, label="least-signal cut")
    b.set_title("(b) as displayed; both edges hit the limits", loc="left")
    b.set_xlabel("time (s)")
    b.set_ylabel("velocity (cm/s)")
    b.legend(loc="lower right", fontsize=6.5, framealpha=0.85)
    c = fig.add_subplot(gs[1, 1])
    show(c, res)
    c.plot(res.t, res.v_max, color=C_MAX, lw=0.9)
    c.plot(res.t, res.v_min, color=C_MIN, lw=0.9)
    c.plot(res.t, case.truth["v_min"], color=C_TRUE, ls="--", lw=0.7, label="truth")
    for vv in spectrogram(res).display_v_limits:
        c.axhline(vv, color="#00e676", ls=":", lw=0.8)
    c.set_title("(c) rows beyond the cut moved by one span", loc="left")
    c.set_xlabel("time (s)")
    c.legend(loc="lower right", fontsize=6.5, framealpha=0.85)
    save(fig, "aliasing.pdf")
    tr, est = np.nanmin(case.truth["v_min"]), np.nanmin(res.v_min)
    return tr, est, res.flags


# ---------------------------------------------------------------------------------------------
# 9. Mean velocity
# ---------------------------------------------------------------------------------------------

def fig_mean(case, res):
    spec = spectrogram(res)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1, 2]))
    tr = case.truth
    c = int(np.nanargmax(tr["v_max"]))
    craw = spec.order[c]
    ns = noise_stats(spec, LikelihoodConfig())
    sv = side_view(spec, "pos")
    vv = np.arange(sv.depth) * sv.speed_per_row
    w = np.clip(level_to_power(sv.level[:, craw], 40) - level_to_power(ns.level, 40), 0, None)
    inside = (vv <= res.v_max[c]) & (np.arange(sv.depth) >= 1)
    a = ax[0]
    a.plot(w / w.max(), vv, color="0.6", lw=0.7, label="weight (all rows)")
    a.fill_betweenx(vv, 0, np.where(inside, w / w.max(), 0), color=C_BLUE, alpha=0.5, label="used (inside edge)")
    a.axhline(res.v_max[c], color=C_MAX, lw=1, label="$v_{max}$")
    a.axhline(res.v_mean_pos[c], color="#b8860b", lw=1.2, ls="--", label="mean")
    a.set_ylim(0, vv[-1])
    a.set_xlabel("de-compressed power (norm.)")
    a.set_ylabel("velocity (cm/s)")
    a.set_title("(a) one column", loc="left")
    a.legend(loc="upper right", fontsize=6)
    b = ax[1]
    b.plot(res.t, tr["v_max"], color="0.5", lw=0.8, ls="--", label="true $v_{max}$")
    b.plot(res.t, res.v_max, color=C_MAX, lw=0.9, label="$v_{max}$")
    b.plot(res.t, tr["v_mean_pos"], color="0.2", lw=0.8, ls="--", label="true mean")
    b.plot(res.t, res.v_mean_pos, color="#b8860b", lw=1.0, label="mean (estimate)")
    b.set_xlim(0.2, 1.6)
    b.set_ylim(0, None)
    b.set_xlabel("time (s)")
    b.set_title("(b) positive side over time", loc="left")
    b.legend(loc="upper right", fontsize=6, ncol=2)
    save(fig, "mean.pdf")


# ---------------------------------------------------------------------------------------------
# 10. The complete output, and the clinician summary figure
# ---------------------------------------------------------------------------------------------

def fig_overview():
    case, img, res = run_case(waveform="systolic", modality="CW", clicks=True, snr_db=15, seed=2, v_top=50.0,
                              v_bottom=-250.0, peak_velocity=180.0)
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    show(ax[0], res)
    ax[0].plot(res.t, res.v_max, color=C_MAX, lw=0.9, label="$v_{max}$")
    ax[0].plot(res.t, res.v_min, color=C_MIN, lw=0.9, label="$v_{min}$")
    ax[0].plot(res.t, res.v_mean_neg, color=C_MEAN, lw=0.7, ls="--", label="mean (neg. side)")
    ax[0].set_ylabel("velocity (cm/s)")
    ax[0].legend(loc="lower right", ncol=3, framealpha=0.85)
    ax[0].set_title("Synthetic CW aortic outflow with valve clicks: all outputs on one time axis", loc="left")
    ax[1].plot(res.t, res.ecg, color=C_ECG, lw=0.8)
    for x in res.t_rpeaks:
        for a in ax:
            a.axvline(x, color="k", ls=":", lw=0.7)
    for x in res.t_avc:
        ax[1].axvline(x, color="#c2185b", ls="--", lw=1.0)
    ax[1].set_yticks([])
    ax[1].set_ylabel("ECG")
    ax[1].set_xlabel("time (s)   (dotted: R-peaks, magenta: aortic valve closure)")
    ax[1].set_xlim(res.t[0], res.t[-1])
    save(fig, "overview.pdf")
    return res


def fig_clinician(case, res):
    spec = spectrogram(res)
    dc = DPConfig()
    ns = noise_stats(spec, LikelihoodConfig())
    d = signal_density(spec, ns, LikelihoodConfig(), dc)
    H = spec.display_height
    greedy = {}
    for side in ("pos", "neg"):
        sv = side_view(spec, side, image=d)
        cost = edge_cost(sv.level, dc.edge_density, max(1, int(round(dc.inner_band_rel * H))), dc.height_prior,
                           int(round(dc.wall_band_rel * H)))
        g = cost.argmin(axis=0).astype(float) * sv.speed_per_row * sv.sign
        g[spec.sweep] = np.nan
        greedy[side] = g[spec.order]
    t0, t1 = 0.45, 1.25
    fig, ax = plt.subplots(1, 4, figsize=(6.8, 2.0), sharey=True)
    show(ax[0], res)
    show(ax[1], res, image=d, cmap="magma", vmax=1)
    show(ax[2], res)
    ax[2].plot(res.t, greedy["pos"], ".", color="#ff9100", ms=1.6)
    ax[2].plot(res.t, greedy["neg"], ".", color="#ff9100", ms=1.6)
    show(ax[3], res)
    ax[3].plot(res.t, res.v_max, color=C_MAX, lw=1.0)
    ax[3].plot(res.t, res.v_min, color=C_MIN, lw=1.0)
    for a, tt in zip(ax, ("1. the picture", "2. how flow-like\nis each spot?", "3. where does flow\nend, column by column?",
                          "4. best smooth line\nthrough all columns")):
        a.set_xlim(t0, t1)
        a.set_title(tt, loc="left", fontsize=7.5)
        a.set_xticks([])
    ax[0].set_ylabel("velocity")
    ax[0].set_yticks([])
    save(fig, "clinician_steps.pdf")


PARAM_PANELS = [  # (title, section, key, low, high)
    (r"edge level $\theta$", "dp", "edge_density", 0.35, 0.65),
    (r"density window", "likelihood", ("density_window_px_rows", "density_window_px_cols"),
     (5, 3), (15, 9)),
    (r"inner band $h$ (of $H$)", "dp", "inner_band_rel", 0.025, 0.1),
    (r"height prior $\rho$", "dp", "height_prior", 0.002, 0.05),
    (r"background pct.", "dp", "background_percentile", 0.0, 75.0),
    (r"normalisation pct.", "dp", "normalise_percentile", 0.0, 99.9),
    (r"wall band (of $H$)", "dp", "wall_band_rel", 0.0, 0.06),
    (r"noise margin $M$", "likelihood", "noise_margin", 16.0, 64.0),
]


def fig_param_effects(case, img):
    """Lower / default / higher value of the main envelope parameters on one synthetic beat."""
    t0, t1 = 0.45, 1.25
    base = extract_waveforms(img)
    tr = case.truth

    def err(r):
        m = (base.t >= t0) & (base.t <= t1) & ~np.isnan(r.v_max) & ~np.isnan(tr["v_max"])
        return np.mean(np.abs(np.r_[r.v_max[m] - tr["v_max"][m], r.v_min[m] - tr["v_min"][m]]))

    def fmt(v):
        return "x".join(str(x) for x in v) if isinstance(v, tuple) else f"{v:g}"

    fig, axs = plt.subplots(2, 4, figsize=(6.8, 3.6), sharex=True, sharey=True)
    for ax, (title, sec, key, lo, hi) in zip(axs.ravel(), PARAM_PANELS):
        show(ax, base)
        ax.plot(base.t, tr["v_max"], color=C_TRUE, ls="--", lw=0.6)
        ax.plot(base.t, tr["v_min"], color=C_TRUE, ls="--", lw=0.6)
        for val, col in ((lo, "#00e676"), (None, C_MAX), (hi, "#ff9100")):
            if val is None:
                r, lab = base, "default"
            else:
                ov = {sec: dict(zip(key, val))} if isinstance(key, tuple) else {sec: {key: val}}
                r, lab = extract_waveforms(img, ov), fmt(val)
            ax.plot(r.t, r.v_max, color=col, lw=0.8, label=f"{lab}: {err(r):.1f}")
            ax.plot(r.t, r.v_min, color=col, lw=0.8)
        ax.set_xlim(t0, t1)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.legend(loc="lower right", fontsize=5.5, framealpha=0.8, handlelength=1.0, borderpad=0.3)
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("cm/s")
    save(fig, "param_effects.pdf")


if __name__ == "__main__":
    if "--preview" in sys.argv:
        PREVIEW = sys.argv[sys.argv.index("--preview") + 1]
    PdB, v, tt = fig_spectrogram()
    fig_rendering(PdB, v, tt)
    case, img, res = run_case(**MAIN)
    ok = ~np.isnan(res.v_max)
    mae = np.mean(np.abs(np.r_[res.v_max[ok] - case.truth["v_max"][ok], res.v_min[ok] - case.truth["v_min"][ok]]))
    print("main case: MAE %.2f cm/s, dy %.3f cm/s, dt %.4f s" % (mae, spectrogram(res).dy, spectrogram(res).dt))
    print("display limits", spectrogram(res).display_v_limits, "H", spectrogram(res).display_height)
    fig_threshold(case, res)
    ns = fig_membership(res)
    print("noise stats", vars(ns))
    print("density background", fig_density(res))
    fig_column_cost()
    fig_dp_toy()
    fig_lambda(case, img)
    case3, img3, _ = run_case(**{**MAIN, "snr_db": 3.0})
    fig_param_effects(case3, img3)
    print("aliasing (true min, est min, flags):", fig_aliasing())
    fig_mean(case, res)
    r = fig_overview()
    print("overview: R", r.t_rpeaks, "AVC", r.t_avc)
    fig_clinician(case, res)
    print("written to", OUT)
