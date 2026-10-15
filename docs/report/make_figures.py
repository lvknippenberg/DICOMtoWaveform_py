"""Figures for docs/report/report.tex, computed with the dicom2waveform package (the standalone
script, standalone/dicom2waveform_standalone.py, is generated from it and gives identical results).

    python docs/report/make_figures.py

Real-image figures show only the spectral region and the ECG, but they come from patient
images: they are written to docs/report/figures/ (*.pdf is gitignored) and must not be committed.
Example DICOM locations are those of the development machine (see HANDOFF.md).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from dicom2waveform import extract_waveforms as _extract_waveforms  # noqa: E402
from dicom2waveform.config import load_config  # noqa: E402
from dicom2waveform.envelope.base import Context, side_view  # noqa: E402
from dicom2waveform.envelope.dp import DPEstimator, edge_cost  # noqa: E402
from dicom2waveform.envelope.likelihood import membership, noise_stats  # noqa: E402
from dicom2waveform.io import load_dicom  # noqa: E402


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

warnings.filterwarnings("ignore")
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
PREVIEW = None  # set to a folder (e.g. via --preview DIR) to also write PNG previews


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / name.replace(".pdf", ".png"), dpi=110)

MATLAB = Path(r"C:\Github\DICOMtoWaveform")
NEW = Path(r"C:\DICOM waveforms\DICOM doppler examples\DICOM doppler examples")
EXAMPLES = {
    "pw": NEW / "*L3VHDBZ5",
    "cw": NEW / "01_232_*",
    "tdi": NEW / "01_10638_*",
    "vascular": MATLAB / "Annotated examples" / "4264S7VX",
    "alias": NEW / "01_10529_*",
}
EXAMPLES = {k: (next(v.parent.glob(v.name)) if "*" in v.name else v) for k, v in EXAMPLES.items()}
VENDOR_RGB = (98, 164, 255)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})
C_MAX, C_MIN, C_MEAN, C_ECG = "#ff3b30", "#00c8ff", "#ffd60a", "#2e7d32"


def show_display(ax, res, spec=None, t_offset=0.0):
    """Spectral display (time order) with velocity axis, as plot_result does."""
    spec = spec or spectrogram(res)
    rgb = spec.rgb[:, spec.order]
    t = res.t
    v_top = (0 - spec.zero_row) * spec.dy
    v_bot = (rgb.shape[0] - 1 - spec.zero_row) * spec.dy
    ax.imshow(rgb, aspect="auto", extent=(t[0] - t_offset, t[-1] + spec.dt - t_offset, v_bot, v_top),
              origin="upper", interpolation="nearest")
    if v_top < v_bot:
        ax.invert_yaxis()
    ax.set_ylabel("velocity (cm/s)")


def envelopes(ax, res, means=True, lw=1.0):
    ax.plot(res.t, res.v_max, color=C_MAX, lw=lw, label=r"$v_{max}$")
    ax.plot(res.t, res.v_min, color=C_MIN, lw=lw, label=r"$v_{min}$")
    if means:
        ax.plot(res.t, res.v_mean_pos, color=C_MEAN, lw=0.7, ls="--", label=r"$v_{mean,pos/neg}$")
        ax.plot(res.t, res.v_mean_neg, color=C_MEAN, lw=0.7, ls="--")


def ecg_panel(ax, res, avc=True, clicks=False):
    ax.plot(res.t, res.ecg, color=C_ECG, lw=0.8)
    for tr in res.t_rpeaks:
        ax.axvline(tr, color="k", ls=":", lw=0.8)
    if clicks:
        for tc in res.t_clicks:
            ax.axvline(tc, color="0.6", ls="--", lw=0.6)
    if avc:
        for ta in res.t_avc:
            ax.axvline(ta, color="#c2185b", ls="--", lw=1.0)
    ax.set_ylabel("ECG (a.u.)")
    ax.set_yticks([])
    ax.set_xlabel("time (s)")


def example_figure(key, title, avc=True, clicks=False, ylim=None, cfg=None):
    res = extract_waveforms(EXAMPLES[key], cfg)
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1]))
    show_display(ax[0], res)
    envelopes(ax[0], res)
    if ylim:
        ax[0].set_ylim(*ylim)
    ax[0].legend(loc="upper right", ncol=3, framealpha=0.85)
    f = res.flags
    ax[0].set_title(f"{title}  [{f['modality']}, SNR proxy {f['snr_proxy_db']:.0f} dB, "
                    f"{len(res.t_rpeaks)} R-peaks" + (f", {len(res.t_avc)} AVC" if avc else "") + "]")
    ecg_panel(ax[1], res, avc=avc, clicks=clicks)
    ax[1].set_xlim(res.t[0], res.t[-1])
    save(fig, f"ex_{key}.pdf")
    plt.close(fig)
    return res


def vendor_figure():
    img = load_dicom(EXAMPLES["vascular"])
    res = extract_waveforms(img)
    spec = spectrogram(res)
    region = spec.region
    sub = img.pixels[region.y0:region.y1 + 1, region.x0:region.x1 + 1]
    m = np.all(np.abs(sub.astype(int) - np.asarray(VENDOR_RGB)) <= 12, axis=2)
    rows = np.arange(m.shape[0])[:, None]
    n = m.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (m * rows).sum(axis=0) / n
    r[n == 0] = np.nan
    v_gt = ((r - region.ref_y0) * region.dy)[spec.order]
    est = np.where(v_gt >= 0, res.v_max, res.v_min)
    err = est - v_gt
    ok = ~np.isnan(err)
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1]))
    show_display(ax[0], res)
    ax[0].plot(res.t, v_gt, color="#2f6bff", lw=2.2, alpha=0.9, label="scanner trace")
    ax[0].plot(res.t, res.v_max, color=C_MAX, lw=0.9, label=r"$v_{max}$ (DP)")
    ax[0].plot(res.t, res.v_min, color=C_MIN, lw=0.9, label=r"$v_{min}$ (DP)")
    ax[0].legend(loc="upper right", ncol=3, framealpha=0.85)
    ax[0].set_title(f"Example 4: vascular PW with the scanner's own envelope trace "
                    f"(trace masked before estimation); MAE {np.nanmean(np.abs(err)):.1f} cm/s")
    ax[1].plot(res.t, err, color="k", lw=0.7)
    ax[1].axhline(0, color="0.6", lw=0.5)
    ax[1].set_ylabel("DP - trace\n(cm/s)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_xlim(res.t[0], res.t[-1])
    save(fig, "ex_vascular.pdf")
    plt.close(fig)
    return float(np.nanmean(np.abs(err))), float(np.nanmean(err)), int(ok.sum())


def alias_figure():
    raw = extract_waveforms(EXAMPLES["alias"], {"aliasing": {"correct": False}})
    res = extract_waveforms(EXAMPLES["alias"])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=False)
    show_display(ax[0], raw)
    envelopes(ax[0], raw, means=False)
    ax[0].set_title("(a) as displayed: flow wraps around\n(first DP pass; aliasing suspected)")
    show_display(ax[1], res)
    envelopes(ax[1], res, means=False)
    for v in spectrogram(res).display_v_limits:
        ax[1].axhline(v, color="w", ls="--", lw=0.7)
    ax[1].set_title("(b) after automatic per-column baseline shift\n(dashed: display limits)")
    ax[1].set_ylabel("")
    for a in ax:
        a.set_xlabel("time (s)")
    ax[0].legend(loc="lower right", ncol=2, framealpha=0.85)
    fig.tight_layout()
    save(fig, "ex_alias.pdf")
    plt.close(fig)
    return res.meta["aliasing"], raw.flags


def dp_steps_figure():
    """DP on a synthetic PW display (no patient data): display, membership, density + path, one column."""
    from dicom2waveform.synthetic import SyntheticSpec, generate

    case = generate(SyntheticSpec(waveform="pos_neg", modality="PW", snr_db=10.0, seed=3, clicks=False,
                                  mirror_db=0.0, clutter_db=6.0))
    img = case.image
    cfg = load_config()
    res = extract_waveforms(img, cfg)
    spec = spectrogram(res)
    ns = noise_stats(spec, cfg.likelihood)
    memb = membership(spec.level, ns)
    memb[:, spec.sweep] = 0
    dens = signal_density(spec, ns, cfg.likelihood, cfg.dp)
    o = spec.order
    t = res.t
    v_top = (0 - spec.zero_row) * spec.dy
    v_bot = (spec.level.shape[0] - 1 - spec.zero_row) * spec.dy
    ext = (t[0], t[-1] + spec.dt, v_bot, v_top)
    truth = case.truth
    fig = plt.figure(figsize=(6.6, 5.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.2, 1], hspace=0.45, wspace=0.08)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    titles = ["(a) rendered display (synthetic PW, SNR 10 dB, wall clutter, speckle)",
              r"(b) soft membership $m$ (0 = noise, 1 = signal)",
              r"(c) normalised local density $d$ with DP edges; dashed = true edges"]
    for a, img_, tt, cmap in zip(axes, [spec.rgb[:, o], memb[:, o], dens[:, o]], titles, [None, "magma", "magma"]):
        a.imshow(img_, aspect="auto", extent=ext, origin="upper", interpolation="nearest", cmap=cmap,
                 vmin=0, vmax=None if cmap is None else 1)
        if v_top < v_bot:
            a.invert_yaxis()
        a.set_title(tt, loc="left")
        a.set_ylabel("cm/s")
    axes[2].plot(t, res.v_max, color=C_MAX, lw=1.0)
    axes[2].plot(t, res.v_min, color=C_MIN, lw=1.0)
    axes[2].plot(t, truth["v_max"], color="w", lw=0.8, ls="--")
    axes[2].plot(t, truth["v_min"], color="w", lw=0.8, ls="--")
    axes[2].set_xlabel("time (s)")
    for a in axes[:2]:
        a.set_xticklabels([])
    # one column: density profile and edge cost on the positive side
    c = int(np.nanargmax(truth["v_max"]))
    craw = o[c]
    sv = side_view(spec, "pos", image=dens)
    H = spec.display_height
    wall = int(round(cfg.dp.wall_band_rel * H))
    cost = edge_cost(sv.level, cfg.dp.edge_density, max(1, int(round(cfg.dp.inner_band_rel * H))),
                       cfg.dp.height_prior, wall)
    v = np.arange(sv.depth) * sv.speed_per_row
    ax = fig.add_subplot(gs[1:, 1])
    ax.plot(sv.level[:, craw], v, color="k", lw=1, label="density $d$")
    cc = cost[:, craw]
    ax.plot((cc - cc.min()) / (cc.max() - cc.min()), v, color="#1565c0", lw=1, label="edge cost (scaled)")
    ax.axvline(cfg.dp.edge_density, color="0.5", ls=":", lw=0.8)
    ax.axhline(res.v_max[c], color=C_MAX, lw=1, label="chosen edge")
    ax.axhline(truth["v_max"][c], color="0.3", ls="--", lw=0.8, label="true edge")
    ax.text(cfg.dp.edge_density, v[-1] * 0.98, r" $\theta$", va="top", fontsize=8)
    ax.set_ylim(0, v[-1])
    ax.set_xlim(0, 1.05)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("velocity (cm/s)")
    ax.set_title(f"(d) one column\n(t = {t[c]:.2f} s, positive side)", loc="left")
    ax.legend(loc="lower right", fontsize=6)
    save(fig, "dp_steps.pdf")
    plt.close(fig)
    ok = ~np.isnan(res.v_max)
    return float(np.mean(np.abs(np.r_[res.v_max[ok] - truth["v_max"][ok], res.v_min[ok] - truth["v_min"][ok]])))


def params_figure():
    """Effect of the two main DP parameters on one PW beat."""
    f = EXAMPLES["pw"]
    base = extract_waveforms(f)
    tr = base.t_rpeaks
    t0, t1 = (tr[0], tr[1]) if len(tr) >= 2 else (base.t[0], base.t[-1])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for a, key, vals, lab in [(ax[0], "edge_density", (0.3, 0.5, 0.7), r"$\theta$ ="),
                              (ax[1], "reference_accel_cm_s2", (160.0, 640.0, 2560.0), r"$a_{ref}$ =")]:
        show_display(a, base)
        for val, col in zip(vals, ("#00e676", C_MAX, "#ff9100")):
            r = extract_waveforms(f, {"dp": {key: val}})
            a.plot(r.t, r.v_max, color=col, lw=0.9, label=f"{lab} {val:g}")
        a.set_xlim(t0, t1)
        a.set_ylim(-10, None)
        a.set_xlabel("time (s)")
        a.legend(loc="upper right", framealpha=0.85)
    ax[0].set_title(r"(a) edge density $\theta$ (default 0.5)")
    ax[1].set_title(r"(b) acceleration prior $a_{ref}$ (default 640 cm/s$^2$)")
    ax[1].set_ylabel("")
    fig.tight_layout()
    save(fig, "params.pdf")
    plt.close(fig)


if __name__ == "__main__":
    if "--preview" in sys.argv:
        PREVIEW = sys.argv[sys.argv.index("--preview") + 1]
    print("dp synthetic MAE (cm/s):", dp_steps_figure())
    for key, title, kw in [
        ("pw", "Example 1: PW, mitral inflow", dict(avc=False)),
        ("cw", "Example 2: CW, aortic outflow", dict(avc=True, clicks=True)),
        ("tdi", "Example 3: tissue Doppler (colour map)", dict(avc=False)),
    ]:
        r = example_figure(key, title, **kw)
        print(key, r.flags["modality"], len(r.t_rpeaks), len(r.t_avc), r.meta["runtime_s"])
    print("vendor (MAE, bias, n):", vendor_figure())
    print("alias:", alias_figure())
    params_figure()
    print("written to", OUT)
