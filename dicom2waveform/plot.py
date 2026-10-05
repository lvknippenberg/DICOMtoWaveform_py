"""QC figures.

Figures show only the spectral region and the ECG strip (no header area), so
burned-in patient identifiers are not reproduced. Still: do not commit them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline import WaveformResult


def plot_result(res: WaveformResult, path: str | Path | None = None, title: str = "",
                extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
    """Display with envelopes, the ECG with R-peaks, and the velocity signals.

    Requires a result computed with keep_diagnostics=True.
    extra: optional {label: (v_max, v_min)} (time order) overlaid for comparison.
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = res.diagnostics["spectrogram"]
    o = res.diagnostics["order"]
    rgb = spec.rgb[:, o]
    t = res.t
    # rows -> velocity for imshow: row 0 is at the top (after aliasing correction
    # the image is the unwrapped, taller display)
    v_top = (0 - spec.zero_row) * spec.dy
    v_bot = (rgb.shape[0] - 1 - spec.zero_row) * spec.dy
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1, 2]))
    ax[0].imshow(rgb, aspect="auto", extent=(t[0], t[-1] + spec.dt, v_bot, v_top),
                 origin="upper", interpolation="nearest")
    if v_top < v_bot:
        ax[0].invert_yaxis()
    ax[0].plot(t, res.v_max, "r", lw=1.0, label="v_max")
    ax[0].plot(t, res.v_min, "c", lw=1.0, label="v_min")
    ax[0].plot(t, res.v_mean_pos, "y", lw=0.8, label="v_mean_pos")
    ax[0].plot(t, res.v_mean_neg, "m", lw=0.8, label="v_mean_neg")
    if extra:
        for i, (lab, (vmx, vmn)) in enumerate(extra.items()):
            c = plt.cm.tab10(i % 10)
            ax[0].plot(t, vmx, color=c, lw=0.6, ls="--", label=lab)
            ax[0].plot(t, vmn, color=c, lw=0.6, ls="--")
    ax[0].set_ylabel("velocity (cm/s)")
    ax[0].legend(loc="upper right", fontsize=7, ncol=4)
    f = res.flags
    alias = ""
    if f.get("aliasing_corrected"):
        alias = ", aliasing corrected"
        lo, hi = spec.display_v_limits
        for v in (lo, hi):  # the scanner's display window
            ax[0].axhline(v, color="w", ls="--", lw=0.6)
    elif f.get("aliasing_suspected"):
        alias = ", ALIASING?"
    ax[0].set_title(f"{title}  [{f.get('modality')}, {res.meta.get('method')}, SNR proxy "
                    f"{f.get('snr_proxy_db', float('nan')):.1f} dB{alias}]", fontsize=10)
    ax[1].plot(t, res.ecg, "g", lw=0.8)
    for tr in res.t_rpeaks:
        for a in ax:
            a.axvline(tr, color="k", ls=":", lw=0.6)
    for tc in res.t_clicks:
        ax[1].axvline(tc, color="0.6", ls="--", lw=0.5)
    for ta in res.t_avc:
        ax[1].axvline(ta, color="m", ls="--", lw=0.9)
    ax[1].set_ylabel("ECG (a.u.)")
    ax[1].set_title("ECG" + (" (clipped by display)" if f.get("ecg_clipped") else "")
                    + "   R-peaks dotted, valve clicks dashed grey, aortic valve closure dashed magenta",
                    fontsize=9)
    ax[2].plot(t, res.v_max, "r", lw=1, label="v_max")
    ax[2].plot(t, res.v_min, "b", lw=1, label="v_min")
    ax[2].plot(t, res.v_mean, "k", lw=1, label="v_mean")
    ax[2].plot(t, res.v_mean_pos, color="orange", lw=0.8, label="v_mean_pos")
    ax[2].plot(t, res.v_mean_neg, color="purple", lw=0.8, label="v_mean_neg")
    ax[2].axhline(0, color="gray", lw=0.5)
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("velocity (cm/s)")
    ax[2].legend(loc="upper right", fontsize=7, ncol=5)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=80)
        plt.close(fig)
    return fig


def normalise_brightness(spec, target: float = 0.5, white_percentile: float = 99.5,
                         gamma_limits: tuple[float, float] = (0.35, 3.0)) -> np.ndarray:
    """Display-only brightness normalisation of the spectral region (H, W, 3 uint8).

    Scanner gain and compression make some displays very dark (faint TDI) and
    others nearly saturated (CW). On the display level (spec.level, colour maps
    already converted): black point = median of the region (the background),
    white point = the white_percentile of the signal pixels, then a gamma curve
    that maps the median signal pixel to `target` of full scale. Colour (TDI)
    displays (TDI) are redrawn with a smooth copper colour map (scaling the
    scanner's own colours clips them and shifts the hue). Overlays (baseline,
    text, ticks, ECG) and sweep columns are left as drawn. Does not affect the
    estimation, which normalises its own signal density.
    """
    rgb = spec.rgb.astype(float)
    # grey displays: the drawn grey level itself (spec.level may have click
    # columns bridged); colour maps: the converted level
    L = rgb.mean(axis=2) if spec.colormap == "grey" else spec.level.astype(float)
    use = spec.valid & ~spec.sweep[None, :]
    if spec.overlay is not None:
        use &= ~spec.overlay
    vals = L[use]
    if vals.size == 0:
        return spec.rgb.copy()
    black = float(np.percentile(vals, 50))
    top = float(np.percentile(vals, 99.9))
    signal = vals[vals > black + 0.1 * max(top - black, 1.0)]
    if signal.size < 50:
        return spec.rgb.copy()
    white = max(float(np.percentile(signal, white_percentile)), black + 30.0)
    n = np.clip((L - black) / (white - black), 0.0, 1.0)
    m = float(np.clip(np.median((signal - black) / (white - black)), 1e-3, 0.999))
    gamma = float(np.clip(np.log(target) / np.log(m), *gamma_limits))
    out = 255.0 * n ** gamma
    if spec.colormap == "grey":
        new = np.repeat(out[..., None], 3, axis=2)
    else:
        from matplotlib import colormaps

        new = 255.0 * colormaps["copper"](out / 255.0)[..., :3]
    new = np.where(use[..., None], new, rgb)
    return np.clip(np.round(new), 0, 255).astype(np.uint8)


def plot_overlay(res: WaveformResult, image, path: str | Path | None = None, title: str = "",
                 target_width: int = 2000, show_avc_on_tdi: bool = False, normalised_panel: bool = True):
    """Full DICOM frame with the outputs drawn on top, for clinical review:
    v_max / v_min, the extracted ECG trace, R-peaks, valve clicks and AVC.

    Shows the whole frame, including burned-in patient details: for use within
    the institution only, never commit the figures. Requires a result computed
    with keep_diagnostics=True; `image` is the DicomImage it was computed from.
    AVC is hidden on TDI by default (no valve signal in tissue Doppler).
    The overlay is drawn on the frame as displayed (original brightness).
    normalised_panel: add a copy of the ECG + spectrum strip below the frame,
    column-aligned, with the spectrum brightness-normalised (normalise_brightness)
    and the same overlay.
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    spec = res.diagnostics["spectrogram"]
    trace = res.diagnostics["ecg_trace"]
    reg = spec.region
    o = spec.order
    W = len(o)
    z0 = spec.info.get("display_zero_row", spec.zero_row)
    x = reg.x0 + np.arange(W)

    def by_column(a):
        out = np.full(W, np.nan)
        out[o] = a
        return out

    def rows(v):
        return reg.y0 + z0 + by_column(v) / spec.dy

    def columns(times):
        idx = np.clip(np.round(np.asarray(times) / spec.dt).astype(int), 0, W - 1)
        return reg.x0 + o[idx]

    px = image.pixels if image.is_rgb else np.repeat(image.pixels[..., None], 3, axis=2)
    H, Wimg = px.shape[:2]
    scale = max(1, int(round(target_width / Wimg)))
    ecg_top = np.nanmin(trace.row) - 10 if trace.found and np.isfinite(trace.row).any() else reg.y0
    panel = None
    disp = res.diagnostics.get("display_spectrogram", spec)
    if normalised_panel and disp.rgb is not None:
        r0, r1 = int(max(0, min(ecg_top, reg.y0) - 10)), int(min(H, reg.y1 + 12))
        norm = px.copy()
        norm[reg.y0:reg.y1 + 1, reg.x0:reg.x1 + 1] = normalise_brightness(disp)
        panel = norm[r0:r1]
    gap = 6 if panel is not None else 0
    Htot = H + gap + (panel.shape[0] if panel is not None else 0)
    fig = plt.figure(figsize=(Wimg * scale / 100, Htot * scale / 100), dpi=100, facecolor="#3c3c3c")
    ax = fig.add_axes([0, 1 - H / Htot, 1, H / Htot])
    ax.imshow(px, extent=(-0.5, Wimg - 0.5, H - 0.5, -0.5), interpolation="nearest")
    if panel is not None:
        hp = panel.shape[0]
        axp = fig.add_axes([0, 0, 1, hp / Htot])
        axp.imshow(panel, extent=(-0.5, Wimg - 0.5, r1 - 0.5, r0 - 0.5), interpolation="nearest")
        axp.axis("off")
        axp.text(0.005, 0.99, "spectrum with normalised brightness (display only)", transform=axp.transAxes,
                 color="white", fontsize=9, va="top", bbox=dict(facecolor="black", alpha=0.6, lw=0))
    show_avc = spec.modality != "TDI" or show_avc_on_tdi

    def draw(a):
        lw = 1.0
        a.plot(x, rows(res.v_max), color="#ffd400", lw=lw)
        a.plot(x, rows(res.v_min), color="#ff3cff", lw=lw)
        if trace.found:
            r = trace.row.astype(float).copy()
            r[spec.sweep] = np.nan
            a.plot(x, r, color="red", lw=lw)
        for xr in (columns(res.t_rpeaks) if len(res.t_rpeaks) else []):
            a.plot([xr, xr], [ecg_top, reg.y1], color="white", lw=0.8, ls=":")
        if show_avc:
            for xa in (columns(res.t_avc) if len(res.t_avc) else []):
                a.plot([xa, xa], [reg.y0, reg.y1], color="#00e5ff", lw=1.0, ls="--")
        for xc in (columns(res.t_clicks) if len(res.t_clicks) else []):
            a.plot([xc], [reg.y1 + 6], marker="^", color="white", ms=6, mec="black", mew=0.5)

    draw(ax)
    if panel is not None:
        draw(axp)
    ax.set_xlim(-0.5, Wimg - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")
    if panel is not None:
        axp.set_xlim(-0.5, Wimg - 0.5)
        axp.set_ylim(r1 - 0.5, r0 - 0.5)
    handles = [Line2D([], [], color="#ffd400", label="v_max"), Line2D([], [], color="#ff3cff", label="v_min"),
               Line2D([], [], color="red", label="ECG (extracted)"),
               Line2D([], [], color="white", ls=":", label="R-peak"),
               Line2D([], [], color="white", marker="^", ls="", label="valve click")]
    if show_avc:
        handles.append(Line2D([], [], color="#00e5ff", ls="--", label="AVC"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.7, facecolor="black",
              labelcolor="white", ncol=len(handles))
    if title:
        ax.text(0.5, 0.995, title, transform=ax.transAxes, color="white", fontsize=10, ha="center", va="top",
                bbox=dict(facecolor="black", alpha=0.6, lw=0))
    if path is not None:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig
