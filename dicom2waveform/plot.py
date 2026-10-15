"""QC figures.

Figures show only the spectral region and the ECG strip (no header area), so
burned-in patient identifiers are not reproduced. Still: do not commit them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline import WaveformResult


# (colour, line style, line width) of each signal, shared by plot_result and plot_overlay. Positive
# side warm, negative side cool; edges solid, means dashed. None of the colours is green or yellow,
# the usual colours of the scanner's own ECG trace, so the extracted ECG stays distinguishable from it.
OVERLAY_STYLE = {
    "v_max": ("#ff3b30", "-", 2.0),
    "v_mean_pos": ("#ff9f0a", "--", 1.6),
    "v_min": ("#0a84ff", "-", 2.0),
    "v_mean_neg": ("#64d2ff", "--", 1.6),
    "v_mean": ("#e5e5ea", "-", 1.3),  # plot_result's velocity panel only
    "ecg": ("#ff2bd6", "-", 1.8),
    "rpeak": ("#d0d0d0", ":", 1.2),
    "avc": ("white", "--", 1.8),
}

# plot_result: dark panels like the scanner display, so every signal has the same colour on the
# image and on the plain panels
PLOT_THEME = {
    "figure": "#16181d",
    "panel": "#0b0c0f",
    "text": "#e6e7ea",
    "muted": "#9a9ea6",
    "grid": "#2b2f36",
    "rpeak_guide": "#6e737c",  # the R-peak line through the panels, kept quieter than the signals
}


def _outlined(lw: float):
    # a black outline keeps every line visible on black, grey, white and copper displays
    from matplotlib import patheffects

    return [patheffects.Stroke(linewidth=lw + 2.2, foreground="black"), patheffects.Normal()]


def _no_flow_hidden(v):
    # side means are 0 where that side has no flow; do not draw them along the baseline
    v = np.asarray(v, float).copy()
    v[v == 0] = np.nan
    return v


def plot_result(res: WaveformResult, path: str | Path | None = None, title: str = "",
                extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                show_avc_on_tdi: bool = False, dpi: int = 130):
    """Analysed spectrum with the envelopes, the ECG with R-peaks and aortic valve closure (AVC),
    and the velocity signals, on one time axis. Colours: OVERLAY_STYLE.

    Works on any result (uses res.spectrum, no diagnostics needed). The figure shows the
    spectral strip and the burned-in ECG just above it (with the extracted ECG drawn over it, to
    check the tracing), not the header area with patient details.
    extra: optional {label: (v_max, v_min)} (time order) overlaid for comparison.
    AVC is hidden on TDI by default (no valve signal in tissue Doppler).
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = PLOT_THEME
    st = OVERLAY_STYLE
    f = res.flags
    t = res.t
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    t_lim = (t[0] - dt / 2, t[-1] + dt / 2)
    show_avc = f.get("modality") != "TDI" or show_avc_on_tdi

    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, facecolor=th["figure"],
                           gridspec_kw=dict(height_ratios=[3, 1.1, 2], hspace=0.08))
    for a in ax:
        a.set_facecolor(th["panel"])
        a.tick_params(colors=th["muted"], labelsize=9, length=3)
        for s in a.spines.values():
            s.set_color(th["grid"])
        a.yaxis.label.set_color(th["text"])
        a.xaxis.label.set_color(th["text"])
    for a in ax[1:]:
        a.grid(True, color=th["grid"], lw=0.6)
        a.set_axisbelow(True)

    def line(a, v, key, outline=False, **kw):
        col, ls, lw = st[key]
        a.plot(t, v, color=col, ls=ls, lw=lw, path_effects=_outlined(lw) if outline else None, **kw)

    # --- 1: spectrum as displayed (row 0 at the top), with the burned-in ECG above it, the extracted
    # ECG drawn over that, and the envelopes
    a = ax[0]
    sp = res.spectrum
    if sp is not None:
        a.imshow(sp.rgb, aspect="auto", extent=(*t_lim, sp.v_bottom, sp.v_top), origin="upper",
                 interpolation="auto")
        a.set_ylim(sp.v_bottom, sp.v_top)
        # velocity ticks on the spectrum only, not on the ECG rows above it
        from matplotlib.ticker import MaxNLocator

        v_edge = float(sp.velocity(sp.n_ecg_rows))
        lo_v, hi_v = sorted((v_edge, sp.v_bottom))
        a.set_yticks([v for v in MaxNLocator(nbins=7).tick_values(lo_v, hi_v) if lo_v <= v <= hi_v])
        if sp.ecg_row is not None:
            col, ls, _ = st["ecg"]
            a.plot(t, sp.velocity(sp.ecg_row), color=col, ls=ls, lw=1.1)  # thin: the scanner's trace stays visible
    if f.get("aliasing_corrected"):
        for v in res.meta.get("velocity_limits_cm_s", ()):  # the scanner's display window
            a.axhline(v, color=th["text"], ls=(0, (1, 3)), lw=1.0)
    for key, v in (("v_mean_pos", _no_flow_hidden(res.v_mean_pos)), ("v_mean_neg", _no_flow_hidden(res.v_mean_neg)),
                   ("v_max", res.v_max), ("v_min", res.v_min)):
        line(a, v, key, outline=True)
    extra_colours = ("#bf5af2", "#ffd60a", "#30d158", "#ffffff")
    for i, (lab, (vmx, vmn)) in enumerate((extra or {}).items()):
        c = extra_colours[i % len(extra_colours)]
        a.plot(t, vmx, color=c, lw=1.2, ls=(0, (4, 2)), path_effects=_outlined(1.2))
        a.plot(t, vmn, color=c, lw=1.2, ls=(0, (4, 2)), path_effects=_outlined(1.2))
    a.set_ylabel("velocity (cm/s)")

    # --- 2: ECG; R-peak dots, AVC dashed
    a = ax[1]
    line(a, res.ecg, "ecg")
    if len(res.t_rpeaks):
        ecg_r = np.interp(res.t_rpeaks, t, np.nan_to_num(res.ecg))
        a.plot(res.t_rpeaks, ecg_r, ls="", marker="o", ms=6, color=st["ecg"][0], mec=th["text"], mew=1.0)
    ecg_ok = np.isfinite(res.ecg).any()
    lo, hi = (np.nanmin(res.ecg), np.nanmax(res.ecg)) if ecg_ok else (-1.0, 1.0)
    pad = 0.12 * (hi - lo or 1.0)
    a.set_ylim(lo - pad, hi + pad)
    a.set_ylabel("ECG (a.u.)")
    if f.get("ecg_clipped"):
        a.text(0.995, 0.95, "clipped by display", transform=a.transAxes, ha="right", va="top",
               color=th["muted"], fontsize=8)

    # --- 3: velocity signals without the image
    a = ax[2]
    a.axhline(0, color=th["muted"], lw=0.8)
    for key, v in (("v_mean_pos", _no_flow_hidden(res.v_mean_pos)), ("v_mean_neg", _no_flow_hidden(res.v_mean_neg)),
                   ("v_mean", res.v_mean), ("v_max", res.v_max), ("v_min", res.v_min)):
        line(a, v, key)
    a.set_ylabel("velocity (cm/s)")
    a.set_xlabel("time (s)")
    a.set_xlim(*t_lim)

    # events through all panels: R-peaks as a quiet dotted guide, AVC dashed
    for a in ax:
        for tr in res.t_rpeaks:
            a.axvline(tr, color=th["rpeak_guide"], ls=st["rpeak"][1], lw=1.0, zorder=1.5)
        if show_avc:
            col, ls, lw = st["avc"]
            for ta in res.t_avc:
                a.axvline(ta, color=col, ls=ls, lw=1.3, path_effects=_outlined(1.3) if a is ax[0] else None,
                          zorder=1.6)

    # title and one legend for all panels (same colours everywhere)
    alias = ("aliasing corrected" if f.get("aliasing_corrected")
             else "aliasing suspected" if f.get("aliasing_suspected") else "")
    info = "   ".join(s for s in (
        str(f.get("modality", "")), f"SNR proxy {f.get('snr_proxy_db', float('nan')):.1f} dB",
        f"{len(res.t_rpeaks)} R-peaks",
        f"{len(res.t_avc)} AVC" if show_avc else "", alias) if s)
    fig.text(0.06, 0.975, title or Path(str(res.meta.get("source", ""))).name, color=th["text"],
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.06, 0.948, info, color=th["muted"], fontsize=9.5, va="top")
    labels = {"v_max": "v_max", "v_mean_pos": "mean (pos)", "v_min": "v_min", "v_mean_neg": "mean (neg)",
              "v_mean": "v_mean", "ecg": "ECG"}
    handles = [Line2D([], [], color=st[k][0], ls=st[k][1], lw=st[k][2], label=lab) for k, lab in labels.items()]
    handles.append(Line2D([], [], color=th["rpeak_guide"], ls=st["rpeak"][1], lw=1.2, marker="o", ms=5,
                          mfc=st["ecg"][0], mec=th["text"], label="R-peak"))
    if show_avc:
        handles.append(Line2D([], [], color=st["avc"][0], ls=st["avc"][1], lw=1.3, label="AVC"))
    handles += [Line2D([], [], color=extra_colours[i % len(extra_colours)], ls=(0, (4, 2)), lw=1.2, label=lab)
                for i, lab in enumerate(extra or {})]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=(len(handles) + 1) // 2,
               fontsize=8.5, frameon=False, labelcolor=th["text"], handlelength=2.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.905, bottom=0.06)
    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor=th["figure"])
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
    # grey displays: the drawn grey level itself (spec.level has the overlays
    # filled); colour maps: the converted level
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
    v_max / v_min, the side means v_mean_pos / v_mean_neg (dashed), the extracted ECG trace,
    R-peaks and AVC (colours: OVERLAY_STYLE).

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
    outlined, no_flow_hidden = _outlined, _no_flow_hidden

    def draw(a):
        for v, (col, ls, lw) in ((res.v_max, OVERLAY_STYLE["v_max"]), (res.v_min, OVERLAY_STYLE["v_min"]),
                                 (no_flow_hidden(res.v_mean_pos), OVERLAY_STYLE["v_mean_pos"]),
                                 (no_flow_hidden(res.v_mean_neg), OVERLAY_STYLE["v_mean_neg"])):
            a.plot(x, rows(v), color=col, ls=ls, lw=lw, path_effects=outlined(lw))
        if trace.found:
            r = trace.row.astype(float).copy()
            r[spec.sweep] = np.nan
            col, ls, lw = OVERLAY_STYLE["ecg"]
            a.plot(x, r, color=col, ls=ls, lw=lw, path_effects=outlined(lw))
        col, ls, lw = OVERLAY_STYLE["rpeak"]
        for xr in (columns(res.t_rpeaks) if len(res.t_rpeaks) else []):
            a.plot([xr, xr], [ecg_top, reg.y1], color=col, lw=lw, ls=ls, path_effects=outlined(lw))
        if show_avc:
            col, ls, lw = OVERLAY_STYLE["avc"]
            for xa in (columns(res.t_avc) if len(res.t_avc) else []):
                a.plot([xa, xa], [reg.y0, reg.y1], color=col, lw=lw, ls=ls, path_effects=outlined(lw))

    draw(ax)
    if panel is not None:
        draw(axp)
    ax.set_xlim(-0.5, Wimg - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")
    if panel is not None:
        axp.set_xlim(-0.5, Wimg - 0.5)
        axp.set_ylim(r1 - 0.5, r0 - 0.5)
    labels = {"v_max": "v_max", "v_min": "v_min", "v_mean_pos": "mean (pos)", "v_mean_neg": "mean (neg)",
              "ecg": "ECG (extracted)", "rpeak": "R-peak"}
    if show_avc:
        labels["avc"] = "AVC"
    handles = [Line2D([], [], color=OVERLAY_STYLE[k][0], ls=OVERLAY_STYLE[k][1], lw=OVERLAY_STYLE[k][2], label=lab,
                      path_effects=outlined(OVERLAY_STYLE[k][2])) for k, lab in labels.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.7, facecolor="black",
              labelcolor="white", ncol=len(handles))
    if title:
        ax.text(0.5, 0.995, title, transform=ax.transAxes, color="white", fontsize=10, ha="center", va="top",
                bbox=dict(facecolor="black", alpha=0.6, lw=0))
    if path is not None:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig
