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
    extent = (t[0], t[-1] + spec.dt, spec.v_limits[0] if spec.dy > 0 else spec.v_limits[0],
              spec.v_limits[1])
    # rows -> velocity for imshow: row 0 is at the top
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
    ax[0].set_title(f"{title}  [{f.get('modality')}, {res.meta.get('method')}, SNR proxy "
                    f"{f.get('snr_proxy_db', float('nan')):.1f} dB"
                    f"{', ALIASING?' if f.get('aliasing_suspected') else ''}]", fontsize=10)
    ax[1].plot(t, res.ecg, "g", lw=0.8)
    for tr in res.t_rpeaks:
        for a in ax:
            a.axvline(tr, color="k", ls=":", lw=0.6)
    ax[1].set_ylabel("ECG (a.u.)")
    ax[1].set_title("ECG" + (" (clipped by display)" if f.get("ecg_clipped") else ""), fontsize=9)
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
