"""Build the benchmark report (markdown + figures) from the benchmark CSVs.

Usage: python scripts/make_report.py <results_dir>
Reads synthetic_raw.csv, vendor_raw.csv and (optionally) real_raw.csv /
real_pairs.csv; writes <results_dir>/report/REPORT.md and figures.
Figures contain only synthetic images and aggregate numbers (no patient data).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dicom2waveform.evaluation.metrics import bootstrap_paired  # noqa: E402

# validated categorical order (dataviz reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#52514e"
IMAGE_METHODS = ["threshold", "otsu", "latham", "dp"]
SPECTRUM_METHODS = ["percentile", "gm", "mgm", "snsi", "asnsi", "qia"]
MAIN = IMAGE_METHODS[:3] + SPECTRUM_METHODS + ["dp"]
LABEL = {"threshold": "Fixed threshold", "otsu": "Otsu + morphology", "percentile": "Percentile (IPS)",
         "gm": "GM", "mgm": "MGM", "snsi": "SNSI", "asnsi": "Adaptive SNSI (Kathpalia)",
         "qia": "QIA (Li)", "latham": "OMM (Latham)", "dp": "DP (this work)",
         "A_colwise_asnsi_raw": "A: column-wise", "B_colwise_asnsi_median": "B: + 1-D smoothing",
         "C_segmentation": "C: 2-D segmentation", "D_dp": "D: 2-D DP"}
FAIL_NMAE = 0.05  # failure: mean |error| over 5% of the display span


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def fig_new(w, h, n=1, m=1, **kw):
    fig, ax = plt.subplots(n, m, figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, ax


def snr_curves(df, methods, fname, out, ylabel="Envelope error (% of display span, median)", metric="env_nmae",
               ref="dp", title=""):
    mods = ["PW", "CW", "TDI"]
    fig, axes = fig_new(12, 3.6, 1, 3, sharey=True)
    for ax, mod in zip(axes, mods):
        d = df[df.modality == mod]
        for i, m in enumerate(methods):
            g = d[d.variant == m].groupby("snr_db")[metric].median() * 100
            if g.empty:
                continue
            ax.plot(g.index, g.values, color=SLOTS[i], lw=2, marker="o", ms=4, label=LABEL.get(m, m))
            ax.annotate(LABEL.get(m, m).split(" (")[0], (g.index[-1], g.values[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=INK2, va="center")
        if ref and ref not in methods:
            g = d[d.variant == ref].groupby("snr_db")[metric].median() * 100
            ax.plot(g.index, g.values, color=REF, lw=2, ls="--", label=LABEL[ref] + " (reference)")
        style(ax, f"{mod}")
        ax.set_xlabel("synthetic SNR (dB)", color=INK2, fontsize=8)
        ax.set_yscale("log")
    axes[0].set_ylabel(ylabel, color=INK2, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=min(len(lab), 7), frameon=False,
               fontsize=8, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11, y=0.99)
    fig.savefig(out / fname, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def hbar(values: pd.Series, fname, out, xlabel, title, highlight=("dp",)):
    values = values.sort_values(ascending=False)
    fig, ax = fig_new(7, 0.32 * len(values) + 1)
    colors = [SLOTS[0] if k in highlight else "#9d9c97" for k in values.index]
    ax.barh(range(len(values)), values.values, color=colors, height=0.7)
    for i, v in enumerate(values.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([LABEL.get(k, k) for k in values.index], fontsize=8, color=INK)
    style(ax, title)
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def main(argv=None):
    argv = argv or sys.argv[1:]
    res = Path(argv[0])
    out = res / "report"
    out.mkdir(parents=True, exist_ok=True)
    syn = pd.read_csv(res / "synthetic_raw.csv")
    if "error" in syn:
        n_err = syn["error"].notna().sum()
        syn = syn[syn["error"].isna()]
    else:
        n_err = 0
    syn["failure"] = syn["env_nmae"] > FAIL_NMAE
    syn["snr_band"] = pd.cut(syn["snr_db"], [-99, 3, 15, 99], labels=["low (<=3 dB)", "medium (6-15 dB)", "high (>=20 dB)"])
    test = syn[syn.split == "test"]
    lines = []
    w = lines.append
    w("# Doppler envelope benchmark report\n")
    w("Generated by `scripts/make_report.py`. Primary endpoint (fixed before the benchmark was run): "
      "**normalised envelope MAE** = mean |error| of v_max and v_min, divided by the display velocity span, "
      "over samples with data. Synthetic results are reported on the held-out split (seed 1) unless noted; "
      f"the dev split (seed 0) is shown alongside. Failure = normalised MAE > {FAIL_NMAE:.0%}. "
      f"{len(syn.case.unique())} synthetic cases, {n_err} failed runs.\n")

    # ------------------------------------------------------------------ main table
    main = test[test.variant.isin(MAIN)]
    agg = main.groupby("variant").agg(
        nmae_median=("env_nmae", "median"), nmae_mean=("env_nmae", "mean"),
        nmae_p95=("env_nmae", lambda x: np.nanpercentile(x, 95)),
        mae_cm_s=("env_mae", "median"), vmax_rmse=("v_max.rmse", "median"),
        bias_vmax=("v_max.bias", "median"), peak_abs_err=("v_max.peak_abs_err", "median"),
        vti_rel_err=("v_max.vti_rel_err", "median"), t_peak_err_ms=("v_max.t_peak_err", lambda x: 1000 * np.nanmedian(x)),
        failure=("failure", "mean"), mean_env_nmae=("mean_nmae", "median"), runtime_ms=("runtime_s", lambda x: 1000 * x.mean()))
    low = main[main.snr_db <= 3].groupby("variant").env_nmae.median().rename("low_snr_nmae")
    agg = agg.join(low).loc[[m for m in MAIN if m in agg.index]]
    agg.index = [LABEL[m] for m in agg.index]
    w("## 1. Main benchmark (synthetic, held-out split)\n")
    w(md_table(agg.sort_values("nmae_median")))
    w("\n*nmae* = normalised envelope MAE; *mae_cm_s* = median MAE in cm/s (depends on modality range); "
      "*peak_abs_err* = per-beat |peak velocity error| of v_max (cm/s); *vti_rel_err* = per-beat relative VTI error; "
      "*t_peak_err_ms* = per-beat time-to-peak error; *mean_env_nmae* = normalised MAE of the three mean envelopes. "
      "*runtime_ms* is the estimator time measured while 12 benchmark processes ran in parallel; the full "
      "pipeline takes ~0.2 s per 1024x768 image on an idle machine.\n")

    for title, sel in (("Spectral Doppler (PW + CW)", ~main.tissue), ("Tissue Doppler (TDI)", main.tissue)):
        t = main[sel].pivot_table(index="variant", columns="snr_band", values="env_nmae", aggfunc="median", observed=False)
        t = t.loc[[m for m in MAIN if m in t.index]]
        t.index = [LABEL[m] for m in t.index]
        w(f"\n### {title}: median normalised envelope MAE by SNR band\n")
        w(md_table(t))

    snr_curves(syn[syn.variant.isin(IMAGE_METHODS)], IMAGE_METHODS, "fig_snr_image.png", out, ref=None,
               title="Image-domain methods: error vs SNR (all seeds)")
    snr_curves(syn[syn.variant.isin(SPECTRUM_METHODS + ["dp"])], SPECTRUM_METHODS, "fig_snr_spectrum.png", out,
               title="Spectrum-domain (IPS) methods vs SNR, DP dashed for reference (all seeds)")
    w("\n![image methods vs SNR](fig_snr_image.png)\n\n![spectrum methods vs SNR](fig_snr_spectrum.png)\n")

    # ------------------------------------------------------------------ paired comparisons
    w("\n## 2. Paired comparison with DP (held-out split, per-image normalised MAE)\n")
    piv = test.pivot_table(index="case", columns="variant", values="env_nmae")
    rows = []
    for m in MAIN:
        if m == "dp" or m not in piv:
            continue
        b = bootstrap_paired(piv["dp"].values, piv[m].values)
        rows.append({"method": LABEL[m], "median diff (DP - method)": b["diff"], "95% CI low": b["ci_low"],
                     "95% CI high": b["ci_high"], "fraction of images DP better": b["frac_a_better"], "n": b["n"]})
    w(md_table(pd.DataFrame(rows).set_index("method"), ".4f"))

    # ------------------------------------------------------------------ factorial
    w("\n## 3. Spectral vs tissue Doppler x SNR interaction\n")
    w("Delta = (MAE_TDI,low - MAE_TDI,high) - (MAE_spec,low - MAE_spec,high), medians of normalised MAE; "
      "low = SNR <= 3 dB, high = SNR >= 20 dB. Positive: TDI degrades more with falling SNR.\n")
    fr = []
    for m in MAIN:
        d = syn[syn.variant == m]
        q = lambda tis, lo: d[(d.tissue == tis) & ((d.snr_db <= 3) if lo else (d.snr_db >= 20))].env_nmae.median()  # noqa: E731
        fr.append({"method": LABEL[m], "spec high": q(False, False), "spec low": q(False, True),
                   "TDI high": q(True, False), "TDI low": q(True, True),
                   "Delta": (q(True, True) - q(True, False)) - (q(False, True) - q(False, False))})
    w(md_table(pd.DataFrame(fr).set_index("method"), ".4f"))

    # ------------------------------------------------------------------ 2-D experiment
    w("\n## 4. Does 2-D information help? (Plan section 54)\n")
    d2 = syn[syn.group == "2d"]
    order = sorted(d2.variant.unique())
    t = d2.pivot_table(index="variant", columns="snr_band", values="env_nmae", aggfunc="median", observed=False).loc[order]
    w("A = independent column-wise estimate, B = A + 1-D smoothing, C = 2-D segmentation + boundary, "
      "D = 2-D likelihood + globally optimised curve. All seeds.\n")
    w(md_table(t))
    pick = [v for v in ("A_colwise_asnsi_raw", "B_colwise_asnsi_median", "C_segmentation", "D_dp") if v in order]
    snr_curves(d2, pick, "fig_2d.png", out, ref=None, title="Column-wise (A, B) vs 2-D (C, D) estimation")
    w("\n![2-D experiment](fig_2d.png)\n")

    # ------------------------------------------------------------------ ablation
    w("\n## 5. Ablations (synthetic, all seeds; vendor traces where available)\n")
    ab = syn[syn.group.isin(["ablation"]) | (syn.variant == "dp")]
    t = ab.groupby("variant").agg(synthetic_nmae_median=("env_nmae", "median"),
                                  synthetic_nmae_mean=("env_nmae", "mean"),
                                  low_snr=("env_nmae", lambda x: np.nan),
                                  mean_env_nmae=("mean_nmae", "median"))
    t["low_snr"] = ab[ab.snr_db <= 3].groupby("variant").env_nmae.median()
    vend = None
    if (res / "vendor_raw.csv").exists():
        vend = pd.read_csv(res / "vendor_raw.csv")
        vend = vend[vend.get("error").isna()] if "error" in vend else vend
        t["vendor_mae_cm_s"] = vend.groupby("variant")["trace.mae"].mean()
    w(md_table(t))

    # ------------------------------------------------------------------ vendor
    if vend is not None:
        w("\n## 6. Real images with vendor ground truth (5 vascular, Philips iU22)\n")
        w("Vendor auto-trace drawn on the display, masked out before estimation. Per column the trace is "
          "compared with v_max (trace > 0) or v_min (trace < 0). DP edge density was chosen by "
          "leave-one-image-out on these images (held-out MAE 2.16 cm/s); all other methods use published/default settings.\n")
        vm = vend[vend.variant.isin(MAIN)]
        tab = vm.pivot_table(index="variant", columns="case", values="trace.mae")
        tab["mean"] = tab.mean(axis=1)
        tab["bias"] = vm.groupby("variant")["trace.bias"].mean()
        tab["corr"] = vm.groupby("variant")["trace.corr"].mean()
        tab.index = [LABEL[m] for m in tab.index]
        w(md_table(tab.sort_values("mean"), ".2f"))
        hbar(vm.groupby("variant")["trace.mae"].mean(), "fig_vendor.png", out, "MAE vs vendor trace (cm/s, mean of 5 images)",
             "Real images: agreement with the scanner's own envelope trace")
        w("\n![vendor](fig_vendor.png)\n")

    # ------------------------------------------------------------------ real, unsupervised
    if (res / "real_raw.csv").exists():
        real = pd.read_csv(res / "real_raw.csv")
        real = real[real["error"].isna()] if "error" in real else real
        w("\n## 7. Real cardiac images without reference (unsupervised)\n")
        w(f"{real.case.nunique()} images. Without ground truth only consistency can be measured: agreement "
          "with the other methods (median over methods of the mean |difference|, cm/s), sample-to-sample jitter "
          "(S1, cm/s per sample) and runtime. Low jitter alone is not accuracy.\n")
        rm = real[real.variant.isin(MAIN)]
        t = rm.groupby("variant").agg(jitter_vmax=("v_max.s1", "median"), jitter_vmin=("v_min.s1", "median"),
                                      runtime_ms=("runtime_s", lambda x: 1000 * x.mean()))
        if (res / "real_pairs.csv").exists():
            pr = pd.read_csv(res / "real_pairs.csv")
            pr = pr[pr.a.isin(MAIN) & pr.b.isin(MAIN)]
            both = pd.concat([pr.rename(columns={"a": "m"})[["case", "m", "env_mad"]],
                              pr.rename(columns={"b": "m"})[["case", "m", "env_mad"]]])
            t["median_disagreement_cm_s"] = both.groupby("m").env_mad.median()
            dp_pairs = pr[(pr.a == "dp") | (pr.b == "dp")].copy()
            dp_pairs["other"] = np.where(dp_pairs.a == "dp", dp_pairs.b, dp_pairs.a)
            t["vs_dp_cm_s"] = dp_pairs.groupby("other").env_mad.median()
        t.index = [LABEL[m] for m in t.index]
        w(md_table(t, ".2f"))
        if "snr_proxy_db" in real:
            pass

    # ------------------------------------------------------------------ SNR proxy mapping
    w("\n## 8. How hard are the real images? (image-domain SNR proxy)\n")
    sp = syn[syn.variant == "dp"].groupby("snr_db").snr_proxy_db.median()
    w("Median SNR proxy (dB) of the synthetic images per nominal SNR, to place real images on the synthetic "
      "SNR axis (the real images' proxies are listed in the QC summaries):\n")
    w(md_table(sp.to_frame("synthetic SNR proxy (dB)"), ".1f"))

    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main())
