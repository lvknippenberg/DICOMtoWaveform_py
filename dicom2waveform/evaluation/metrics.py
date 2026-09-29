"""Evaluation metrics (Plan sections 26-31, 50-51).

All functions take time-ordered series with NaN for samples without data.

Primary endpoint (fixed before seeing results): normalised envelope MAE =
MAE / display velocity span, over valid samples, for v_max and v_min pooled
('env_nmae').
"""

from __future__ import annotations

import numpy as np

SIGNALS = ("v_max", "v_min", "v_mean", "v_mean_pos", "v_mean_neg")


def _valid(est, ref, mask=None):
    m = ~np.isnan(est) & ~np.isnan(ref)
    if mask is not None:
        m &= mask
    return m


def pointwise(est: np.ndarray, ref: np.ndarray, span: float, px: float, mask=None,
              rel_min: float | None = None) -> dict[str, float]:
    """Pointwise errors between an estimated and a reference series (cm/s).

    span: display velocity span (cm/s), for normalisation.
    px: velocity per pixel, sets the over/under-estimation tolerance (2 px).
    """
    m = _valid(est, ref, mask)
    out: dict[str, float] = {}
    if m.sum() < 3:
        return {k: np.nan for k in ("mae", "rmse", "bias", "medae", "p95ae", "maxae", "nmae", "rel_err",
                                   "corr", "r2", "over", "under", "s1", "s2")}
    e, r = est[m], ref[m]
    d = e - r
    ad = np.abs(d)
    out["mae"] = float(ad.mean())
    out["rmse"] = float(np.sqrt((d ** 2).mean()))
    out["bias"] = float(d.mean())
    out["medae"] = float(np.median(ad))
    out["p95ae"] = float(np.percentile(ad, 95))
    out["maxae"] = float(ad.max())
    out["nmae"] = out["mae"] / span
    rel_min = rel_min if rel_min is not None else 0.05 * span
    big = np.abs(r) > rel_min
    out["rel_err"] = float(np.mean(ad[big] / np.abs(r[big]))) if big.sum() >= 3 else np.nan
    if np.std(e) > 0 and np.std(r) > 0:
        out["corr"] = float(np.corrcoef(e, r)[0, 1])
        out["r2"] = float(1 - np.sum(d ** 2) / np.sum((r - r.mean()) ** 2))
    else:
        out["corr"] = out["r2"] = np.nan
    delta = max(2 * px, 0.01 * span)
    # over/under in terms of magnitude, so it means the same on both sides
    sgn = np.sign(r + 1e-12 * np.sign(e))
    sgn[sgn == 0] = 1
    out["over"] = float(np.mean(sgn * d > delta))
    out["under"] = float(np.mean(sgn * d < -delta))
    out.update(smoothness(est))
    return out


def smoothness(v: np.ndarray) -> dict[str, float]:
    """Mean |first| and |second| differences (cm/s per sample), within data segments."""
    d1 = np.diff(v)
    d2 = v[2:] - 2 * v[1:-1] + v[:-2]
    return {"s1": float(np.nanmean(np.abs(d1))) if np.any(~np.isnan(d1)) else np.nan,
            "s2": float(np.nanmean(np.abs(d2))) if np.any(~np.isnan(d2)) else np.nan}


def beat_windows(t: np.ndarray, r_times: np.ndarray) -> list[tuple[int, int]]:
    """Sample index ranges [a, b) between consecutive R-peaks."""
    idx = np.searchsorted(t, r_times)
    return [(int(a), int(b)) for a, b in zip(idx[:-1], idx[1:]) if b - a > 5]


def _beat_features(v: np.ndarray, t: np.ndarray, a: int, b: int, frac: float = 0.2):
    seg = np.abs(v[a:b])
    if np.all(np.isnan(seg)) or np.nanmax(seg) <= 0:
        return None
    pk = int(np.nanargmax(seg))
    vp = float(seg[pk])
    above = np.flatnonzero(np.nan_to_num(seg) >= frac * vp)
    dt = t[1] - t[0]
    return {"peak": vp, "t_peak": t[a + pk], "t_onset": t[a + above[0]], "t_end": t[a + above[-1]],
            "vti": float(np.nansum(seg) * dt)}


def per_beat(est: np.ndarray, ref: np.ndarray, t: np.ndarray, r_times: np.ndarray) -> dict[str, float]:
    """Clinically derived quantities per beat: peak velocity, VTI and timing."""
    errs = {"peak_err": [], "peak_abs_err": [], "vti_err": [], "vti_rel_err": [],
            "t_peak_err": [], "t_onset_err": [], "t_end_err": []}
    for a, b in beat_windows(t, r_times):
        fr = _beat_features(ref, t, a, b)
        if fr is None or fr["peak"] < 1e-6:
            continue
        fe = _beat_features(est, t, a, b)
        if fe is None:
            fe = {"peak": 0.0, "vti": 0.0, "t_peak": np.nan, "t_onset": np.nan, "t_end": np.nan}
        errs["peak_err"].append(fe["peak"] - fr["peak"])
        errs["peak_abs_err"].append(abs(fe["peak"] - fr["peak"]))
        errs["vti_err"].append(fe["vti"] - fr["vti"])
        errs["vti_rel_err"].append(abs(fe["vti"] - fr["vti"]) / max(fr["vti"], 1e-9))
        for k in ("t_peak", "t_onset", "t_end"):
            errs[f"{k}_err"].append(abs(fe[k] - fr[k]))
    out = {}
    for k, v in errs.items():
        out[k] = float(np.nanmean(v)) if len(v) and np.any(~np.isnan(v)) else np.nan
    out["n_beats"] = len(errs["peak_err"])
    return out


def evaluate_signals(est: dict[str, np.ndarray], ref: dict[str, np.ndarray], t: np.ndarray,
                     span: float, px: float, r_times: np.ndarray | None = None,
                     mask: np.ndarray | None = None, signals=SIGNALS) -> dict[str, float]:
    """Flat dict 'signal.metric' for all signals, plus the primary endpoint."""
    out: dict[str, float] = {}
    for s in signals:
        if s not in est or s not in ref:
            continue
        for k, v in pointwise(est[s], ref[s], span, px, mask).items():
            out[f"{s}.{k}"] = v
        if r_times is not None and len(r_times) > 2 and s in ("v_max", "v_min"):
            for k, v in per_beat(est[s], ref[s], t, r_times).items():
                out[f"{s}.{k}"] = v
    if "v_max.mae" in out and "v_min.mae" in out:
        out["env_nmae"] = float(np.nanmean([out["v_max.nmae"], out["v_min.nmae"]]))
        out["env_mae"] = float(np.nanmean([out["v_max.mae"], out["v_min.mae"]]))
    if "v_mean.mae" in out:
        out["mean_nmae"] = float(np.nanmean([out.get(f"{s}.nmae", np.nan)
                                             for s in ("v_mean", "v_mean_pos", "v_mean_neg")]))
    return out


def bland_altman(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    m = ~np.isnan(est) & ~np.isnan(ref)
    d = est[m] - ref[m]
    if len(d) < 2:
        return {"bias": np.nan, "sd": np.nan, "loa_low": np.nan, "loa_high": np.nan, "n": len(d)}
    b, sd = float(d.mean()), float(d.std(ddof=1))
    return {"bias": b, "sd": sd, "loa_low": b - 1.96 * sd, "loa_high": b + 1.96 * sd, "n": int(len(d))}


def bootstrap_paired(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, stat=np.median,
                     seed: int = 0) -> dict[str, float]:
    """Paired difference a - b (per image): point estimate and 95% bootstrap CI."""
    m = ~np.isnan(a) & ~np.isnan(b)
    d = np.asarray(a)[m] - np.asarray(b)[m]
    if len(d) < 3:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(d)}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)])
    return {"diff": float(stat(d)), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(d)),
            "frac_a_better": float(np.mean(d < 0))}


def summary_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {k: np.nan for k in ("mean", "median", "sd", "iqr", "p95", "worst")}
    q1, q3 = np.percentile(x, [25, 75])
    return {"mean": float(x.mean()), "median": float(np.median(x)), "sd": float(x.std()),
            "iqr": float(q3 - q1), "p95": float(np.percentile(x, 95)), "worst": float(x.max())}
