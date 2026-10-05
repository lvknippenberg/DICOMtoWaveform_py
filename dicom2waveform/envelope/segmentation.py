"""Class B: image-domain estimators (segmentation + boundary extraction).

`SegmentationEstimator` implements the stage decomposition of Sulas et al.
2021 (preprocessing -> binarisation -> post-processing -> envelope
extraction) with every stage configurable, so that stages can be ablated:

    preprocess : none | gaussian | median | background | tophat        (P0-P4)
    binarize   : fixed | otsu | percentile | local | background |
                 column_max | relative_max                               (B1-B5+)
    morphology : list of [op, size] with op in open, close, dilate, erode,
                 fill, remove_small                                      (sec. 17)
    components : none | baseline | plausible | largest                  (sec. 18)
    extract    : outermost | largest_run | gradient                     (E1, E2/E3, E4)

Sizes are given in physical units (s) along time and as a fraction of the
display velocity span along velocity, then converted to pixels per image.

`LathamEstimator` is the OMM method of Latham et al. 2021 (IEEE TUFFC
68(3):465-481), implemented from the paper (section II-B..D, Fig. 5).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, transform

from .base import Context, EnvelopeEstimator, SIDES, side_view
from .likelihood import otsu_threshold

REF_DT = 0.003521  # s per column of the common Philips 75 mm/s display
REF_SPAN_ROWS = 385  # rows of the common Philips spectral display


def _px(ctx: Context, size_s: float, size_rel: float) -> tuple[int, int]:
    """(rows, cols) for a size given in seconds and in fraction of the velocity span."""
    H = ctx.spec.display_height
    rows = max(1, int(round(size_rel * H))) if size_rel > 0 else 1
    cols = max(1, int(round(size_s / ctx.spec.dt))) if size_s > 0 else 1
    return rows, cols


def _ref_px(ctx: Context, n: int) -> tuple[int, int]:
    """A size of n pixels at the reference display resolution -> (rows, cols) here."""
    H = ctx.spec.display_height
    return max(1, int(round(n * H / REF_SPAN_ROWS))), max(1, int(round(n * REF_DT / ctx.spec.dt)))


def _footprint(shape: str, rows: int, cols: int) -> np.ndarray:
    if shape == "ellipse":
        return morphology.ellipse(max(cols // 2, 0), max(rows // 2, 0)).astype(bool)[:max(rows, 1), :max(cols, 1)] \
            if rows > 1 and cols > 1 else np.ones((rows, cols), bool)
    return np.ones((rows, cols), bool)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


class SegmentationEstimator(EnvelopeEstimator):
    name = "segmentation"
    klass = "image"
    defaults = {
        "preprocess": "gaussian",
        "pre_size_s": 0.004,  # gaussian sigma / median / tophat size along time
        "pre_size_rel": 0.005,  # ... along velocity, fraction of display span
        "background_size_s": 0.1,  # low-frequency background (P3, B5)
        "background_size_rel": 0.1,
        "binarize": "otsu",
        "fixed_level": 60.0,  # B1: display level
        "percentile": 90.0,  # B3: image percentile
        "local_block_rel": 0.15,  # B4: block size, fraction of display span
        "local_offset": -5.0,
        "k_sigma": 3.0,  # B5: (I - B) > k * sigma_noise
        "column_fraction": 0.3,  # fraction of column max
        "relative_fraction": 0.3,  # fraction of image max
        "morphology": [["close", 3], ["open", 3], ["fill"], ["remove_small", 60]],
        "footprint": "rect",
        "components": "plausible",
        "min_width_s": 0.05,  # 'plausible': components spanning at least this time
        "near_rows_rel": 0.03,  # ... or within this distance of a kept component
        "extract": "outermost",
        "min_run_px": 1,
        "gradient_search_rel": 0.03,  # E4: search +- this around the E1 edge
    }

    # --- stages --------------------------------------------------------
    def preprocess(self, ctx: Context) -> np.ndarray:
        p = self.params
        img = ctx.spec.level.astype(float)
        rows, cols = _px(ctx, p["pre_size_s"], p["pre_size_rel"])
        kind = p["preprocess"]
        if kind == "none":
            return img
        if kind == "gaussian":
            return ndimage.gaussian_filter(img, sigma=(rows, cols), mode="nearest")
        if kind == "median":
            return ndimage.median_filter(img, size=(2 * rows + 1, 2 * cols + 1), mode="nearest")
        if kind == "background":
            br, bc = _px(ctx, p["background_size_s"], p["background_size_rel"])
            return img - ndimage.gaussian_filter(img, sigma=(br, bc), mode="nearest")
        if kind == "tophat":
            fp = np.ones((2 * rows * 4 + 1, 2 * cols * 4 + 1), bool)
            return ndimage.white_tophat(img, footprint=fp)
        raise ValueError(f"unknown preprocess '{kind}'")

    def binarize(self, img: np.ndarray, ctx: Context) -> np.ndarray:
        p = self.params
        valid = ctx.spec.valid & ~ctx.spec.sweep[None, :]
        vals = img[valid]
        kind = p["binarize"]
        if kind == "fixed":
            return img >= p["fixed_level"]
        if kind == "otsu":
            lo, hi = float(vals.min()), float(vals.max())
            t, _ = otsu_threshold(vals, value_range=(lo, hi if hi > lo else lo + 1))
            return img >= t
        if kind == "percentile":
            return img >= np.percentile(vals, p["percentile"])
        if kind == "local":
            H = img.shape[0]
            bs = max(3, int(round(p["local_block_rel"] * H)) | 1)
            return img >= filters.threshold_local(img, block_size=bs, offset=p["local_offset"])
        if kind == "background":
            br, bc = _px(ctx, p["background_size_s"], p["background_size_rel"])
            B = ndimage.median_filter(img, size=(max(3, br // 2) | 1, max(3, bc // 2) | 1), mode="nearest") \
                if False else ndimage.gaussian_filter(img, sigma=(br, bc), mode="nearest")
            r = img - B
            rv = r[valid]
            sig = 1.4826 * np.median(np.abs(rv - np.median(rv))) + 1e-6
            return r > p["k_sigma"] * sig
        if kind == "column_max":
            return img >= p["column_fraction"] * img.max(axis=0, keepdims=True)
        if kind == "relative_max":
            return img >= p["relative_fraction"] * np.percentile(vals, 99.9)
        raise ValueError(f"unknown binarize '{kind}'")

    def morph(self, M: np.ndarray, ctx: Context) -> np.ndarray:
        for step in self.params["morphology"]:
            op = step[0]
            n = int(step[1]) if len(step) > 1 else 0
            if op == "fill":
                M = ndimage.binary_fill_holes(M)
                continue
            if op == "remove_small":
                r, c = _ref_px(ctx, 1)
                M = morphology.remove_small_objects(M, max_size=max(1, int(n * r * c)))
                continue
            r, c = _ref_px(ctx, n)
            fp = _footprint(self.params["footprint"], r, c)
            f = {"open": ndimage.binary_opening, "close": ndimage.binary_closing,
                 "dilate": ndimage.binary_dilation, "erode": ndimage.binary_erosion}[op]
            M = f(M, structure=fp)
        return M

    def components(self, M: np.ndarray, ctx: Context) -> np.ndarray:
        kind = self.params["components"]
        if kind == "none":
            return M
        lab, n = ndimage.label(M, structure=np.ones((3, 3)))
        if n == 0:
            return M
        objs = ndimage.find_objects(lab)
        if kind == "largest":
            sizes = ndimage.sum(M, lab, index=np.arange(1, n + 1))
            return lab == (int(np.argmax(sizes)) + 1)
        if kind == "baseline":
            z = int(round(ctx.spec.zero_row))
            band = slice(max(0, z - 3), z + 4)
            keep = np.unique(lab[band])
            return np.isin(lab, keep[keep > 0])
        if kind == "plausible":
            wmin = self.params["min_width_s"] / ctx.spec.dt
            widths = np.array([sl[1].stop - sl[1].start for sl in objs])
            keep = widths >= wmin
            if keep.any():
                near = int(round(self.params["near_rows_rel"] * M.shape[0]))
                kept = np.isin(lab, np.flatnonzero(keep) + 1)
                grown = ndimage.binary_dilation(kept, structure=np.ones((2 * near + 1, 3), bool))
                touch = np.unique(lab[grown & M])
                keep[touch[touch > 0] - 1] = True
            return np.isin(lab, np.flatnonzero(keep) + 1)
        raise ValueError(f"unknown components '{kind}'")

    def extract_side(self, Mside: np.ndarray, Iside: np.ndarray) -> np.ndarray:
        kind = self.params["extract"]
        D, W = Mside.shape
        k = np.zeros(W)
        minrun = self.params["min_run_px"]
        for c in range(W):
            col = Mside[:, c]
            if not col.any():
                continue
            runs = [r for r in _runs(col) if r[1] - r[0] + 1 >= minrun]
            if not runs:
                continue
            if kind in ("outermost", "gradient"):
                k[c] = runs[-1][1]
            elif kind == "largest_run":
                k[c] = max(runs, key=lambda r: r[1] - r[0])[1]
            else:
                raise ValueError(f"unknown extract '{kind}'")
        if kind == "gradient":
            g = -np.gradient(ndimage.gaussian_filter1d(Iside, 1.0, axis=0), axis=0)
            w = max(1, int(round(self.params["gradient_search_rel"] * 2 * D)))
            for c in np.flatnonzero(k > 0):
                a, b = int(max(0, k[c] - w)), int(min(D - 1, k[c] + w))
                k[c] = a + int(np.argmax(g[a:b + 1, c]))
        return k

    def estimate(self, ctx: Context):
        spec = ctx.spec
        I = self.preprocess(ctx)
        M = self.binarize(I, ctx)
        M[:, spec.sweep] = False
        M = self.morph(M, ctx)
        M = self.components(M, ctx)
        out = {}
        for side in SIDES:
            sv = side_view(spec, side, image=M.astype(np.uint8))
            si = side_view(spec, side, image=I)
            k = self.extract_side(sv.level.astype(bool), si.level).astype(float)
            k[spec.sweep] = np.nan
            out[side] = k
        return self._result(out["pos"], out["neg"], mask=M, preprocessed=I)


class ThresholdEstimator(SegmentationEstimator):
    """Estimator 1: simple threshold, outermost pixel above it (no cleanup)."""

    name = "threshold"
    defaults = {**SegmentationEstimator.defaults, "preprocess": "none", "binarize": "fixed",
                "morphology": [], "components": "none", "extract": "outermost"}


class OtsuEstimator(SegmentationEstimator):
    """Estimator 2: global Otsu + morphology + outermost pixel."""

    name = "otsu"
    defaults = {**SegmentationEstimator.defaults, "preprocess": "gaussian", "binarize": "otsu",
                "morphology": [["close", 3], ["open", 3], ["fill"], ["remove_small", 60]],
                "components": "none", "extract": "outermost"}


class LathamEstimator(EnvelopeEstimator):
    """OMM: Otsu + hierarchical morphological filtering (Latham et al. 2021).

    Implemented as published, on the rendered image:
      - image resampled to the paper's pixel grid: 5 ms per column and 512
        rows over the full display span (the paper's 512-point FFT over
        +-8 kHz at 16 kHz sampling), so that the published pixel areas
        (300/500/100) and structuring elements keep their meaning;
      - 5x5 Gaussian, sigma = 1;
      - Otsu threshold chosen by the effectiveness metric EM over ten window
        lengths 1..100 ms. ADAPTATION: the paper recomputes the STFT with each
        window; from an image this is emulated by averaging the display along
        time over the equivalent duration. The chosen threshold is applied to
        the standard (unaveraged) image, as in the paper;
      - removal of bins below the wall filter (200 Hz ~ 6 rows of 31.25 Hz);
      - Fig. 5 hierarchy (levels 1-2 and cleaning), 8-connectivity;
      - MFE = white pixel next to the longest run of black pixels;
      - 10-point moving average (50 ms);
      - positive and negative sides processed separately.
    Not reproducible: audio-domain filtering and the spectrogram parameters.
    """

    name = "latham"
    klass = "image"
    defaults = {
        "col_dt_s": 0.005,
        "rows_full_span": 512,
        "gauss_size": 5,
        "gauss_sigma": 1.0,
        "em_windows_s": [0.001 + i * (0.099 / 9) for i in range(10)],
        "wall_rows": 6,
        "areas": [300, 500, 100],
        "moving_average": 10,
    }

    @staticmethod
    def _n_objects(B: np.ndarray) -> int:
        return ndimage.label(B, structure=np.ones((3, 3)))[1]

    @staticmethod
    def _remove(B: np.ndarray, area: int) -> np.ndarray:
        return morphology.remove_small_objects(B, max_size=area - 1, connectivity=2)

    def hierarchical(self, B: np.ndarray) -> np.ndarray:
        a1, a2, a3 = self.params["areas"]
        B = B.copy()
        B[0] = True  # [01]
        if self._n_objects(B) > 1:  # [02] level 1
            B = ndimage.binary_fill_holes(B)  # [03]
            B = self._remove(B, a1)  # [04]
            if self._n_objects(B) > 1:  # [05] level 2
                B = ndimage.binary_opening(B, structure=np.ones((3, 1), bool))  # [06]
                B[0] = True  # [07]
                B = ndimage.binary_fill_holes(B)  # [08]
                B = self._remove(B, a2)  # [09]
        B = ndimage.binary_fill_holes(B)  # [12]
        B = ndimage.binary_opening(B, structure=np.ones((5, 1), bool))  # [13]
        B = ndimage.binary_opening(B, structure=np.ones((1, 5), bool))  # [14]
        B[0] = True  # [15]
        B = self._remove(B, a3)  # [16]
        return B

    @staticmethod
    def mfe(B: np.ndarray) -> np.ndarray:
        """White pixel next to the longest run of black pixels, per column."""
        D, W = B.shape
        k = np.zeros(W)
        for c in range(W):
            col = B[:, c]
            gaps = _runs(~col)
            if not gaps:
                k[c] = D - 1
                continue
            a, b = max(gaps, key=lambda r: r[1] - r[0])
            k[c] = a - 1 if a > 0 else 0
        return k

    def estimate(self, ctx: Context):
        spec, p = ctx.spec, self.params
        H, W = spec.level.shape
        Wn = max(8, int(round(W * spec.dt / p["col_dt_s"])))
        Hn = int(round(p["rows_full_span"] * H / spec.display_height))  # 512 rows per displayed span
        img = spec.level.astype(float).copy()
        img[:, spec.sweep] = 0.0
        R = transform.resize(img, (Hn, Wn), order=1, mode="edge", anti_aliasing=True, preserve_range=True)
        # 5x5 Gaussian, sigma 1 (truncate so that the kernel is 5x5)
        G = ndimage.gaussian_filter(R, sigma=p["gauss_sigma"], truncate=(p["gauss_size"] // 2) / p["gauss_sigma"])
        # Otsu threshold with the best effectiveness metric over window lengths
        best_t, best_em = None, -1.0
        for w_s in p["em_windows_s"]:
            n = max(1, int(round(w_s / p["col_dt_s"])))
            A = ndimage.uniform_filter1d(G, n, axis=1, mode="nearest") if n > 1 else G
            t, em = otsu_threshold(A)
            if em > best_em:
                best_t, best_em = t, em
        Bfull = G >= best_t
        z_new = spec.zero_row * Hn / H
        row_scale = H / Hn
        out = {}
        up_is_positive = spec.dy < 0
        for side in SIDES:
            zi = int(round(z_new))
            if (side == "pos") == up_is_positive:
                Bs = Bfull[:zi + 1][::-1]
            else:
                Bs = Bfull[zi:]
            Bs = Bs[p["wall_rows"]:]
            if Bs.shape[0] < 3:
                out[side] = np.zeros(W)
                continue
            Bs = self.hierarchical(Bs)
            k = self.mfe(Bs)
            # rows below the wall-filter crop are not signal: an edge at the
            # first kept row means no flow on this side
            k = np.where(k > 0, k + p["wall_rows"], 0.0)
            n = p["moving_average"]
            if n > 1:
                k = ndimage.uniform_filter1d(k, n, mode="nearest")
            k = k * row_scale
            # back to the original columns
            xs_new = (np.arange(Wn) + 0.5) * W / Wn - 0.5
            kk = np.interp(np.arange(W), xs_new, k)
            kk[spec.sweep] = np.nan
            out[side] = kk
        return self._result(out["pos"], out["neg"], threshold=best_t, em=best_em, binary=Bfull)
