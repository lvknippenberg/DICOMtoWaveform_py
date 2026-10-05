"""Common interface for envelope estimators.

Every estimator returns, per column, the outer boundary of the spectrum on
the positive side and on the negative side of the zero-velocity line, as a
distance from the baseline in rows ("edge index" k: k = 0 is the zero
velocity row, k increases with |v|). Sides are handled through `SideImage`,
which flips the image so that every side looks the same (baseline at k = 0,
increasing speed with k) whatever the display orientation (PhysicalDeltaY
sign) and baseline position.

A side without flow has its edge at k = 0 (velocity 0), so v_max and v_min
are defined in every data column. Sweep columns are NaN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..config import LikelihoodConfig, DPConfig
from ..render import Spectrogram
from .likelihood import NoiseStats, noise_stats

SIDES = ("pos", "neg")


@dataclass
class SideImage:
    """One half of the spectrogram, re-indexed outward from the baseline."""

    level: np.ndarray  # (D, W) display level, overlays filled
    valid: np.ndarray  # (D, W)
    rows: np.ndarray  # (D,) original row index of each k
    speed_per_row: float  # |dy| (cm/s per row)
    sign: int  # +1 for the positive side, -1 for the negative side
    sweep: np.ndarray  # (W,) bool

    @property
    def depth(self) -> int:
        return self.level.shape[0]

    def velocity(self, k: np.ndarray | float) -> np.ndarray:
        return self.sign * np.asarray(k, float) * self.speed_per_row


def side_view(spec: Spectrogram, side: str, image: np.ndarray | None = None) -> SideImage:
    """Rows from the zero row outward on the given side.

    `image` (H, W) can replace spec.level (e.g. a preprocessed image).
    """
    img = spec.level if image is None else image
    H = img.shape[0]
    z = int(round(spec.zero_row))
    # Row index increases downward; velocity = (r - z) * dy.
    # Positive velocities lie at rows r with (r - z) * dy > 0.
    up_is_positive = spec.dy < 0
    if (side == "pos") == up_is_positive:
        rows = np.arange(z, -1, -1)
    else:
        rows = np.arange(z, H)
    rows = rows[(rows >= 0) & (rows < H)]
    return SideImage(
        level=img[rows], valid=spec.valid[rows], rows=rows,
        speed_per_row=abs(spec.dy), sign=1 if side == "pos" else -1, sweep=spec.sweep,
    )


@dataclass
class Context:
    """Shared, lazily computed per-image quantities."""

    spec: Spectrogram
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    _noise: NoiseStats | None = None
    aliasing: dict[str, Any] = field(default_factory=dict)  # set by the pipeline

    @property
    def noise(self) -> NoiseStats:
        if self._noise is None:
            self._noise = noise_stats(self.spec, self.likelihood)
        return self._noise


@dataclass
class EnvelopeEstimate:
    """Outer edges in rows from the baseline (NaN in the sweep)."""

    k_pos: np.ndarray
    k_neg: np.ndarray
    conf_pos: np.ndarray | None = None
    conf_neg: np.ndarray | None = None
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def velocities(self, spec: Spectrogram) -> tuple[np.ndarray, np.ndarray]:
        """(v_max, v_min) in cm/s."""
        s = abs(spec.dy)
        return self.k_pos * s, -self.k_neg * s


class EnvelopeEstimator(ABC):
    """Base class. Subclasses set `name`, `klass` ('spectrum' | 'image') and
    `defaults`, and implement `estimate`.

    There is no fit(): nothing is learned. Parameters are passed at
    construction (from YAML via `make_estimator`) and recorded in the result.
    """

    name: ClassVar[str] = "base"
    klass: ClassVar[str] = "image"
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise KeyError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        self.params = {**self.defaults, **params}

    @abstractmethod
    def estimate(self, ctx: Context) -> EnvelopeEstimate: ...

    def __call__(self, spec: Spectrogram, likelihood: LikelihoodConfig | None = None,
                 dp: DPConfig | None = None) -> EnvelopeEstimate:
        ctx = Context(spec, likelihood or LikelihoodConfig(), dp or DPConfig())
        return self.estimate(ctx)

    def _result(self, k_pos, k_neg, conf_pos=None, conf_neg=None, **diag) -> EnvelopeEstimate:
        return EnvelopeEstimate(k_pos=k_pos, k_neg=k_neg, conf_pos=conf_pos, conf_neg=conf_neg,
                                method=self.name, params=dict(self.params), diagnostics=diag)


class ColumnwiseEstimator(EnvelopeEstimator):
    """Estimators that treat every column (spectrum) independently.

    Subclasses implement `edge_column(spectrum, ctx) -> float` on a one-sided
    spectrum (index 0 = baseline) or override `edge_side` for vectorised code.
    """

    klass = "spectrum"

    def spectra(self, side: SideImage, ctx: Context) -> np.ndarray:
        """One-sided spectra (D, W) used by the method. Default: display level."""
        return side.level.astype(float)

    def edge_column(self, s: np.ndarray, ctx: Context) -> float:  # pragma: no cover
        raise NotImplementedError

    def edge_side(self, side: SideImage, ctx: Context) -> tuple[np.ndarray, np.ndarray | None]:
        S = self.spectra(side, ctx)
        W = S.shape[1]
        out = np.full(W, np.nan)
        for c in range(W):
            if side.sweep[c]:
                continue
            out[c] = self.edge_column(S[:, c], ctx)
        return out, None

    def estimate(self, ctx: Context) -> EnvelopeEstimate:
        res = {}
        for side in SIDES:
            sv = side_view(ctx.spec, side)
            k, conf = self.edge_side(sv, ctx)
            k = np.asarray(k, float)
            k[ctx.spec.sweep] = np.nan
            res[side] = (k, conf)
        return self._result(res["pos"][0], res["neg"][0], res["pos"][1], res["neg"][1])
