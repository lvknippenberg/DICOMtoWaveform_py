"""Envelope estimators and their registry.

Benchmark hierarchy of Plan.txt section 23:

    A threshold    fixed threshold, outermost pixel               (image)
    B otsu         Otsu + morphology                              (image)
    C percentile   integrated-power percentile                    (spectrum)
    D gm           geometric method                               (spectrum)
    E mgm          modified geometric method                      (spectrum)
    F snsi         signal-noise slope intersection                (spectrum)
    G asnsi        adaptive SNSI/GM, Kathpalia et al. 2016        (spectrum)
    H qia          quadratic iteration algorithm, Li et al. 2019  (spectrum)
    I latham       Otsu + hierarchical morphology (OMM), Latham et al. 2021 (image)
    J dp           likelihood image + dynamic-programming curve   (image, primary)

plus `segmentation`, the fully configurable image pipeline used for ablations.
"""

from __future__ import annotations

from typing import Any

from .base import Context, EnvelopeEstimate, EnvelopeEstimator, SideImage, side_view
from .dp import DPEstimator
from .segmentation import LathamEstimator, OtsuEstimator, SegmentationEstimator, ThresholdEstimator
from .spectral import (AdaptiveSNSIEstimator, GMEstimator, MGMEstimator, PercentileEstimator,
                       QIAEstimator, SNSIEstimator)

ESTIMATORS: dict[str, type[EnvelopeEstimator]] = {
    cls.name: cls
    for cls in (ThresholdEstimator, OtsuEstimator, PercentileEstimator, GMEstimator, MGMEstimator,
                SNSIEstimator, AdaptiveSNSIEstimator, QIAEstimator, LathamEstimator, DPEstimator,
                SegmentationEstimator)
}

BENCHMARK_ORDER = ["threshold", "otsu", "percentile", "gm", "mgm", "snsi", "asnsi", "qia", "latham", "dp"]


def make_estimator(name: str, **params: Any) -> EnvelopeEstimator:
    try:
        cls = ESTIMATORS[name]
    except KeyError:
        raise KeyError(f"Unknown estimator '{name}'. Available: {sorted(ESTIMATORS)}") from None
    return cls(**params)


__all__ = ["Context", "EnvelopeEstimate", "EnvelopeEstimator", "SideImage", "side_view",
           "ESTIMATORS", "BENCHMARK_ORDER", "make_estimator"]
