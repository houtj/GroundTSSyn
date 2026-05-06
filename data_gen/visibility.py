from __future__ import annotations

import math
from typing import Any

from data_gen.types import PrimitiveType

# Visibility = sigmoid((m - TAU) / K), m built from amplitude SNR + duration
# saturation + neighbor masking. Constants derived from SNR-experience anchors:
#   peak2peak/σ ≈ 2–3   → low  (v ≈ 0.14)
#   peak2peak/σ ≈ 10    → reliable (v ≈ 0.77)
#   peak2peak/σ ≥ 35    → saturated (v ≈ 1.0)
TAU = 7.0
K = 2.5
D_REF = 3.0
GAMMA = 0.1


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _duration_factor(d: float) -> float:
    return 1.0 - math.exp(-d / D_REF)


def compute_visibility(
    feats: dict[str, Any],
    type_: PrimitiveType,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    sigma: float,
) -> float:
    duration = float(feats.get("duration", 0))

    if type_ == PrimitiveType.PLATEAU:
        # Plateau is duration-only: a flat segment is visible iff it's long
        # enough to register as a distinct rest. Linear in samples so the
        # shared sigmoid (TAU=7) reads as "needs ~7+ samples to be visible".
        m = duration
    else:
        if sigma <= 0.0:
            return 1.0
        p2p = float(feats.get("peak2peak", 0.0))
        amp_term = p2p / sigma
        df = _duration_factor(duration)
        d_left = float(left.get("duration", 0.0)) if left else 0.0
        d_right = float(right.get("duration", 0.0)) if right else 0.0
        denom = duration + GAMMA * max(d_left, d_right)
        mask = duration / denom if denom > 0 else 1.0
        m = amp_term * df * mask

    return _sigmoid((m - TAU) / K)
