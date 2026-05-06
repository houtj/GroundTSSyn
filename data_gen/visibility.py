from __future__ import annotations

import math
from typing import Any

from data_gen.types import PrimitiveType

# Generic (increase/decrease/oscillation): visibility depends on amplitude SNR,
# duration saturation, and neighbor masking. Anchors:
#   peak2peak/σ ≈ 2–3   → low  (v ≈ 0.14)
#   peak2peak/σ ≈ 10    → reliable (v ≈ 0.77)
#   peak2peak/σ ≥ 35    → saturated (v ≈ 1.0)
TAU = 7.0
K = 2.5
D_REF = 3.0
GAMMA = 0.1

# Spike/dip: point-like, naturally very visible — purely amplitude-based.
# Anchors derived from the rule "invisible below ~1.3σ, visible above ~3σ":
#   τ = 3.0   (v = 0.5 at peak2peak = 3σ)
#   k = 1/√3  (v ≈ 0.05 at 1.3σ, v ≈ 0.97 at 5σ)
# No duration or neighbor-mask factor: spikes stand out by amplitude alone.
TAU_SPIKE = 3.0
K_SPIKE = 1.0 / math.sqrt(3)  # ≈ 0.577

# Change-point visibility: same SNR-anchor sigmoid as the generic primitive
# visibility — invisible boundary when slope+level match exactly, half-visible
# when combined SNR ≈ 7, saturated above ~15.
TAU_CP = 7.0
K_CP = 2.5


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

    elif type_ in (PrimitiveType.SPIKE, PrimitiveType.DIP):
        # Spikes and dips are point-like — visibility is purely amplitude-based.
        # No duration saturation (spikes are always short); no neighbor mask
        # (the sharp transient is distinguishable from surrounding context).
        if sigma <= 0.0:
            return 1.0
        m = float(feats.get("peak2peak", 0.0)) / sigma
        return _sigmoid((m - TAU_SPIKE) / K_SPIKE)

    else:
        # increase, decrease, oscillation: amplitude + duration + neighbor mask.
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


def _boundary_slopes(
    feats: dict[str, Any], type_: PrimitiveType
) -> tuple[float, float]:
    """Returns (left_edge_slope, right_edge_slope), signed, in samples⁻¹."""
    if type_ == PrimitiveType.PLATEAU:
        return 0.0, 0.0
    if type_ == PrimitiveType.INCREASE:
        s = float(feats.get("slope", 0.0))
        return s, s
    if type_ == PrimitiveType.DECREASE:
        s = -float(feats.get("slope", 0.0))
        return s, s
    if type_ == PrimitiveType.SPIKE:
        return float(feats.get("rise_slope", 0.0)), -float(feats.get("fall_slope", 0.0))
    if type_ == PrimitiveType.DIP:
        return -float(feats.get("fall_slope", 0.0)), float(feats.get("rise_slope", 0.0))
    if type_ == PrimitiveType.OSCILLATION:
        s = float(feats.get("center_slope", 0.0))
        return s, s
    return 0.0, 0.0


def _boundary_trend_values(
    feats: dict[str, Any], type_: PrimitiveType
) -> tuple[float, float]:
    """Returns (left_edge_trend_value, right_edge_trend_value).

    For non-oscillation types the trend line passes through start/end exactly,
    matching the actual signal samples (so neighbors agree → Δlevel = 0). For
    oscillation, the least-squares center line generally diverges from the
    instantaneous endpoint samples, so Δlevel can be non-zero.
    """
    if type_ == PrimitiveType.OSCILLATION:
        c0 = float(feats.get("center_intercept", feats.get("start_value", 0.0)))
        s = float(feats.get("center_slope", 0.0))
        d = float(feats.get("duration", 0))
        return c0, c0 + s * d
    return float(feats.get("start_value", 0.0)), float(feats.get("end_value", 0.0))


def compute_cp_visibility(
    left_feats: dict[str, Any],
    left_type: PrimitiveType,
    right_feats: dict[str, Any],
    right_type: PrimitiveType,
    sigma: float,
) -> float:
    """Visibility of the boundary between two adjacent primitives ∈ [0,1].

    Combines slope mismatch (SNR over the shorter neighbor's window) with
    trend-level mismatch (direct amplitude SNR), then sigmoid-normalizes.
    """
    _, s_out = _boundary_slopes(left_feats, left_type)
    s_in, _ = _boundary_slopes(right_feats, right_type)
    _, c_out = _boundary_trend_values(left_feats, left_type)
    c_in, _ = _boundary_trend_values(right_feats, right_type)

    delta_s = s_in - s_out
    delta_c = c_in - c_out

    if sigma <= 0.0:
        return 1.0 if (abs(delta_s) > 0 or abs(delta_c) > 0) else 0.0

    L = max(min(float(left_feats.get("duration", 1)),
                float(right_feats.get("duration", 1))), 1.0)
    delta_s_norm = abs(delta_s) * math.sqrt(L) / sigma
    delta_c_norm = abs(delta_c) / sigma
    m = math.sqrt(delta_s_norm**2 + delta_c_norm**2)
    return _sigmoid((m - TAU_CP) / K_CP)
