from __future__ import annotations

from typing import Any, Callable

import numpy as np

from data_gen.types import PrimitiveType


def _fwhm(values: np.ndarray, baseline: float, peak: float) -> float:
    """Full-width at half-maximum, with linear interpolation between samples."""
    half = (baseline + peak) / 2.0
    above = (values >= half) if peak >= baseline else (values <= half)
    if not np.any(above):
        return 0.0
    idxs = np.where(above)[0]
    return float(idxs[-1] - idxs[0] + 1)


def _classify_asymmetry(a: int, b: int) -> str:
    """Return 'symmetric' when |a - b| <= 1, else 'asymmetric'."""
    return "symmetric" if abs(a - b) <= 1 else "asymmetric"


def _classify_shape(values: np.ndarray, eps: float = 0.1) -> str:
    """Classify a monotonic ramp as linear / convex / concave by midpoint deviation."""
    n = len(values)
    if n < 3:
        return "linear"
    range_ = abs(values[-1] - values[0])
    if range_ < 1e-9:
        return "linear"
    linear_mid = 0.5 * (values[0] + values[-1])
    actual_mid = values[n // 2]
    deviation = (actual_mid - linear_mid) / range_
    if deviation < -eps:
        return "convex"
    if deviation > eps:
        return "concave"
    return "linear"


_RECOVERY_FULL_THRESHOLD = 0.95


def _classify_recovery(recovery: float) -> str:
    return "full" if recovery >= _RECOVERY_FULL_THRESHOLD else "partial"


def extract_spike(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    start_value = float(v[0])
    peak_idx = int(np.argmax(v))
    peak_value = float(v[peak_idx])
    peak2peak = peak_value - start_value
    duration = len(v) - 1
    rise_duration = peak_idx
    fall_duration = duration - rise_duration
    end_value = float(v[-1])
    recovery = (
        1.0 - (end_value - start_value) / peak2peak if peak2peak > 0 else 1.0
    )
    recovery = float(np.clip(recovery, 0.0, 1.0))
    width = _fwhm(v, start_value, peak_value)
    sharpness = peak2peak / width if width > 0 else float("nan")
    asymmetry = _classify_asymmetry(rise_duration, fall_duration)
    rise_slope = peak2peak / rise_duration if rise_duration > 0 else float("inf")
    fall_slope = peak2peak / fall_duration if fall_duration > 0 else float("inf")
    return {
        "start_value": start_value,
        "peak_value": peak_value,
        "peak2peak": peak2peak,
        "duration": duration,
        "rise_duration": rise_duration,
        "recovery": recovery,
        "recovery_type": _classify_recovery(recovery),
        "fall_duration": fall_duration,
        "end_value": end_value,
        "sharpness": sharpness,
        "asymmetry": asymmetry,
        "rise_slope": rise_slope,
        "fall_slope": fall_slope,
    }


def extract_dip(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    start_value = float(v[0])
    trough_idx = int(np.argmin(v))
    trough_value = float(v[trough_idx])
    peak2peak = start_value - trough_value
    duration = len(v) - 1
    fall_duration = trough_idx
    rise_duration = duration - fall_duration
    end_value = float(v[-1])
    recovery = (
        1.0 - (start_value - end_value) / peak2peak if peak2peak > 0 else 1.0
    )
    recovery = float(np.clip(recovery, 0.0, 1.0))
    width = _fwhm(v, start_value, trough_value)
    sharpness = peak2peak / width if width > 0 else float("nan")
    asymmetry = _classify_asymmetry(fall_duration, rise_duration)
    rise_slope = peak2peak / rise_duration if rise_duration > 0 else float("inf")
    fall_slope = peak2peak / fall_duration if fall_duration > 0 else float("inf")
    return {
        "start_value": start_value,
        "trough_value": trough_value,
        "peak2peak": peak2peak,
        "duration": duration,
        "fall_duration": fall_duration,
        "recovery": recovery,
        "recovery_type": _classify_recovery(recovery),
        "rise_duration": rise_duration,
        "end_value": end_value,
        "sharpness": sharpness,
        "asymmetry": asymmetry,
        "rise_slope": rise_slope,
        "fall_slope": fall_slope,
    }


def extract_increase(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    start_value = float(v[0])
    end_value = float(v[-1])
    peak2peak = end_value - start_value
    duration = len(v) - 1
    shape = _classify_shape(v)
    slope = peak2peak / duration if duration > 0 else float("inf")
    return {
        "start_value": start_value,
        "peak2peak": peak2peak,
        "duration": duration,
        "shape": shape,
        "end_value": end_value,
        "slope": slope,
    }


def extract_decrease(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    start_value = float(v[0])
    end_value = float(v[-1])
    peak2peak = start_value - end_value
    duration = len(v) - 1
    shape = _classify_shape(v)
    slope = peak2peak / duration if duration > 0 else float("inf")
    return {
        "start_value": start_value,
        "peak2peak": peak2peak,
        "duration": duration,
        "shape": shape,
        "end_value": end_value,
        "slope": slope,
    }


def extract_oscillation(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    start_value = float(v[0])
    end_value = float(v[-1])
    peak2peak = float(v.max() - v.min())
    duration = len(v) - 1
    mean_value = float(v.mean())
    centered = v - mean_value
    sign_changes = int(np.sum(np.diff(np.sign(centered)) != 0))
    frequency = sign_changes / (2.0 * duration) if duration > 0 else 0.0
    period = 1.0 / frequency if frequency > 0 else float("inf")
    n_cycles = frequency * duration
    diff = end_value - start_value
    center_trend = (
        "upward" if diff > 0.05 * peak2peak
        else "downward" if diff < -0.05 * peak2peak
        else "flat"
    )
    # envelope_behavior / envelope_asymmetry / regularity / waveform_shape:
    # not reliably recoverable from signal alone. Caller should merge with the
    # Primitive's `random` dict to populate them.
    return {
        "start_value": start_value,
        "peak2peak": peak2peak,
        "duration": duration,
        "frequency": frequency,
        "envelope_behavior": None,
        "envelope_asymmetry": None,
        "regularity": None,
        "waveform_shape": None,
        "period": period,
        "n_cycles": n_cycles,
        "center_trend": center_trend,
        "mean_value": mean_value,
        "end_value": end_value,
    }


def extract_plateau(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    start_value = float(v[0])
    end_value = float(v[-1])
    duration = len(v) - 1
    return {
        "start_value": start_value,
        "duration": duration,
        "end_value": end_value,
    }


EXTRACTORS: dict[PrimitiveType, Callable[[list[float]], dict[str, Any]]] = {
    PrimitiveType.SPIKE: extract_spike,
    PrimitiveType.DIP: extract_dip,
    PrimitiveType.INCREASE: extract_increase,
    PrimitiveType.DECREASE: extract_decrease,
    PrimitiveType.OSCILLATION: extract_oscillation,
    PrimitiveType.PLATEAU: extract_plateau,
}


def extract(values: list[float], type: PrimitiveType) -> dict[str, Any]:
    """
    Compute every feature listed in feature_schema.yaml for `type`, derived
    from `values` alone. Definitive features are recovered (start_value=values[0],
    duration=len-1, peak2peak=range), random features are recovered where the
    signal supports it (recovery, rise_duration), and derived features are
    computed by formula. For oscillation, waveform_shape / envelope_behavior /
    envelope_asymmetry / regularity are returned as None — they're not reliably
    recoverable from signal alone; merge with the source Primitive's `random`
    dict when needed.
    """
    return EXTRACTORS[type](values)
