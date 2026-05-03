from __future__ import annotations

import numpy as np


def z_normalize(signal: list[float]) -> list[float]:
    normed, _, _ = z_normalize_with_stats(signal)
    return normed


def z_normalize_with_stats(
    signal: list[float],
) -> tuple[list[float], float, float]:
    """Return (z-normed signal, mean, std). std is floored to 1e-9."""
    arr = np.asarray(signal, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    safe_std = max(std, 1e-9)
    return ((arr - mean) / safe_std).tolist(), mean, std
