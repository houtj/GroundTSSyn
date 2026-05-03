from __future__ import annotations

from typing import Any

import numpy as np


def compute_globals(
    signal: list[float], primitive_peak2peaks: list[float]
) -> dict[str, Any]:
    """
    Compute the `global.*` pseudo-primitive features from the (z-normed) signal
    and the per-primitive peak2peak values measured on z-normed slices.
    """
    arr = np.asarray(signal, dtype=float)
    n = len(arr)
    p2ps = np.asarray(
        [p for p in primitive_peak2peaks if np.isfinite(p)], dtype=float
    )

    if n >= 2:
        spectrum = np.abs(np.fft.rfft(arr - arr.mean()))
        if len(spectrum) > 1:
            peak_bin = int(np.argmax(spectrum[1:]) + 1)
            frequency = float(peak_bin / n)
        else:
            frequency = 0.0
    else:
        frequency = 0.0

    if n > 0:
        counts, edges = np.histogram(arr, bins=50)
        peak_bin = int(np.argmax(counts))
        histogram_peak = float(0.5 * (edges[peak_bin] + edges[peak_bin + 1]))
    else:
        histogram_peak = 0.0

    return {
        "min": float(arr.min()) if n > 0 else 0.0,
        "max": float(arr.max()) if n > 0 else 0.0,
        "mean": float(arr.mean()) if n > 0 else 0.0,
        "std": float(arr.std()) if n > 0 else 0.0,
        "peak2peak": float(arr.max() - arr.min()) if n > 0 else 0.0,
        "duration": int(n),
        "peak2peak_mean": float(p2ps.mean()) if p2ps.size > 0 else 0.0,
        "peak2peak_q25": float(np.quantile(p2ps, 0.25)) if p2ps.size > 0 else 0.0,
        "peak2peak_median": float(np.median(p2ps)) if p2ps.size > 0 else 0.0,
        "peak2peak_q75": float(np.quantile(p2ps, 0.75)) if p2ps.size > 0 else 0.0,
        "frequency": frequency,
        "histogram_peak": histogram_peak,
    }
