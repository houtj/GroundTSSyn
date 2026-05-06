from __future__ import annotations

import numpy as np

from data_gen.config import Config


def sample_noise_level(cfg: Config, rng: np.random.Generator) -> float:
    """Sample σ from cfg.noise_level. Returns 0.0 when add_noise is False."""
    if not cfg.add_noise:
        return 0.0
    lo, hi = float(cfg.noise_level[0]), float(cfg.noise_level[1])
    if hi <= lo:
        return lo
    if cfg.pdf_noise_level == "uniform":
        return float(rng.uniform(lo, hi))
    return float(rng.uniform(lo, hi))


def add_white_noise(
    signal: list[float], sigma: float, rng: np.random.Generator
) -> list[float]:
    """Add per-sample N(0, σ²) noise. σ=0 returns a copy of the input."""
    if sigma <= 0.0:
        return list(signal)
    arr = np.asarray(signal, dtype=float)
    noise = rng.normal(0.0, sigma, size=arr.shape)
    return (arr + noise).tolist()
