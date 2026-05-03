from __future__ import annotations

from typing import Any

import numpy as np

from data_gen.schema import FeatureSpec


def _coerce_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def sample_uniform_continuous(low: float, high: float, rng=None) -> float:
    return float(_coerce_rng(rng).uniform(low, high))


def sample_uniform_integer(low: int, high: int, rng=None) -> int:
    """Inclusive [low, high]."""
    return int(_coerce_rng(rng).integers(low, high + 1))


def sample_uniform_categorical(options: list[Any], rng=None) -> Any:
    return options[int(_coerce_rng(rng).integers(0, len(options)))]


def sample_linear_integer(
    low: int, high: int, low_weight: float, high_weight: float, rng=None
) -> int:
    """Integer in [low, high] with weights linearly interpolated between endpoints."""
    n = high - low + 1
    xs = np.arange(n, dtype=float)
    if n == 1:
        return low
    weights = low_weight + (high_weight - low_weight) * xs / (n - 1)
    weights /= weights.sum()
    return int(low + _coerce_rng(rng).choice(n, p=weights))


def sample_skewed_normal(
    low: float, high: float, average_weight: float, extreme_weight: float, rng=None
) -> float:
    """
    Mixture of a peaked truncated normal at the midpoint and a uniform over [low, high].
    Mixture weight = average_weight / (average_weight + extreme_weight) for the normal part.
    """
    rng = _coerce_rng(rng)
    total = average_weight + extreme_weight
    if total <= 0:
        return sample_uniform_continuous(low, high, rng)
    p_normal = average_weight / total
    if rng.random() < p_normal:
        mid = 0.5 * (low + high)
        std = (high - low) / 6.0  # ~99.7% within range; we still clip
        # Rejection-sample to keep within [low, high]
        for _ in range(50):
            x = rng.normal(mid, std)
            if low <= x <= high:
                return float(x)
        return float(np.clip(mid, low, high))
    return sample_uniform_continuous(low, high, rng)


def sample_skewed_normal_integer(
    low: int, high: int, average_weight: float, extreme_weight: float, rng=None
) -> int:
    x = sample_skewed_normal(low, high, average_weight, extreme_weight, rng)
    return int(np.clip(round(x), low, high))


def sample_weighted_categorical(
    options: list[Any], weights: list[float], rng=None
) -> Any:
    rng = _coerce_rng(rng)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    return options[int(rng.choice(len(options), p=weights_arr))]


def sample_random_feature(spec: FeatureSpec, rng=None) -> Any:
    """
    Sample a single value for a `random` feature according to its schema.
    Supports the pdfs that random features actually declare today (uniform).
    Falls back to uniform for unsupported pdfs with a clear error.
    """
    if spec.generative != "random":
        raise ValueError(f"sample_random_feature called on {spec.generative} feature")
    if spec.pdf is None or spec.scope is None:
        raise ValueError("random feature must declare both pdf and scope")

    if spec.type == "categorical":
        if spec.pdf == "uniform":
            return sample_uniform_categorical(list(spec.scope), rng)
        raise NotImplementedError(f"categorical pdf={spec.pdf} not supported")

    low, high = spec.scope[0], spec.scope[1]
    if spec.type == "continuous":
        if spec.pdf == "uniform":
            return sample_uniform_continuous(float(low), float(high), rng)
        raise NotImplementedError(f"continuous pdf={spec.pdf} not supported")
    if spec.type == "integer":
        if spec.pdf == "uniform":
            return sample_uniform_integer(int(low), int(high), rng)
        raise NotImplementedError(f"integer pdf={spec.pdf} not supported")

    raise ValueError(f"unknown feature type {spec.type}")
