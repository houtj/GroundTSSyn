from __future__ import annotations

from typing import Callable

import numpy as np

from data_gen.sampling import sample_random_feature
from data_gen.schema import Schema
from data_gen.types import Primitive, PrimitiveType


def realize_spike(
    start_value: float,
    peak2peak: float,
    duration: int,
    *,
    rise_duration: int,
    recovery: float,
) -> list[float]:
    fall_duration = duration - rise_duration
    if rise_duration < 1 or fall_duration < 1:
        raise ValueError(
            f"spike: need rise_duration >= 1 and fall_duration >= 1; got "
            f"rise_duration={rise_duration}, fall_duration={fall_duration}, "
            f"duration={duration}"
        )
    peak = start_value + peak2peak
    end = start_value + (1.0 - recovery) * peak2peak
    rise = np.linspace(start_value, peak, rise_duration + 1)
    fall = np.linspace(peak, end, fall_duration + 1)[1:]
    return np.concatenate([rise, fall]).tolist()


def realize_dip(
    start_value: float,
    peak2peak: float,
    duration: int,
    *,
    fall_duration: int,
    recovery: float,
) -> list[float]:
    rise_duration = duration - fall_duration
    if fall_duration < 1 or rise_duration < 1:
        raise ValueError(
            f"dip: need fall_duration >= 1 and rise_duration >= 1; got "
            f"fall_duration={fall_duration}, rise_duration={rise_duration}, "
            f"duration={duration}"
        )
    trough = start_value - peak2peak
    end = start_value - (1.0 - recovery) * peak2peak
    fall = np.linspace(start_value, trough, fall_duration + 1)
    rise = np.linspace(trough, end, rise_duration + 1)[1:]
    return np.concatenate([fall, rise]).tolist()


def _convex_f(t: np.ndarray) -> np.ndarray:
    """Mathematically convex f: f''>0 (slow start, fast end). f(0)=0, f(1)=1."""
    k = 2.0
    return (np.exp(k * t) - 1.0) / (np.exp(k) - 1.0)


def _concave_f(t: np.ndarray) -> np.ndarray:
    """Mathematically concave f: f''<0 (fast start, slow end). f(0)=0, f(1)=1."""
    k = 9.0
    return np.log1p(k * t) / np.log1p(k)


def realize_increase(
    start_value: float,
    peak2peak: float,
    duration: int,
    *,
    shape: str,
) -> list[float]:
    """y = start + p2p * f. Signal y''>0 ⇒ convex (exp-like, accelerating)."""
    t = np.linspace(0.0, 1.0, duration + 1)
    if shape == "linear":
        f = t
    elif shape == "convex":
        f = _convex_f(t)
    elif shape == "concave":
        f = _concave_f(t)
    else:
        raise ValueError(f"unknown increase shape: {shape!r}")
    return (start_value + peak2peak * f).tolist()


def realize_decrease(
    start_value: float,
    peak2peak: float,
    duration: int,
    *,
    shape: str,
) -> list[float]:
    """y = start - p2p * f. Signal y''>0 ⇒ convex requires f'' < 0 (log-like, decelerating drop)."""
    t = np.linspace(0.0, 1.0, duration + 1)
    if shape == "linear":
        f = t
    elif shape == "convex":
        f = _concave_f(t)  # signal convex ⇒ f concave for decrease
    elif shape == "concave":
        f = _convex_f(t)
    else:
        raise ValueError(f"unknown decrease shape: {shape!r}")
    return (start_value - peak2peak * f).tolist()


def realize_oscillation(
    start_value: float,
    peak2peak: float,
    duration: int,
    *,
    frequency: float,
    envelope_behavior: str,
    envelope_asymmetry: str,
    regularity: str,  # noqa: ARG001  (reserved; unused in v1)
    waveform_shape: str,
) -> list[float]:
    n = duration + 1
    t = np.arange(n)
    arg = 2.0 * np.pi * frequency * t

    if waveform_shape == "sinusoidal":
        wave = np.sin(arg)
    elif waveform_shape == "square":
        wave = np.sign(np.sin(arg))
        wave[wave == 0] = 1.0
    elif waveform_shape == "triangle":
        wave = (2.0 / np.pi) * np.arcsin(np.sin(arg))
    elif waveform_shape == "sawtooth":
        wave = 2.0 * ((frequency * t) % 1.0) - 1.0
    else:
        raise ValueError(f"unknown waveform_shape: {waveform_shape!r}")

    norm_t = t / max(n - 1, 1)
    if envelope_behavior == "sustained":
        env = np.ones_like(norm_t)
    elif envelope_behavior == "decaying":
        env = 1.0 - 0.6 * norm_t
    elif envelope_behavior == "resonating":
        env = 0.4 + 0.6 * norm_t
    else:
        raise ValueError(f"unknown envelope_behavior: {envelope_behavior!r}")

    if envelope_asymmetry == "symmetric":
        modulated = wave * env
    elif envelope_asymmetry == "lower_anchored":
        upper = np.maximum(wave, 0.0) * env
        lower = np.minimum(wave, 0.0)
        modulated = upper + lower
    elif envelope_asymmetry == "upper_anchored":
        upper = np.maximum(wave, 0.0)
        lower = np.minimum(wave, 0.0) * env
        modulated = upper + lower
    else:
        raise ValueError(f"unknown envelope_asymmetry: {envelope_asymmetry!r}")

    raw = (peak2peak / 2.0) * modulated
    return (start_value + raw - raw[0]).tolist()


def realize_plateau(start_value: float, duration: int) -> list[float]:
    return [float(start_value)] * (duration + 1)


REALIZERS: dict[PrimitiveType, Callable[..., list[float]]] = {
    PrimitiveType.SPIKE: realize_spike,
    PrimitiveType.DIP: realize_dip,
    PrimitiveType.INCREASE: realize_increase,
    PrimitiveType.DECREASE: realize_decrease,
    PrimitiveType.OSCILLATION: realize_oscillation,
    PrimitiveType.PLATEAU: realize_plateau,
}


def generate(
    type: PrimitiveType,
    definitive: dict[str, float | int],
    schema: Schema,
    rng: np.random.Generator | None = None,
) -> Primitive:
    """
    Generate a single primitive realization.
    1. Sample each random feature from its schema scope/pdf.
    2. Apply runtime constraints on random features (e.g. spike's
       rise_duration must satisfy 1 <= rise_duration <= duration - 1).
    3. Call the type-specific realizer.
    4. Return Primitive(type, values, definitive, random).

    Length invariant: len(primitive.values) == duration + 1.
    """
    rng = rng if rng is not None else np.random.default_rng()

    expected_definitive = set(schema.definitive_features(type).keys())
    missing = expected_definitive - definitive.keys()
    if missing:
        raise ValueError(
            f"generate({type.value}): missing definitive features {sorted(missing)}"
        )

    random_features: dict[str, float | int | str] = {}
    for fname, fspec in schema.random_features(type).items():
        random_features[fname] = sample_random_feature(fspec, rng)

    duration = int(definitive["duration"])
    if type == PrimitiveType.SPIKE:
        random_features["rise_duration"] = int(
            np.clip(random_features["rise_duration"], 1, duration - 1)
        )
    elif type == PrimitiveType.DIP:
        random_features["fall_duration"] = int(
            np.clip(random_features["fall_duration"], 1, duration - 1)
        )

    realizer = REALIZERS[type]
    values = realizer(**definitive, **random_features)

    return Primitive(
        type=type,
        values=values,
        definitive=dict(definitive),
        random=random_features,
    )
