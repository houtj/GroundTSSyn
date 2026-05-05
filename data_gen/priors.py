from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from data_gen.config import Config
from data_gen.features import extract
from data_gen.realize import generate
from data_gen.schema import Schema
from data_gen.types import PrimitiveType

EPS = 1e-4


@dataclass
class EmpiricalPriors:
    """
    Empirical per-(primitive_type, feature) prior distributions, built by
    realizing K fresh primitives per type at startup. Used by steps 15-16
    distinctiveness scoring.
    """

    K: int = 200
    cache: dict[tuple[PrimitiveType, str], np.ndarray] = field(default_factory=dict)
    string_cache: dict[tuple[PrimitiveType, str], dict[str, int]] = field(
        default_factory=dict
    )

    @classmethod
    def build(
        cls,
        cfg: Config,
        schema: Schema,
        rng: np.random.Generator,
        *,
        K: int = 200,
        signal_std: float | None = None,
    ) -> "EmpiricalPriors":
        """
        Build empirical priors from K fresh primitive realizations.

        If `signal_std` is provided, scale the prior samples by 1/signal_std
        to align them with z-normed feature space. Integer features
        (durations) and ratio features (asymmetry, recovery, regularity) are
        z-invariant and are NOT scaled.
        """
        priors = cls(K=K)
        for ptype in PrimitiveType:
            samples_per_feature: dict[str, list[float]] = {}
            string_counts: dict[str, dict[str, int]] = {}
            for _ in range(K):
                definitive = _sample_definitive(cfg, ptype, rng)
                try:
                    prim = generate(ptype, definitive, schema, rng)
                except Exception:
                    continue
                feats = extract(prim.values, ptype)
                # merge non-recoverable random string fields (oscillation only)
                if ptype == PrimitiveType.OSCILLATION:
                    for key in ("waveform_shape", "regularity"):
                        if prim.random.get(key) is not None:
                            feats[key] = prim.random[key]
                for fname, fval in feats.items():
                    if isinstance(fval, (int, float)) and math.isfinite(float(fval)):
                        samples_per_feature.setdefault(fname, []).append(float(fval))
                    elif isinstance(fval, str):
                        counts = string_counts.setdefault(fname, {})
                        counts[fval] = counts.get(fval, 0) + 1
            for fname, vals in samples_per_feature.items():
                arr = np.asarray(vals, dtype=float)
                if signal_std is not None and signal_std > 0:
                    arr = _z_scale(arr, ptype, fname, schema, signal_std)
                priors.cache[(ptype, fname)] = arr
            for fname, counts in string_counts.items():
                priors.string_cache[(ptype, fname)] = counts
        return priors

    def values_for(self, ptype: PrimitiveType, feature: str) -> np.ndarray:
        return self.cache.get((ptype, feature), np.asarray([], dtype=float))

    def prob_op(
        self, ptype: PrimitiveType, feature: str, op: str, threshold: float
    ) -> float:
        vals = self.values_for(ptype, feature)
        if vals.size == 0:
            return 0.5
        sat = _apply_op(vals, op, threshold)
        return float(np.clip(sat.mean(), EPS, 1.0 - EPS))

    def prob_compare(
        self,
        t1: PrimitiveType,
        f1: str,
        op: str,
        t2: PrimitiveType,
        f2: str,
    ) -> float:
        x = self.values_for(t1, f1)
        y = self.values_for(t2, f2)
        if x.size == 0 or y.size == 0:
            return 0.5
        n = min(x.size, y.size)
        sat = _apply_op(x[:n], op, y[:n])
        return float(np.clip(sat.mean(), EPS, 1.0 - EPS))

    def prob_compare_to_constant(
        self, ptype: PrimitiveType, feature: str, op: str, value: float
    ) -> float:
        return self.prob_op(ptype, feature, op, value)

    def prob_self(self, ptype: PrimitiveType, feature: str, value: str) -> float:
        counts = self.string_cache.get((ptype, feature), {})
        total = sum(counts.values())
        if total == 0:
            return 0.5
        c = counts.get(value, 0)
        return float(np.clip(c / total, EPS, 1.0 - EPS))

    @staticmethod
    def info(p: float) -> float:
        return -math.log(max(p, EPS))


def _sample_definitive(
    cfg: Config, ptype: PrimitiveType, rng: np.random.Generator
) -> dict[str, float | int]:
    """Sample a representative definitive triple for prior MC realization."""
    p2p_low, p2p_high = cfg.peak2peak_range(in_event=True)
    if ptype in (PrimitiveType.SPIKE, PrimitiveType.DIP):
        duration = int(rng.integers(cfg.min_spike_length, cfg.max_spike_length + 1))
    else:
        d_low = cfg.min_primitive_length
        d_high = max(cfg.max_spike_length, 30)
        duration = int(rng.integers(d_low, d_high + 1))

    if ptype == PrimitiveType.PLATEAU:
        return {"start_value": 0.0, "duration": duration}
    return {
        "start_value": 0.0,
        "peak2peak": float(rng.uniform(p2p_low, p2p_high)),
        "duration": duration,
    }


_Z_INVARIANT_FEATURES = {
    "duration",
    "rise_duration",
    "fall_duration",
    "recovery",
    "regularity",
    "asymmetry",
    "n_cycles",
    "shape",
    "concavity",
    "direction",
    "waveform_shape",
    "frequency",
    "period",
}


def _z_scale(
    arr: np.ndarray,
    ptype: PrimitiveType,
    fname: str,
    schema: Schema,
    signal_std: float,
) -> np.ndarray:
    """Scale a prior sample array into z-normed feature space."""
    if fname in _Z_INVARIANT_FEATURES:
        return arr
    spec = schema.features(ptype).get(fname)
    if spec is not None and spec.type == "integer":
        return arr
    return arr / signal_std


def _apply_op(x: np.ndarray, op: str, y: float | np.ndarray) -> np.ndarray:
    if op == "<":
        return x < y
    if op == "<=":
        return x <= y
    if op == ">":
        return x > y
    if op == ">=":
        return x >= y
    if op == "==":
        return x == y
    if op == "!=":
        return x != y
    raise ValueError(f"unknown operator: {op!r}")
