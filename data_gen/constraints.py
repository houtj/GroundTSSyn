from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel

from data_gen.config import Config
from data_gen.priors import EPS, EmpiricalPriors
from data_gen.recompute import NormedFeatures
from data_gen.schema import FeatureSpec, Schema
from data_gen.types import Event, PrimitiveType


class Operator(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="


class FeatureRef(BaseModel):
    primitive: int | Literal["global"]
    feature: str


class NumericalConstraint(BaseModel):
    kind: Literal["numerical"] = "numerical"
    target: FeatureRef
    op: Operator
    threshold: float
    realized_value: float
    p_satisfied: float
    info: float
    salience: float
    weight: float


class ComparableConstraint(BaseModel):
    kind: Literal["comparable"] = "comparable"
    lhs: FeatureRef
    op: Operator
    rhs: FeatureRef
    realized_lhs: float
    realized_rhs: float
    p_satisfied: float
    info: float
    salience: float
    weight: float


Constraint = Union[NumericalConstraint, ComparableConstraint]


class GroundedPrimitive(BaseModel):
    template_index: int
    type: PrimitiveType
    requested_count: int
    actual_count: int
    constraints: list[Constraint]


class ConstraintsResult(BaseModel):
    grounded: list[GroundedPrimitive]


P_LOW = 0.01
P_HIGH = 0.99
CONTINUOUS_OPS = (Operator.LT, Operator.LE, Operator.GT, Operator.GE)
INTEGER_OPS = (
    Operator.LT,
    Operator.LE,
    Operator.GT,
    Operator.GE,
    Operator.EQ,
    Operator.NE,
)


def sample_constraints(
    cfg: Config,
    schema: Schema,
    event_template: Event,
    normed: NormedFeatures,
    priors: EmpiricalPriors,
    rng: np.random.Generator,
) -> ConstraintsResult:
    """Steps 13-16: sample constraints for each in-event template position."""
    template_types = [p.type for p in event_template.primitives]
    first_occ = normed.in_event[0] if normed.in_event else []

    grounded: list[GroundedPrimitive] = []
    for pos, ptype in enumerate(template_types):
        feats = first_occ[pos] if pos < len(first_occ) else {}
        candidates_num = _enumerate_numerical(pos, ptype, feats, schema, priors)
        candidates_cmp = _enumerate_comparable(
            pos, ptype, feats, template_types, first_occ, normed.globals, schema, priors
        )

        n_requested = _sample_count(cfg, rng)
        picked = _sample_pool(
            candidates_num, candidates_cmp, n_requested, cfg, rng
        )
        grounded.append(
            GroundedPrimitive(
                template_index=pos,
                type=ptype,
                requested_count=n_requested,
                actual_count=len(picked),
                constraints=picked,
            )
        )
    return ConstraintsResult(grounded=grounded)


def _sample_count(cfg: Config, rng: np.random.Generator) -> int:
    low, high = int(cfg.nb_constraints[0]), int(cfg.nb_constraints[1])
    options = list(range(low, high + 1))
    weights = np.asarray(cfg.pdf_nb_constraints, dtype=float)
    if len(weights) != len(options):
        raise ValueError(
            f"pdf_nb_constraints length {len(weights)} != nb_constraints range {len(options)}"
        )
    weights = weights / weights.sum()
    return int(options[int(rng.choice(len(options), p=weights))])


def _sample_pool(
    candidates_num: list[Constraint],
    candidates_cmp: list[Constraint],
    n: int,
    cfg: Config,
    rng: np.random.Generator,
) -> list[Constraint]:
    picked: list[Constraint] = []
    pools: dict[str, list[Constraint]] = {
        "numerical": list(candidates_num),
        "comparable": list(candidates_cmp),
    }
    weights = {
        "numerical": float(cfg.numerical_constraint_weight),
        "comparable": float(cfg.comparable_constraint_weight),
    }
    while len(picked) < n:
        active = [k for k in ("numerical", "comparable") if pools[k]]
        if not active:
            break
        ws = np.asarray([weights[k] for k in active], dtype=float)
        ws = ws / ws.sum()
        kind = active[int(rng.choice(len(active), p=ws))]
        pool = pools[kind]
        cand_weights = np.asarray([c.weight for c in pool], dtype=float)
        if cand_weights.sum() <= 0:
            chosen_idx = int(rng.integers(0, len(pool)))
        else:
            cand_weights = cand_weights / cand_weights.sum()
            chosen_idx = int(rng.choice(len(pool), p=cand_weights))
        picked.append(pool.pop(chosen_idx))
    return picked


def _enumerate_numerical(
    pos: int,
    ptype: PrimitiveType,
    feats: dict[str, Any],
    schema: Schema,
    priors: EmpiricalPriors,
) -> list[Constraint]:
    out: list[Constraint] = []
    for fname, spec in schema.features(ptype).items():
        if "numerical" not in spec.constraint:
            continue
        v = feats.get(fname)
        if not _is_finite_number(v):
            continue
        ops = INTEGER_OPS if spec.type == "integer" else CONTINUOUS_OPS
        salience = schema.salience(ptype, fname)
        for threshold in _candidate_thresholds(float(v), spec.type):
            for op in ops:
                if not _op_satisfied_by(op, float(v), threshold):
                    continue
                p = priors.prob_op(ptype, fname, op.value, threshold)
                if p <= P_LOW or p >= P_HIGH:
                    continue
                info = -math.log(max(p, EPS))
                weight = salience * info
                out.append(
                    NumericalConstraint(
                        target=FeatureRef(primitive=pos, feature=fname),
                        op=op,
                        threshold=threshold,
                        realized_value=float(v),
                        p_satisfied=p,
                        info=info,
                        salience=salience,
                        weight=weight,
                    )
                )
    return out


def _enumerate_comparable(
    pos: int,
    ptype: PrimitiveType,
    feats: dict[str, Any],
    template_types: list[PrimitiveType],
    first_occ: list[dict[str, Any]],
    globals_: dict[str, Any],
    schema: Schema,
    priors: EmpiricalPriors,
) -> list[Constraint]:
    out: list[Constraint] = []
    for fname, spec in schema.features(ptype).items():
        if "comparable" not in spec.constraint:
            continue
        if not spec.comparable_features:
            continue
        x = feats.get(fname)
        if not _is_finite_number(x):
            continue
        x_val = float(x)
        ops = _comparable_ops_for(spec.type)
        salience = schema.salience(ptype, fname)
        for entry in spec.comparable_features:
            try:
                target_type_str, target_feature = entry.split(".", 1)
            except ValueError:
                continue
            for target_pos, y_val, target_spec in _resolve_targets(
                pos,
                ptype,
                fname,
                target_type_str,
                target_feature,
                template_types,
                first_occ,
                globals_,
                schema,
            ):
                for op in _ops_for_relation(x_val, y_val, ops):
                    if target_pos == "global":
                        target_ptype = PrimitiveType(target_type_str) if target_type_str != "global" else None
                        # global fallback: treat as constant
                        p = priors.prob_compare_to_constant(
                            ptype, fname, op.value, y_val
                        )
                    else:
                        target_ptype = template_types[target_pos]
                        p = priors.prob_compare(
                            ptype, fname, op.value, target_ptype, target_feature
                        )
                    if p <= P_LOW or p >= P_HIGH:
                        continue
                    info = -math.log(max(p, EPS))
                    weight = salience * info
                    out.append(
                        ComparableConstraint(
                            lhs=FeatureRef(primitive=pos, feature=fname),
                            op=op,
                            rhs=FeatureRef(primitive=target_pos, feature=target_feature),
                            realized_lhs=x_val,
                            realized_rhs=y_val,
                            p_satisfied=p,
                            info=info,
                            salience=salience,
                            weight=weight,
                        )
                    )
    return out


def _resolve_targets(
    self_pos: int,
    self_type: PrimitiveType,
    self_feature: str,
    target_type_str: str,
    target_feature: str,
    template_types: list[PrimitiveType],
    first_occ: list[dict[str, Any]],
    globals_: dict[str, Any],
    schema: Schema,
):
    """Yield (target_pos, y_val, target_feature_spec) for each valid resolution."""
    if target_type_str == "global":
        if target_feature not in globals_:
            return
        y = globals_[target_feature]
        if not _is_finite_number(y):
            return
        yield "global", float(y), None
        return

    try:
        ttype = PrimitiveType(target_type_str)
    except ValueError:
        return
    target_spec = schema.features(ttype).get(target_feature)
    if target_spec is None:
        return
    for tpos, t in enumerate(template_types):
        if t != ttype:
            continue
        if tpos == self_pos and target_feature == self_feature:
            continue
        if tpos >= len(first_occ):
            continue
        y = first_occ[tpos].get(target_feature)
        if not _is_finite_number(y):
            continue
        yield tpos, float(y), target_spec


def _comparable_ops_for(ftype: str) -> tuple[Operator, ...]:
    if ftype == "integer":
        return INTEGER_OPS
    return CONTINUOUS_OPS


def _ops_for_relation(
    x: float, y: float, ops: tuple[Operator, ...]
) -> list[Operator]:
    """Pick the operators that are currently true for the (x, y) relation."""
    out: list[Operator] = []
    if x > y and Operator.GT in ops:
        out.append(Operator.GT)
    if x >= y and Operator.GE in ops:
        out.append(Operator.GE)
    if x < y and Operator.LT in ops:
        out.append(Operator.LT)
    if x <= y and Operator.LE in ops:
        out.append(Operator.LE)
    if x == y and Operator.EQ in ops:
        out.append(Operator.EQ)
    if x != y and Operator.NE in ops:
        out.append(Operator.NE)
    return out


def _candidate_thresholds(v: float, ftype: str) -> list[float]:
    """Generate round-ish thresholds near `v` for numerical constraint enumeration."""
    cands: set[float] = set()
    if ftype == "integer":
        cands.update({float(round(v) + d) for d in (-3, -2, -1, 0, 1, 2, 3)})
    else:
        cands.add(round(v))
        cands.add(round(v * 2) / 2)
        cands.add(_round_to_n_sig(v, 1))
        if abs(v) >= 1:
            cands.add(round(v / 5) * 5)
        if abs(v) >= 5:
            cands.add(round(v / 10) * 10)
        cands.add(round(v, 1))
    finite = [float(c) for c in cands if math.isfinite(c)]
    return sorted(set(finite))


def _round_to_n_sig(x: float, n: int) -> float:
    if x == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(x)))
    factor = 10 ** (n - 1 - magnitude)
    return round(x * factor) / factor


def _op_satisfied_by(op: Operator, v: float, t: float) -> bool:
    if op == Operator.LT:
        return v < t
    if op == Operator.LE:
        return v <= t
    if op == Operator.GT:
        return v > t
    if op == Operator.GE:
        return v >= t
    if op == Operator.EQ:
        return v == t
    if op == Operator.NE:
        return v != t
    return False


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def check_op(a: float, op: str, b: float) -> bool:
    """Apply a comparison operator to two scalar values."""
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise ValueError(f"unknown operator: {op!r}")


def evaluate_constraint(
    c: Constraint,
    feats_by_pos: list[dict[str, Any]],
    globals_: dict[str, Any],
) -> bool | None:
    """
    Evaluate a single constraint against per-template-position features for
    a single occurrence. Returns True/False, or None if the relevant feature
    is missing or non-finite (treat as "cannot evaluate").
    """
    if isinstance(c, NumericalConstraint):
        pos = c.target.primitive
        if not isinstance(pos, int) or pos < 0 or pos >= len(feats_by_pos):
            return None
        v = feats_by_pos[pos].get(c.target.feature)
        if not _is_finite_number(v):
            return None
        return check_op(float(v), c.op.value, c.threshold)

    lhs_pos = c.lhs.primitive
    if not isinstance(lhs_pos, int) or lhs_pos < 0 or lhs_pos >= len(feats_by_pos):
        return None
    lhs = feats_by_pos[lhs_pos].get(c.lhs.feature)
    if not _is_finite_number(lhs):
        return None
    if c.rhs.primitive == "global":
        rhs = globals_.get(c.rhs.feature)
    else:
        rhs_pos = c.rhs.primitive
        if not isinstance(rhs_pos, int) or rhs_pos < 0 or rhs_pos >= len(feats_by_pos):
            return None
        rhs = feats_by_pos[rhs_pos].get(c.rhs.feature)
    if not _is_finite_number(rhs):
        return None
    return check_op(float(lhs), c.op.value, float(rhs))
