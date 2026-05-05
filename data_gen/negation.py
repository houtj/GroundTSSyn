"""Negative-grounding generation for discriminative training.

A *negative grounding* is a structurally plausible variant of the positive
grounding (template type sequence + per-position constraints) that does NOT
match the realized signal anywhere. We produce candidates by applying a single
mutation to the positive grounding, then verify by sliding the mutated type
sequence over the post-jitter `placed` primitives and checking whether any
window matches both the type sequence and every constraint.

Mutations:
  * remove_middle    — drop one interior position (requires M >= 3)
  * switch_type      — replace one position's type with a different type
  * swap_positions   — exchange two positions whose types differ
  * invert_operator  — flip a single constraint's comparison operator

Verification uses strict semantics: a constraint that cannot be evaluated
(missing/non-finite feature) is treated as failing — which means the window
does NOT count as a positive match, which means the candidate IS counted as a
valid negative for that window. Equivalently we ask "does any window strictly
satisfy the candidate?" and only emit the negative when the answer is no.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pydantic import BaseModel

from data_gen.constraints import (
    ComparableConstraint,
    Constraint,
    FeatureRef,
    NumericalConstraint,
    Operator,
    SelfConstraint,
    evaluate_constraint,
)
from data_gen.post_match import _features_for_window
from data_gen.schema import Schema
from data_gen.types import ConcatResult, PrimitiveType


class NegativePrimitive(BaseModel):
    template_index: int
    type: PrimitiveType
    constraints: list[Constraint]


class NegativeGrounding(BaseModel):
    grounding: list[NegativePrimitive]
    mutation: str


_OP_INVERSE: dict[Operator, Operator] = {
    Operator.LT: Operator.GE,
    Operator.GE: Operator.LT,
    Operator.LE: Operator.GT,
    Operator.GT: Operator.LE,
    Operator.EQ: Operator.NE,
    Operator.NE: Operator.EQ,
}


# ---------------------------------------------------------------------------
# Constraint helpers (deep-copy + reindex / type-validate)
# ---------------------------------------------------------------------------


def _refs(c: Constraint) -> list[FeatureRef]:
    if isinstance(c, ComparableConstraint):
        return [c.lhs, c.rhs]
    return [c.target]


def _remap_constraint(
    c: Constraint, pos_map: dict[int, int]
) -> Constraint | None:
    """Return a copy of `c` with primitive indices remapped per `pos_map`.
    If any referenced int position is missing from the map (e.g. dropped),
    return None. The literal "global" rhs target is preserved untouched."""
    refs = _refs(c)
    new_refs: list[FeatureRef] = []
    for r in refs:
        if r.primitive == "global":
            new_refs.append(FeatureRef(primitive="global", feature=r.feature))
            continue
        assert isinstance(r.primitive, int)
        if r.primitive not in pos_map:
            return None
        new_refs.append(
            FeatureRef(primitive=pos_map[r.primitive], feature=r.feature)
        )

    data = c.model_dump()
    if isinstance(c, ComparableConstraint):
        data["lhs"] = new_refs[0].model_dump()
        data["rhs"] = new_refs[1].model_dump()
        return ComparableConstraint.model_validate(data)
    if isinstance(c, NumericalConstraint):
        data["target"] = new_refs[0].model_dump()
        return NumericalConstraint.model_validate(data)
    data["target"] = new_refs[0].model_dump()
    return SelfConstraint.model_validate(data)


def _constraint_features_valid(
    c: Constraint, types_by_pos: list[PrimitiveType], schema: Schema
) -> bool:
    """Check that every primitive-position feature reference in `c` is a
    known feature on the type now sitting at that position."""
    for r in _refs(c):
        if r.primitive == "global":
            continue
        assert isinstance(r.primitive, int)
        if r.primitive < 0 or r.primitive >= len(types_by_pos):
            return False
        ptype = types_by_pos[r.primitive]
        if r.feature not in schema.features(ptype):
            return False
    return True


def _clone_constraints(
    constraints_by_pos: list[list[Constraint]],
) -> list[list[Constraint]]:
    return [
        [type(c).model_validate(c.model_dump()) for c in lst]
        for lst in constraints_by_pos
    ]


def _build_negative(
    template_types: list[PrimitiveType],
    constraints_by_pos: list[list[Constraint]],
    mutation: str,
) -> NegativeGrounding:
    items = [
        NegativePrimitive(
            template_index=i, type=t, constraints=list(constraints_by_pos[i])
        )
        for i, t in enumerate(template_types)
    ]
    return NegativeGrounding(grounding=items, mutation=mutation)


# ---------------------------------------------------------------------------
# Mutators — each returns (types, constraints_by_pos, tag) or None
# ---------------------------------------------------------------------------


def _mutate_remove_middle(
    template_types: list[PrimitiveType],
    constraints_by_pos: list[list[Constraint]],
    schema: Schema,
    rng: np.random.Generator,
) -> tuple[list[PrimitiveType], list[list[Constraint]], str] | None:
    M = len(template_types)
    if M < 3:
        return None
    drop = int(rng.integers(1, M - 1))  # uniform in [1, M-2]
    new_types = [t for i, t in enumerate(template_types) if i != drop]
    pos_map = {old: (old if old < drop else old - 1) for old in range(M) if old != drop}

    new_constraints: list[list[Constraint]] = [[] for _ in range(M - 1)]
    for old_pos, lst in enumerate(constraints_by_pos):
        if old_pos == drop:
            continue
        new_pos = pos_map[old_pos]
        for c in lst:
            remapped = _remap_constraint(c, pos_map)
            if remapped is None:
                continue
            new_constraints[new_pos].append(remapped)
    return new_types, new_constraints, f"remove_middle@{drop}"


def _mutate_switch_type(
    template_types: list[PrimitiveType],
    constraints_by_pos: list[list[Constraint]],
    schema: Schema,
    rng: np.random.Generator,
) -> tuple[list[PrimitiveType], list[list[Constraint]], str] | None:
    M = len(template_types)
    if M == 0:
        return None
    pos = int(rng.integers(0, M))
    others = [t for t in PrimitiveType if t != template_types[pos]]
    if not others:
        return None
    new_t = others[int(rng.integers(0, len(others)))]
    new_types = list(template_types)
    new_types[pos] = new_t

    new_constraints = _clone_constraints(constraints_by_pos)
    new_constraints[pos] = [
        c for c in new_constraints[pos]
        if _constraint_features_valid(c, new_types, schema)
    ]
    for other_pos in range(M):
        if other_pos == pos:
            continue
        new_constraints[other_pos] = [
            c for c in new_constraints[other_pos]
            if _constraint_features_valid(c, new_types, schema)
        ]
    return new_types, new_constraints, f"switch_type@{pos}->{new_t.value}"


def _mutate_swap_positions(
    template_types: list[PrimitiveType],
    constraints_by_pos: list[list[Constraint]],
    schema: Schema,
    rng: np.random.Generator,
) -> tuple[list[PrimitiveType], list[list[Constraint]], str] | None:
    M = len(template_types)
    pairs = [
        (i, j)
        for i in range(M) for j in range(i + 1, M)
        if template_types[i] != template_types[j]
    ]
    if not pairs:
        return None
    i, j = pairs[int(rng.integers(0, len(pairs)))]

    new_types = list(template_types)
    new_types[i], new_types[j] = new_types[j], new_types[i]

    pos_map = {p: p for p in range(M)}
    pos_map[i], pos_map[j] = j, i

    new_constraints: list[list[Constraint]] = [[] for _ in range(M)]
    for old_pos, lst in enumerate(constraints_by_pos):
        new_pos = pos_map[old_pos]
        for c in lst:
            remapped = _remap_constraint(c, pos_map)
            if remapped is None:
                continue
            if not _constraint_features_valid(remapped, new_types, schema):
                continue
            new_constraints[new_pos].append(remapped)
    return new_types, new_constraints, f"swap@{i}<->{j}"


def _mutate_invert_operator(
    template_types: list[PrimitiveType],
    constraints_by_pos: list[list[Constraint]],
    schema: Schema,
    rng: np.random.Generator,
) -> tuple[list[PrimitiveType], list[list[Constraint]], str] | None:
    candidates: list[tuple[int, int]] = [
        (pos, ci)
        for pos, lst in enumerate(constraints_by_pos)
        for ci, _ in enumerate(lst)
    ]
    if not candidates:
        return None
    pos, ci = candidates[int(rng.integers(0, len(candidates)))]
    new_constraints = _clone_constraints(constraints_by_pos)
    c = new_constraints[pos][ci]
    if isinstance(c, SelfConstraint):
        return None  # self-constraints have no operator to invert
    new_op = _OP_INVERSE[c.op]
    data = c.model_dump()
    data["op"] = new_op.value
    new_c = type(c).model_validate(data)
    new_constraints[pos][ci] = new_c
    return list(template_types), new_constraints, f"invert_op@{pos}/{ci}"


_MUTATORS: list[
    Callable[
        [
            list[PrimitiveType],
            list[list[Constraint]],
            Schema,
            np.random.Generator,
        ],
        tuple[list[PrimitiveType], list[list[Constraint]], str] | None,
    ]
] = [
    _mutate_remove_middle,
    _mutate_switch_type,
    _mutate_swap_positions,
    _mutate_invert_operator,
]


# ---------------------------------------------------------------------------
# Verifier — slide candidate over the full placed sequence
# ---------------------------------------------------------------------------


def _matches_anywhere(
    candidate_types: list[PrimitiveType],
    candidate_constraints: list[list[Constraint]],
    raw_jittered: ConcatResult,
    z_signal: list[float],
    globals_: dict[str, Any],
) -> bool:
    M = len(candidate_types)
    if M == 0:
        return True  # vacuous match — treat as not-a-valid-negative
    placed = raw_jittered.placed
    n = len(placed)
    if n < M:
        return False
    for start in range(n - M + 1):
        window = placed[start : start + M]
        if [pp.primitive.type for pp in window] != candidate_types:
            continue
        feats_by_pos = _features_for_window(window, z_signal)
        if _strict_constraints_pass(feats_by_pos, candidate_constraints, globals_):
            return True
    return False


def _strict_constraints_pass(
    feats_by_pos: list[dict[str, Any]],
    constraints_by_pos: list[list[Constraint]],
    globals_: dict[str, Any],
) -> bool:
    for constraints in constraints_by_pos:
        for c in constraints:
            sat = evaluate_constraint(c, feats_by_pos, globals_)
            if sat is not True:
                return False
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_negative_groundings(
    template_types: list[PrimitiveType],
    constraints_by_pos: list[list[Constraint]],
    raw_jittered: ConcatResult,
    z_signal: list[float],
    globals_: dict[str, Any],
    schema: Schema,
    rng: np.random.Generator,
    *,
    max_attempts: int = 10,
    target_n: int = 3,
) -> list[NegativeGrounding]:
    """Try up to `max_attempts` mutations, return up to `target_n` verified
    negatives. Each attempt picks a single mutator uniformly at random; if
    the mutator is inapplicable (e.g. invert-operator with no constraints),
    we still consume the attempt budget."""
    out: list[NegativeGrounding] = []
    seen: set[str] = set()  # avoid emitting duplicate (types,constraints) tuples
    for _ in range(max_attempts):
        if len(out) >= target_n:
            break
        op = _MUTATORS[int(rng.integers(0, len(_MUTATORS)))]
        result = op(template_types, constraints_by_pos, schema, rng)
        if result is None:
            continue
        new_types, new_constraints, tag = result
        key = _signature(new_types, new_constraints)
        if key in seen:
            continue
        if _matches_anywhere(
            new_types, new_constraints, raw_jittered, z_signal, globals_
        ):
            continue
        seen.add(key)
        out.append(_build_negative(new_types, new_constraints, tag))
    return out


def _signature(
    types: list[PrimitiveType], constraints_by_pos: list[list[Constraint]]
) -> str:
    type_str = ",".join(t.value for t in types)
    cons_str = "|".join(
        ";".join(repr(c.model_dump()) for c in lst) for lst in constraints_by_pos
    )
    return f"{type_str}::{cons_str}"
