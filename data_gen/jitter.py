from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from data_gen.config import Config
from data_gen.constraints import (
    ComparableConstraint,
    Constraint,
    ConstraintsResult,
    NumericalConstraint,
    SelfConstraint,
    evaluate_constraint,
)
from data_gen.features import extract
from data_gen.layout import Layout
from data_gen.orchestrate import validate_slope
from data_gen.realize import REALIZERS
from data_gen.recompute import _merge_random
from data_gen.schema import Schema
from data_gen.types import Event, Primitive, PrimitiveType


class ConstraintRetryExhausted(RuntimeError):
    """Raised when jitter cannot satisfy a constraint within max_attempts."""


@dataclass
class JitterFallback:
    """Records a position where slope retries were exhausted and we fell back.

    Retained for API compatibility; constraint-driven jitter raises rather
    than falls back, so this list stays empty in current usage.
    """

    event_index: int
    template_index: int
    attempts: int


@dataclass
class JitterReport:
    """Diagnostics for the step-17 jitter pass."""

    fallbacks: list[JitterFallback] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sum-preserving duration jitter
# ---------------------------------------------------------------------------

def _vary_with_constant_sum(
    values: np.ndarray, pct: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-element delta in [-pct·v, +pct·v], re-centred so sum(deltas) == 0."""
    deltas = rng.uniform(-pct, pct, size=values.shape) * values
    deltas -= deltas.mean()
    return values + deltas


def _redistribute_diff(
    rounded: np.ndarray, diff: int, floors: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Apply +/- 1 nudges across positions to absorb `diff` while respecting floors."""
    if diff == 0:
        return rounded
    out = rounded.copy()
    n = len(out)
    indices = np.arange(n)
    safety = 4 * abs(diff) + 4 * n
    while diff != 0 and safety > 0:
        rng.shuffle(indices)
        for i in indices:
            if diff == 0:
                break
            if diff > 0:
                out[i] += 1
                diff -= 1
            else:
                if out[i] - 1 >= floors[i]:
                    out[i] -= 1
                    diff += 1
        safety -= 1
    return out


def jitter_durations_constant_sum(
    template_durations: Sequence[int],
    nb_occurrences: int,
    pct: float,
    rng: np.random.Generator,
    *,
    min_per_position: Sequence[int],
) -> list[list[int]]:
    """
    Returns a (nb_occurrences x nb_in_event) matrix of jittered integer
    durations whose total equals nb_occurrences * sum(template_durations).
    """
    nb_in_event = len(template_durations)
    if nb_occurrences == 0:
        return []
    if pct <= 0 or nb_in_event == 0:
        return [list(template_durations) for _ in range(nb_occurrences)]

    flat = np.tile(np.asarray(template_durations, dtype=float), nb_occurrences)
    jittered = _vary_with_constant_sum(flat, pct, rng)
    rounded = np.round(jittered).astype(int)

    floors = np.tile(np.asarray(min_per_position, dtype=int), nb_occurrences)
    rounded = np.maximum(rounded, floors)

    diff = int(flat.sum()) - int(rounded.sum())
    rounded = _redistribute_diff(rounded, diff, floors, rng)

    return [
        rounded[k * nb_in_event:(k + 1) * nb_in_event].tolist()
        for k in range(nb_occurrences)
    ]


# ---------------------------------------------------------------------------
# Per-feature jitter primitives (uniform-percentage)
# ---------------------------------------------------------------------------

def _jitter_continuous(
    value: float,
    pct: float,
    scope: Sequence[float] | None,
    rng: np.random.Generator,
) -> float:
    if pct <= 0:
        return float(value)
    new = float(value) + rng.uniform(-pct, pct) * float(value)
    if scope is not None:
        new = float(np.clip(new, float(scope[0]), float(scope[1])))
    return float(new)


def _jitter_integer(
    value: int,
    pct: float,
    scope: Sequence[int] | None,
    rng: np.random.Generator,
) -> int:
    if pct <= 0:
        return int(value)
    new = int(round(float(value) + rng.uniform(-pct, pct) * float(value)))
    if scope is not None:
        new = int(np.clip(new, int(scope[0]), int(scope[1])))
    return int(new)


# ---------------------------------------------------------------------------
# Definitive / random feature jitter
# ---------------------------------------------------------------------------

def jitter_definitive(
    template_definitive: dict[str, Any],
    duration_override: int,
    ptype: PrimitiveType,
    schema: Schema,
    cfg: Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Returns a new definitive dict where:
      - duration ← duration_override (already jittered with sum preserved)
      - start_value ← unchanged (level_dependent)
      - any other numerical level-independent feature: uniform-pct jittered
      - level-dependent / categorical features: passed through unchanged
    """
    out = dict(template_definitive)
    out["duration"] = int(duration_override)
    feats = schema.features(ptype)
    for fname, fval in template_definitive.items():
        if fname in ("duration", "start_value"):
            continue
        spec = feats.get(fname)
        if spec is None or spec.level_dependent or spec.type == "categorical":
            continue
        if fname == "peak2peak":
            scope = cfg.peak2peak_range(in_event=True)
        else:
            scope = spec.scope
        if spec.type == "continuous":
            out[fname] = _jitter_continuous(
                float(fval), cfg.jitter_randomness, scope, rng
            )
        else:  # integer
            out[fname] = _jitter_integer(
                int(fval), cfg.jitter_randomness, scope, rng
            )
    return out


def jitter_random(
    random_dict: dict[str, Any],
    jittered_duration: int,
    ptype: PrimitiveType,
    schema: Schema,
    cfg: Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Per-feature jitter on a primitive's `random` dict using the uniform-pct
    rule. Categorical and level-dependent features are passed through.
    Returns a NEW dict; does not mutate input.
    """
    out: dict[str, Any] = dict(random_dict)
    feats = schema.features(ptype)
    for fname, fval in random_dict.items():
        spec = feats.get(fname)
        if spec is None or spec.level_dependent or spec.type == "categorical":
            continue
        if spec.type == "continuous":
            out[fname] = _jitter_continuous(
                float(fval), cfg.jitter_randomness, spec.scope, rng
            )
        else:  # integer
            out[fname] = _jitter_integer(
                int(fval), cfg.jitter_randomness, spec.scope, rng
            )
    if ptype == PrimitiveType.SPIKE and "rise_duration" in out:
        out["rise_duration"] = int(
            np.clip(int(out["rise_duration"]), 1, max(jittered_duration - 1, 1))
        )
    elif ptype == PrimitiveType.DIP and "fall_duration" in out:
        out["fall_duration"] = int(
            np.clip(int(out["fall_duration"]), 1, max(jittered_duration - 1, 1))
        )
    return out


def realize_with_features(
    template_prim: Primitive,
    jittered_definitive: dict[str, Any],
    jittered_random: dict[str, Any],
) -> Primitive:
    """
    Re-realize a primitive with fresh definitive + random dicts. Bypasses
    `generate()` so we can inject specific values.
    """
    realizer = REALIZERS[template_prim.type]
    new_values = realizer(**jittered_definitive, **jittered_random)
    return Primitive(
        type=template_prim.type,
        values=new_values,
        definitive=dict(jittered_definitive),
        random=dict(jittered_random),
    )


# ---------------------------------------------------------------------------
# Constraint classification
# ---------------------------------------------------------------------------

ConstraintStage = Literal["dur", "intra", "cross", "defer"]


def _constraint_pos_refs(c: Constraint) -> set[Any]:
    if isinstance(c, (NumericalConstraint, SelfConstraint)):
        return {c.target.primitive}
    return {c.lhs.primitive, c.rhs.primitive}


def _constraint_feature_names(c: Constraint) -> set[str]:
    if isinstance(c, (NumericalConstraint, SelfConstraint)):
        return {c.target.feature}
    return {c.lhs.feature, c.rhs.feature}


def _feat_level_dependent(
    pos: Any,
    feature: str,
    template_types: list[PrimitiveType],
    schema: Schema,
) -> bool | None:
    """True/False if known; None if pos == "global" or feature missing from schema."""
    if pos == "global":
        return None
    if not isinstance(pos, int) or pos < 0 or pos >= len(template_types):
        return None
    spec = schema.features(template_types[pos]).get(feature)
    if spec is None:
        return None
    return spec.level_dependent


def _is_enforceable(
    c: Constraint, schema: Schema, template_types: list[PrimitiveType]
) -> bool:
    """All referenced features must be level_dependent=false; no global refs.
    Self constraints are categorical and cannot be enforced by jitter."""
    if isinstance(c, SelfConstraint):
        return False
    if isinstance(c, NumericalConstraint):
        refs = [(c.target.primitive, c.target.feature)]
    else:
        refs = [(c.lhs.primitive, c.lhs.feature), (c.rhs.primitive, c.rhs.feature)]
    for pos, fname in refs:
        if pos == "global":
            return False
        ld = _feat_level_dependent(pos, fname, template_types, schema)
        if ld is None or ld is True:
            return False
    return True


def _classify(
    c: Constraint,
    pos: int,
    schema: Schema,
    template_types: list[PrimitiveType],
) -> ConstraintStage:
    if not _is_enforceable(c, schema, template_types):
        return "defer"
    if all(name == "duration" for name in _constraint_feature_names(c)):
        return "dur"
    if _constraint_pos_refs(c) == {pos}:
        return "intra"
    return "cross"


@dataclass
class _ClassifiedConstraints:
    duration: list[Constraint]
    intra_by_pos: list[list[Constraint]]
    cross_by_pos: list[list[Constraint]]


def _classify_constraints(
    constraints_result: ConstraintsResult,
    schema: Schema,
    template_types: list[PrimitiveType],
) -> _ClassifiedConstraints:
    nb_in_event = len(template_types)
    duration: list[Constraint] = []
    intra: list[list[Constraint]] = [[] for _ in range(nb_in_event)]
    cross: list[list[Constraint]] = [[] for _ in range(nb_in_event)]
    for grounded in constraints_result.grounded:
        pos = grounded.template_index
        for c in grounded.constraints:
            stage = _classify(c, pos, schema, template_types)
            if stage == "dur":
                duration.append(c)
            elif stage == "intra":
                intra[pos].append(c)
            elif stage == "cross":
                cross[pos].append(c)
            # "defer" → drop (handled by report-only verify_constraints)
    return _ClassifiedConstraints(
        duration=duration, intra_by_pos=intra, cross_by_pos=cross
    )


# ---------------------------------------------------------------------------
# Per-primitive feature extraction (for constraint evaluation at jitter time)
# ---------------------------------------------------------------------------

def _prim_feats(prim: Primitive) -> dict[str, Any]:
    feats = extract(prim.values, prim.type)
    _merge_random(feats, prim.type, prim.random)
    return feats


def _check_intra(constraints: list[Constraint], prim_feats: dict[str, Any]) -> bool:
    """Evaluate intra-primitive constraints (lhs and rhs both on this primitive)
    by wrapping the single feats dict so `evaluate_constraint`'s positional
    indexing works regardless of the original primitive index."""
    if not constraints:
        return True
    for c in constraints:
        # Build a stub feats_by_pos large enough to index any referenced pos.
        max_pos = -1
        for ref in _constraint_pos_refs(c):
            if isinstance(ref, int) and ref > max_pos:
                max_pos = ref
        feats_by_pos: list[dict[str, Any]] = [dict() for _ in range(max_pos + 1)]
        for ref in _constraint_pos_refs(c):
            if isinstance(ref, int):
                feats_by_pos[ref] = prim_feats
        result = evaluate_constraint(c, feats_by_pos, globals_={})
        if result is not True:
            return False
    return True


def _check_cross(
    cross_by_pos: list[list[Constraint]],
    feats_by_pos: list[dict[str, Any]],
) -> bool:
    for pos_constraints in cross_by_pos:
        for c in pos_constraints:
            result = evaluate_constraint(c, feats_by_pos, globals_={})
            if result is not True:
                return False
    return True


# ---------------------------------------------------------------------------
# Stage 1 — duration matrix retry
# ---------------------------------------------------------------------------

def _check_duration_matrix(
    durations_by_occ: list[list[int]],
    duration_constraints: list[Constraint],
) -> bool:
    if not duration_constraints:
        return True
    for occ_durations in durations_by_occ:
        feats_by_pos = [{"duration": int(d)} for d in occ_durations]
        for c in duration_constraints:
            result = evaluate_constraint(c, feats_by_pos, globals_={})
            if result is not True:
                return False
    return True


def _sample_durations_with_retry(
    template_durations: list[int],
    nb_occurrences: int,
    pct: float,
    rng: np.random.Generator,
    min_per_position: list[int],
    duration_constraints: list[Constraint],
    *,
    max_attempts: int,
) -> list[list[int]]:
    for _ in range(max_attempts):
        durations_by_occ = jitter_durations_constant_sum(
            template_durations,
            nb_occurrences,
            pct,
            rng,
            min_per_position=min_per_position,
        )
        if _check_duration_matrix(durations_by_occ, duration_constraints):
            return durations_by_occ
        if pct <= 0 or nb_occurrences == 0:
            # No randomness to retry — the template itself fails the check.
            break
    raise ConstraintRetryExhausted(
        f"duration matrix: {len(duration_constraints)} duration constraint(s) "
        f"could not be satisfied within {max_attempts} attempts"
    )


# ---------------------------------------------------------------------------
# Stage 2 — per-occurrence build with intra-primitive constraint check
# ---------------------------------------------------------------------------

def jitter_event_occurrence(
    template: Event,
    schema: Schema,
    cfg: Config,
    rng: np.random.Generator,
    *,
    boundary_prev: Primitive | None,
    boundary_next: Primitive | None,
    jittered_durations: Sequence[int],
    intra_constraints_by_pos: list[list[Constraint]] | None = None,
    max_attempts: int = 200,
    event_index: int = 0,
) -> Event:
    """
    Build one event occurrence by jittering each template primitive's
    definitive + random features (definitive duration comes from the
    sum-preserving stage-1 matrix), validating the slope rule, and enforcing
    any intra-primitive level-independent constraints. Raises
    `ConstraintRetryExhausted` if no valid candidate is found within
    `max_attempts` for any position.
    """
    if intra_constraints_by_pos is None:
        intra_constraints_by_pos = [[] for _ in template.primitives]

    primitives: list[Primitive] = []
    template_prims = template.primitives
    for pos, tprim in enumerate(template_prims):
        new_dur = int(jittered_durations[pos])
        intra = intra_constraints_by_pos[pos]
        prev = primitives[-1] if pos > 0 else boundary_prev
        next_ = (
            template_prims[pos + 1]
            if pos + 1 < len(template_prims)
            else boundary_next
        )

        chosen: Primitive | None = None
        for _ in range(max_attempts):
            new_def = jitter_definitive(
                tprim.definitive, new_dur, tprim.type, schema, cfg, rng
            )
            if tprim.type == PrimitiveType.PLATEAU or not tprim.random:
                cand = realize_with_features(tprim, new_def, dict(tprim.random))
                if not validate_slope(prev, cand, next_):
                    continue
                if intra and not _check_intra(intra, _prim_feats(cand)):
                    continue
                chosen = cand
                break

            new_rand = jitter_random(
                tprim.random, new_dur, tprim.type, schema, cfg, rng
            )
            try:
                cand = realize_with_features(tprim, new_def, new_rand)
            except Exception:
                continue
            if not validate_slope(prev, cand, next_):
                continue
            if intra and not _check_intra(intra, _prim_feats(cand)):
                continue
            chosen = cand
            break

        if chosen is None:
            raise ConstraintRetryExhausted(
                f"event={event_index}, template_index={pos}: no candidate "
                f"satisfied slope + intra-primitive constraints within "
                f"{max_attempts} attempts"
            )
        primitives.append(chosen)
    return Event(primitives=primitives)


# ---------------------------------------------------------------------------
# Stage 3 — full pipeline (duration retry → per-occurrence build → cross-prim retry)
# ---------------------------------------------------------------------------

def build_jittered_occurrences(
    cfg: Config,
    schema: Schema,
    layout: Layout,
    event_template: Event,
    inter_event: list[list[Primitive]],
    rng: np.random.Generator,
    constraints_result: ConstraintsResult,
    *,
    max_attempts: int = 200,
) -> tuple[list[Event], JitterReport]:
    """
    Step 17: produce N event occurrences from the shared template by jittering
    each occurrence's definitive + random features, enforcing all
    level-independent (non-global) constraints from `constraints_result`.

    Pipeline:
      1. Sample the (nb_events x nb_in_event) duration matrix with sum
         preservation; retry until duration-only constraints pass.
      2. For each occurrence, build sequentially. Per primitive, retry until
         slope + intra-primitive constraints pass.
      3. After the occurrence is built, evaluate cross-primitive constraints.
         If any fail, retry the whole occurrence (durations stay fixed).

    Raises ConstraintRetryExhausted on any stage exhaustion.
    """
    nb_events = layout.nb_events
    template_types = [p.type for p in event_template.primitives]
    classified = _classify_constraints(constraints_result, schema, template_types)

    template_durations = [
        int(p.definitive["duration"]) for p in event_template.primitives
    ]
    min_per_position = [
        cfg.min_spike_length
        if p.type in (PrimitiveType.SPIKE, PrimitiveType.DIP)
        else cfg.min_primitive_length
        for p in event_template.primitives
    ]
    durations_by_occ = _sample_durations_with_retry(
        template_durations,
        nb_events,
        cfg.jitter_randomness,
        rng,
        min_per_position,
        classified.duration,
        max_attempts=max_attempts,
    )

    occurrences: list[Event] = []
    report = JitterReport()

    for k in range(nb_events):
        # Determine boundary neighbours.
        # Left: last primitive of inter_event[k] if non-empty, else last primitive
        # of the previously-built occurrence (or None for the very first event).
        if inter_event[k]:
            left = inter_event[k][-1]
        elif occurrences:
            left = occurrences[-1].primitives[-1] if occurrences[-1].primitives else None
        else:
            left = None
        # Right: first primitive of inter_event[k+1] if non-empty, else first
        # primitive of the template (next occurrence will start with the same shape).
        if inter_event[k + 1]:
            right = inter_event[k + 1][0]
        elif event_template.primitives:
            right = event_template.primitives[0]
        else:
            right = None

        chosen_occ: Event | None = None
        last_inner_error: ConstraintRetryExhausted | None = None
        for _ in range(max_attempts):
            try:
                occ = jitter_event_occurrence(
                    event_template,
                    schema,
                    cfg,
                    rng,
                    boundary_prev=left,
                    boundary_next=right,
                    jittered_durations=durations_by_occ[k],
                    intra_constraints_by_pos=classified.intra_by_pos,
                    max_attempts=max_attempts,
                    event_index=k,
                )
            except ConstraintRetryExhausted as e:
                last_inner_error = e
                continue
            feats_by_pos = [_prim_feats(p) for p in occ.primitives]
            if _check_cross(classified.cross_by_pos, feats_by_pos):
                chosen_occ = occ
                break

        if chosen_occ is None:
            msg = (
                f"event={k}: cross-primitive constraints could not be satisfied "
                f"within {max_attempts} occurrence attempts"
            )
            if last_inner_error is not None:
                msg += f" (last inner failure: {last_inner_error})"
            raise ConstraintRetryExhausted(msg)
        occurrences.append(chosen_occ)
    return occurrences, report


def assemble_with_occurrences(
    layout: Layout,
    occurrences: list[Event],
    inter_event: list[list[Primitive]],
) -> list[Primitive | Event]:
    """N possibly-different Events interleaved with inter_event slots."""
    items: list[Primitive | Event] = []
    for k, slot in enumerate(inter_event):
        items.extend(slot)
        if k < layout.nb_events:
            items.append(occurrences[k])
    return items
