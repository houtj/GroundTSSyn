from __future__ import annotations

from pydantic import BaseModel

from data_gen.constraints import Constraint, ConstraintsResult, evaluate_constraint
from data_gen.recompute import NormedFeatures
from data_gen.types import ConcatResult, PrimitiveType


class GroundedPrimitiveOut(BaseModel):
    template_index: int
    type: PrimitiveType
    constraints: list[Constraint]
    spans: list[tuple[int, int]]


class BasicOutput(BaseModel):
    """The pipeline's externally-visible output (per pipline.md)."""

    signal: list[float]
    grounding: list[GroundedPrimitiveOut]
    event_spans: list[tuple[int, int]]


class OccurrenceVerification(BaseModel):
    event_index: int
    template_index: int
    constraint_index: int
    satisfied: bool | None  # None when unevaluable


class VerificationSummary(BaseModel):
    total: int
    passed: int
    failed: int
    unevaluable: int


def build_basic_output(
    z_signal: list[float],
    raw: ConcatResult,
    template_types: list[PrimitiveType],
    constraints_result: ConstraintsResult,
) -> BasicOutput:
    """
    Step 19: assemble the per-template-position spans (one span per
    occurrence) and bundle with the constraints from step 16. Inter-event
    primitives are excluded from grounding (per spec).
    """
    nb_in_event = len(template_types)
    spans_by_pos: list[list[tuple[int, int]]] = [[] for _ in range(nb_in_event)]

    pos_in_event = 0
    last_event_index: int | None = None
    for placed in raw.placed:
        if placed.event_index is None:
            continue
        ev = placed.event_index
        if ev != last_event_index:
            pos_in_event = 0
            last_event_index = ev
        if pos_in_event < nb_in_event:
            spans_by_pos[pos_in_event].append(tuple(placed.span))
        pos_in_event += 1

    grounded_by_pos = {g.template_index: g for g in constraints_result.grounded}
    grounding: list[GroundedPrimitiveOut] = []
    for pos in range(nb_in_event):
        g = grounded_by_pos.get(pos)
        constraints = list(g.constraints) if g is not None else []
        grounding.append(
            GroundedPrimitiveOut(
                template_index=pos,
                type=template_types[pos],
                constraints=constraints,
                spans=spans_by_pos[pos],
            )
        )

    return BasicOutput(
        signal=list(z_signal),
        grounding=grounding,
        event_spans=[tuple(s) for s in raw.event_spans],
    )


def verify_constraints(
    constraints_result: ConstraintsResult,
    jittered: NormedFeatures,
) -> list[OccurrenceVerification]:
    """
    For every (occurrence, template_position, constraint), record whether
    the constraint is satisfied by the post-jitter, post-z-norm features.
    Level-independent constraints will pass uniformly across occurrences;
    level-dependent ones may differ.
    """
    out: list[OccurrenceVerification] = []
    for occ_idx, feats_by_pos in enumerate(jittered.in_event):
        for grounded in constraints_result.grounded:
            for c_idx, c in enumerate(grounded.constraints):
                sat = evaluate_constraint(c, feats_by_pos, jittered.globals)
                out.append(
                    OccurrenceVerification(
                        event_index=occ_idx,
                        template_index=grounded.template_index,
                        constraint_index=c_idx,
                        satisfied=sat,
                    )
                )
    return out


def summarize_verification(
    verifications: list[OccurrenceVerification],
) -> VerificationSummary:
    total = len(verifications)
    passed = sum(1 for v in verifications if v.satisfied is True)
    failed = sum(1 for v in verifications if v.satisfied is False)
    unevaluable = sum(1 for v in verifications if v.satisfied is None)
    return VerificationSummary(
        total=total, passed=passed, failed=failed, unevaluable=unevaluable
    )
