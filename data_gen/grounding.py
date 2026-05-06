from __future__ import annotations

from pydantic import BaseModel

from data_gen.constraints import Constraint, ConstraintsResult, evaluate_constraint
from data_gen.post_match import ExtraOccurrence
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
    signal_noisy: list[float]
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
    z_signal_noisy: list[float],
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
        signal_noisy=list(z_signal_noisy),
        grounding=grounding,
        event_spans=[tuple(s) for s in raw.event_spans],
    )


def apply_extra_occurrences(
    basic: BasicOutput,
    extras: list[ExtraOccurrence],
) -> tuple[BasicOutput, list[int]]:
    """
    Append post-match-discovered occurrences to `basic` and re-sort all
    parallel per-occurrence lists by event-span start, so the output is
    temporally consistent.

    Returns the augmented BasicOutput and the list of indices (in the new
    ordering) that came from the post-pass.
    """
    if not extras:
        return basic, []

    event_spans = [tuple(s) for s in basic.event_spans]
    spans_by_pos: list[list[tuple[int, int]]] = [
        [tuple(s) for s in g.spans] for g in basic.grounding
    ]
    is_extra: list[bool] = [False] * len(event_spans)

    for occ in extras:
        if len(occ.primitive_spans) != len(spans_by_pos):
            raise ValueError(
                f"extra occurrence has {len(occ.primitive_spans)} primitive spans "
                f"but template has {len(spans_by_pos)} positions"
            )
        event_spans.append(tuple(occ.event_span))
        for pos, span in enumerate(occ.primitive_spans):
            spans_by_pos[pos].append(tuple(span))
        is_extra.append(True)

    order = sorted(range(len(event_spans)), key=lambda i: event_spans[i][0])
    sorted_event_spans = [event_spans[i] for i in order]
    sorted_spans_by_pos = [[spans[i] for i in order] for spans in spans_by_pos]
    extra_indices = [new_i for new_i, old_i in enumerate(order) if is_extra[old_i]]

    new_grounding = [
        GroundedPrimitiveOut(
            template_index=g.template_index,
            type=g.type,
            constraints=list(g.constraints),
            spans=sorted_spans_by_pos[i],
        )
        for i, g in enumerate(basic.grounding)
    ]

    return (
        BasicOutput(
            signal=list(basic.signal),
            signal_noisy=list(basic.signal_noisy),
            grounding=new_grounding,
            event_spans=sorted_event_spans,
        ),
        extra_indices,
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


def filter_failed_constraints(
    basic: BasicOutput,
    verification: list[OccurrenceVerification],
) -> tuple[BasicOutput, list[OccurrenceVerification]]:
    """
    Remove constraints that failed (satisfied=False) for any occurrence from
    both the BasicOutput grounding and the verification list.
    """
    failed_pairs: set[tuple[int, int]] = set()
    for v in verification:
        if v.satisfied is False:
            failed_pairs.add((v.template_index, v.constraint_index))

    if not failed_pairs:
        return basic, verification

    new_grounding = []
    for g in basic.grounding:
        kept = [
            c for idx, c in enumerate(g.constraints)
            if (g.template_index, idx) not in failed_pairs
        ]
        new_grounding.append(
            GroundedPrimitiveOut(
                template_index=g.template_index,
                type=g.type,
                constraints=kept,
                spans=g.spans,
            )
        )

    new_verification = [
        v for v in verification
        if (v.template_index, v.constraint_index) not in failed_pairs
    ]

    return (
        BasicOutput(
            signal=list(basic.signal),
            signal_noisy=list(basic.signal_noisy),
            grounding=new_grounding,
            event_spans=list(basic.event_spans),
        ),
        new_verification,
    )


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
