from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from data_gen.constraints import Constraint, ConstraintsResult, evaluate_constraint
from data_gen.negation import NegativeGrounding
from data_gen.post_match import ExtraOccurrence
from data_gen.recompute import NormedFeatures
from data_gen.types import ConcatResult, PrimitiveType

_SCHEMA_DOC = (
    "Synthetic time-series sample. "
    "`signal` is the clean z-normed signal; `signal_noisy` adds Gaussian noise of std `noise_sigma`. "
    "The signal is composed of `events` (each an instance of `template`) separated by inter-event primitives. "
    "`whole_time_series_layout` lists every primitive in temporal order with `kind` ('in_event' or 'inter_event'), "
    "`visibility` in [0,1] (how perceptible the primitive is given its amplitude/duration relative to noise), "
    "`cp_visibility` in [0,1] (how perceptible the change-point at its boundary is), and `features` (realized properties). "
    "`events` groups in-event primitives by occurrence for event-level reasoning. "
    "`template` describes the abstract event pattern that each occurrence instantiates. "
    "`grounding` lists template-level constraints satisfied by ALL event occurrences. "
    "`globals` holds signal-level statistics."
)


class TemplatePrimitiveOut(BaseModel):
    template_index: int
    type: PrimitiveType
    duration: int


class TemplateOut(BaseModel):
    nb_primitives: int
    primitives: list[TemplatePrimitiveOut]


class PrimitiveOccurrenceOut(BaseModel):
    template_index: int
    type: PrimitiveType
    span: tuple[int, int]
    visibility: float | None
    cp_visibility: float | None
    features: dict[str, Any]


class EventOccurrenceOut(BaseModel):
    event_index: int
    span: tuple[int, int]
    source: Literal["template", "post_match"]
    primitives: list[PrimitiveOccurrenceOut]


class InEventLayoutEntry(BaseModel):
    kind: Literal["in_event"]
    event_index: int
    template_index: int
    type: PrimitiveType
    span: tuple[int, int]
    visibility: float | None
    cp_visibility: float | None
    features: dict[str, Any]


class InterEventLayoutEntry(BaseModel):
    kind: Literal["inter_event"]
    slot_index: int
    position_in_slot: int
    type: PrimitiveType
    span: tuple[int, int]
    visibility: float | None
    cp_visibility: float | None
    features: dict[str, Any]


LayoutEntry = InEventLayoutEntry | InterEventLayoutEntry


class GroundedPrimitiveOut(BaseModel):
    template_index: int
    type: PrimitiveType
    constraints: list[Constraint]
    spans: list[tuple[int, int]]


class BasicOutput(BaseModel):
    """The pipeline's externally-visible output (per pipline.md)."""

    schema_version: str = "2"
    schema_doc: str = _SCHEMA_DOC
    signal: list[float]
    signal_noisy: list[float]
    signal_length: int = 0
    noise_sigma: float = 0.0
    globals: dict[str, Any] = {}
    template: TemplateOut | None = None
    events: list[EventOccurrenceOut] = []
    whole_time_series_layout: list[LayoutEntry] = []
    grounding: list[GroundedPrimitiveOut]
    event_spans: list[tuple[int, int]]
    neg_groundings: list[NegativeGrounding] = []


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

    Periodical mode: the first/last event may be partial after truncation. Per
    spec, an event is dropped from the annotation if its first or last template
    primitive is missing entirely. The associated event_span is also dropped.
    """
    nb_in_event = len(template_types)

    # Group in-event placed primitives by event_index, indexed by template_position.
    by_event: dict[int, dict[int, tuple[int, int]]] = {}
    for placed in raw.placed:
        if placed.event_index is None:
            continue
        tpos = placed.template_position
        if tpos is None or tpos >= nb_in_event:
            continue
        by_event.setdefault(placed.event_index, {})[tpos] = tuple(placed.span)

    spans_by_pos: list[list[tuple[int, int]]] = [[] for _ in range(nb_in_event)]
    kept_events: list[int] = []
    for ev_idx in sorted(by_event):
        positions = by_event[ev_idx]
        # Drop if the FIRST template primitive is missing (e.g., first event
        # truncated past position 0) or the LAST is missing (last event).
        if 0 not in positions or (nb_in_event - 1) not in positions:
            continue
        kept_events.append(ev_idx)
        for pos, span in positions.items():
            spans_by_pos[pos].append(span)

    kept_event_spans = [
        tuple(raw.event_spans[ev_idx]) for ev_idx in kept_events
        if ev_idx < len(raw.event_spans)
    ]

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
        event_spans=kept_event_spans,
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
