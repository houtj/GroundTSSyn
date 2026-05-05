from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_gen.constraints import Constraint, evaluate_constraint
from data_gen.features import extract
from data_gen.recompute import _merge_random
from data_gen.types import ConcatResult, Event, PlacedPrimitive, PrimitiveType


@dataclass
class ExtraOccurrence:
    event_span: tuple[int, int]
    primitive_spans: list[tuple[int, int]]


def _features_for_window(
    window: list[PlacedPrimitive],
    z_signal: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pp in window:
        s, e = pp.span
        feats = extract(z_signal[s : e + 1], pp.primitive.type)
        _merge_random(feats, pp.primitive.type, pp.primitive.random)
        out.append(feats)
    return out


def _all_constraints_pass(
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


def find_extra_occurrences(
    raw_jittered: ConcatResult,
    z_signal: list[float],
    event_template: Event,
    constraints_by_pos: list[list[Constraint]],
    globals_: dict[str, Any],
) -> list[ExtraOccurrence]:
    """
    Scan inter-event primitive runs for windows whose primitive-type sequence
    matches the event template and whose features satisfy every constraint in
    `constraints_by_pos` (one list per template position, already filtered to
    only surviving/successful constraints). Returns each accepted window as an
    ExtraOccurrence.
    """
    template_types: list[PrimitiveType] = [p.type for p in event_template.primitives]
    M = len(template_types)
    if M == 0:
        return []

    placed = raw_jittered.placed

    runs: list[list[int]] = []
    current: list[int] = []
    for i, pp in enumerate(placed):
        if pp.event_index is None:
            current.append(i)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    extras: list[ExtraOccurrence] = []
    for run in runs:
        if len(run) < M:
            continue
        i = 0
        while i + M <= len(run):
            window_idxs = run[i : i + M]
            window = [placed[j] for j in window_idxs]
            if [pp.primitive.type for pp in window] != template_types:
                i += 1
                continue
            feats_by_pos = _features_for_window(window, z_signal)
            if not _all_constraints_pass(feats_by_pos, constraints_by_pos, globals_):
                i += 1
                continue
            event_span = (window[0].span[0], window[-1].span[1])
            primitive_spans = [pp.span for pp in window]
            extras.append(
                ExtraOccurrence(
                    event_span=event_span,
                    primitive_spans=primitive_spans,
                )
            )
            i += M
    return extras
