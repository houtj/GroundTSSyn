from __future__ import annotations

import math
from typing import Any

from data_gen.features import extract
from data_gen.grounding import (
    BasicOutput,
    EventOccurrenceOut,
    InEventLayoutEntry,
    InterEventLayoutEntry,
    LayoutEntry,
    PrimitiveOccurrenceOut,
    TemplateOut,
    TemplatePrimitiveOut,
)
from data_gen.types import PrimitiveType
from data_gen.visibility import compute_visibility


def sanitize(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with JSON-safe values."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value
    return obj


def _split_vis(feats: dict[str, Any]) -> tuple[float | None, float | None, dict[str, Any]]:
    """Return (visibility, cp_visibility, features_without_vis_keys)."""
    vis = feats.get("visibility")
    cp_vis = feats.get("cp_visibility")
    features = {k: v for k, v in feats.items() if k not in ("visibility", "cp_visibility")}
    return vis, cp_vis, features


def _build_ev_row_mapping(raw_jittered: Any, nb_in_event: int) -> dict[int, int]:
    """
    Map raw event_index -> row index in jittered_features.in_event.

    recompute_features drops periodical-edge events (those missing their first
    or last template primitive), so the row in .in_event doesn't always equal
    the event_index. This replicates the same filtering logic.
    """
    by_event: dict[int, dict[int, bool]] = {}
    for placed in raw_jittered.placed:
        if placed.event_index is None:
            continue
        tpos = placed.template_position
        if tpos is None or not (0 <= tpos < nb_in_event):
            continue
        by_event.setdefault(placed.event_index, {})[tpos] = True

    mapping: dict[int, int] = {}
    row = 0
    for ev_idx in sorted(by_event):
        positions = by_event[ev_idx]
        if nb_in_event > 0 and (0 not in positions or (nb_in_event - 1) not in positions):
            continue
        mapping[ev_idx] = row
        row += 1
    return mapping


def enrich_basic_output(
    basic: BasicOutput,
    *,
    raw_jittered: Any,           # ConcatResult
    event_template: Any,         # Event
    layout: Any,                 # Layout
    jittered_features: Any,      # NormedFeatures
    noise_sigma: float,
    post_match_extras: list[Any],  # list[ExtraOccurrence]
    z_signal: list[float],
) -> BasicOutput:
    """
    Populate the new enrichment fields on BasicOutput:
      - signal_length, noise_sigma, globals
      - template (event layout)
      - whole_time_series_layout (all primitives in temporal order)
      - events (in-event primitives grouped by occurrence)

    Existing signal/signal_noisy/grounding/event_spans are passed through unchanged.
    """
    nb_in_event = len(event_template.primitives)
    ev_row = _build_ev_row_mapping(raw_jittered, nb_in_event)

    # Determine which event span starts survived the grounding filter.
    kept_ev_starts: set[int] = {s[0] for s in basic.event_spans}

    # Build template block from the original event template (pre-jitter types/durations).
    template = TemplateOut(
        nb_primitives=nb_in_event,
        primitives=[
            TemplatePrimitiveOut(
                template_index=i,
                type=p.type,
                duration=layout.event_template_durations[i],
            )
            for i, p in enumerate(event_template.primitives)
        ],
    )

    # Walk placed primitives in order to build the temporal layout.
    # Track inter-event slot/position counters in the same way recompute_features does.
    slot_idx = 0
    pos_in_slot = 0
    last_event_index: int | None = None

    whole_layout: list[LayoutEntry] = []
    # event_index -> ordered list of PrimitiveOccurrenceOut
    event_primitives: dict[int, list[PrimitiveOccurrenceOut]] = {}

    for placed in raw_jittered.placed:
        ev = placed.event_index
        tpos = placed.template_position

        if ev is not None:
            # In-event primitive
            last_event_index = ev

            feats_raw: dict[str, Any] = {}
            row = ev_row.get(ev)
            if row is not None and tpos is not None:
                try:
                    feats_raw = dict(jittered_features.in_event[row][tpos])
                except (IndexError, KeyError, TypeError):
                    pass

            vis, cp_vis, features = _split_vis(feats_raw)
            safe_tpos = tpos if tpos is not None else 0

            prim_out = PrimitiveOccurrenceOut(
                template_index=safe_tpos,
                type=placed.primitive.type,
                span=tuple(placed.span),
                visibility=vis,
                cp_visibility=cp_vis,
                features=sanitize(features),
            )
            event_primitives.setdefault(ev, []).append(prim_out)

            whole_layout.append(InEventLayoutEntry(
                kind="in_event",
                event_index=ev,
                template_index=safe_tpos,
                type=placed.primitive.type,
                span=tuple(placed.span),
                visibility=vis,
                cp_visibility=cp_vis,
                features=sanitize(features),
            ))
        else:
            # Inter-event primitive
            if last_event_index is not None:
                slot_idx = last_event_index + 1
                pos_in_slot = 0
                last_event_index = None

            feats_raw = {}
            try:
                slot_feats = jittered_features.inter_event[slot_idx][pos_in_slot]
                if slot_feats:
                    feats_raw = dict(slot_feats)
            except (IndexError, KeyError, TypeError):
                pass

            if not feats_raw:
                # Periodical mode: demoted edge-event primitives land here but
                # inter_event_counts=0 so nothing is stored in jittered_features.
                # Re-extract from the signal slice.
                s, e = placed.span
                feats_raw = extract(z_signal[s: e + 1], placed.primitive.type)
                vis_val = compute_visibility(feats_raw, placed.primitive.type, None, None, noise_sigma)
                feats_raw["visibility"] = vis_val
                feats_raw["cp_visibility"] = None

            vis, cp_vis, features = _split_vis(feats_raw)

            whole_layout.append(InterEventLayoutEntry(
                kind="inter_event",
                slot_index=slot_idx,
                position_in_slot=pos_in_slot,
                type=placed.primitive.type,
                span=tuple(placed.span),
                visibility=vis,
                cp_visibility=cp_vis,
                features=sanitize(features),
            ))
            pos_in_slot += 1

    # Build events list: keep only events whose span start survived grounding filter.
    template_types = [p.type for p in event_template.primitives]
    events: list[EventOccurrenceOut] = []
    for ev_idx, prims in sorted(event_primitives.items()):
        if not prims:
            continue
        prims_sorted = sorted(prims, key=lambda p: p.template_index)
        ev_span_start = prims_sorted[0].span[0]
        ev_span_end = prims_sorted[-1].span[1]
        if ev_span_start not in kept_ev_starts:
            continue
        events.append(EventOccurrenceOut(
            event_index=ev_idx,
            span=(ev_span_start, ev_span_end),
            source="template",
            primitives=prims_sorted,
        ))

    # Add post-match extra occurrences.
    for occ in post_match_extras:
        prims_out: list[PrimitiveOccurrenceOut] = []
        for i, span in enumerate(occ.primitive_spans):
            s, e = span
            ptype = template_types[i] if i < len(template_types) else PrimitiveType.PLATEAU
            z_slice = z_signal[s: e + 1]
            feats_raw = extract(z_slice, ptype)
            vis = compute_visibility(feats_raw, ptype, None, None, noise_sigma)
            _, _, features = _split_vis(feats_raw)
            prims_out.append(PrimitiveOccurrenceOut(
                template_index=i,
                type=ptype,
                span=tuple(span),
                visibility=vis,
                cp_visibility=None,
                features=sanitize(features),
            ))
        events.append(EventOccurrenceOut(
            event_index=-1,
            span=tuple(occ.event_span),
            source="post_match",
            primitives=prims_out,
        ))

    events.sort(key=lambda e: e.span[0])

    return BasicOutput(
        schema_version=basic.schema_version,
        schema_doc=basic.schema_doc,
        signal=basic.signal,
        signal_noisy=basic.signal_noisy,
        signal_length=len(basic.signal),
        noise_sigma=noise_sigma,
        globals=sanitize(jittered_features.globals),
        template=template,
        events=events,
        whole_time_series_layout=whole_layout,
        grounding=basic.grounding,
        event_spans=basic.event_spans,
    )
