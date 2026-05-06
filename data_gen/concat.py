from __future__ import annotations

import numpy as np

from data_gen.types import ConcatResult, Event, PlacedPrimitive, Primitive


def _flatten(
    layout: list[Primitive | Event],
) -> list[tuple[Primitive, int | None, int | None]]:
    """Expand events into their constituent primitives, tagging each with its
    event_index and (for in-event primitives) its template position."""
    flat: list[tuple[Primitive, int | None, int | None]] = []
    event_index = 0
    for item in layout:
        if isinstance(item, Event):
            for tpos, p in enumerate(item.primitives):
                flat.append((p, event_index, tpos))
            event_index += 1
        elif isinstance(item, Primitive):
            flat.append((item, None, None))
        else:
            raise TypeError(
                f"layout entry must be Primitive or Event, got {type(item).__name__}"
            )
    return flat


def concatenate(layout: list[Primitive | Event]) -> ConcatResult:
    """
    Shift+merge concatenation of primitives in temporal order.

    For each adjacent pair, shift the next primitive so its first sample equals
    the previous primitive's last sample, then drop that first sample (shared
    boundary). No z-normalization is performed.

    Spans are inclusive [start, end]; adjacent primitives share their boundary
    index, so primitive[i].span[1] == primitive[i+1].span[0]. The PlacedPrimitive
    stores the full shifted values array (length = duration + 1), so adjacent
    placed primitives have the same numeric value at their shared index.

    Final length: for n primitives with lengths L_i (= duration_i + 1),
    total = sum(L_i) - (n - 1).
    """
    if not layout:
        return ConcatResult(signal=[], placed=[], layout=[], event_spans=[])

    flat = _flatten(layout)
    n_events = sum(1 for item in layout if isinstance(item, Event))

    placed: list[PlacedPrimitive] = []
    event_spans_starts: dict[int, int] = {}
    event_spans_ends: dict[int, int] = {}

    for i, (prim, event_idx, tpos) in enumerate(flat):
        v = np.asarray(prim.values, dtype=float)
        if i == 0:
            shifted = v
            span_start = 0
        else:
            prev_last = placed[-1].primitive.values[-1]
            shift = prev_last - float(v[0])
            shifted = v + shift
            span_start = placed[-1].span[1]      # shared boundary
        span_end = span_start + len(shifted) - 1

        placed.append(
            PlacedPrimitive(
                primitive=Primitive(
                    type=prim.type,
                    values=shifted.tolist(),
                    definitive=prim.definitive,
                    random=prim.random,
                    derived=prim.derived,
                ),
                span=(span_start, span_end),
                event_index=event_idx,
                template_position=tpos,
            )
        )

        if event_idx is not None:
            event_spans_starts.setdefault(event_idx, span_start)
            event_spans_ends[event_idx] = span_end

    # Build the global signal: first primitive in full, then each subsequent
    # primitive minus its (duplicated) first sample.
    signal_arr = np.concatenate(
        [np.asarray(placed[0].primitive.values, dtype=float)]
        + [np.asarray(pp.primitive.values, dtype=float)[1:] for pp in placed[1:]]
    )
    event_spans = [
        (event_spans_starts[i], event_spans_ends[i]) for i in range(n_events)
    ]

    return ConcatResult(
        signal=signal_arr.tolist(),
        placed=placed,
        layout=list(layout),
        event_spans=event_spans,
    )


def truncate_concat_result(
    raw: ConcatResult, left_trim: int, signal_length: int, nb_in_event: int
) -> ConcatResult:
    """
    Crop a ConcatResult to the window [left_trim, left_trim + signal_length - 1]
    in the original (full-tile) signal coordinate system. Used by the periodical
    pipeline so the M-event tile (which overshoots the signal length) becomes a
    signal of exactly `signal_length` samples.

    Per-primitive values and spans are clipped to the visible window; primitives
    that fall entirely outside it are dropped. Event spans are clipped likewise;
    events with no remaining visible primitives are dropped from event_spans.

    Per the periodical-mode rule: an event whose first or last template
    primitive has been truncated entirely is no longer considered an event —
    its surviving primitives are re-tagged as inter-event (event_index=None,
    template_position=None) so downstream consumers (grounding, plotting,
    post_match) treat them accordingly.
    """
    if left_trim < 0 or signal_length <= 0:
        raise ValueError(
            f"invalid truncation: left_trim={left_trim}, signal_length={signal_length}"
        )
    full_len = len(raw.signal)
    right_excl = left_trim + signal_length
    if right_excl > full_len:
        raise ValueError(
            f"truncation window [{left_trim}, {right_excl}) exceeds full signal "
            f"length {full_len}"
        )

    new_signal = raw.signal[left_trim:right_excl]
    win_end = left_trim + signal_length - 1

    # First pass: detect events whose first or last template primitive has been
    # truncated entirely. Those events get demoted to inter-event in pass 2.
    visible_positions_by_event: dict[int, set[int]] = {}
    for pp in raw.placed:
        if pp.event_index is None or pp.template_position is None:
            continue
        a, b = pp.span
        if max(a, left_trim) > min(b, win_end):
            continue
        visible_positions_by_event.setdefault(pp.event_index, set()).add(
            pp.template_position
        )
    demoted_events: set[int] = {
        ev for ev, positions in visible_positions_by_event.items()
        if 0 not in positions or (nb_in_event - 1) not in positions
    }

    # Compact event_index for surviving events: original ev_idx -> 0..K-1.
    surviving_orig = sorted(
        ev for ev, positions in visible_positions_by_event.items()
        if ev not in demoted_events
    )
    event_index_remap: dict[int, int] = {
        orig: new for new, orig in enumerate(surviving_orig)
    }

    new_placed: list[PlacedPrimitive] = []
    for pp in raw.placed:
        a, b = pp.span
        vis_a = max(a, left_trim)
        vis_b = min(b, win_end)
        if vis_a > vis_b:
            continue
        cut_left = vis_a - a
        cut_right = b - vis_b
        new_values = pp.primitive.values[cut_left : len(pp.primitive.values) - cut_right]
        new_span = (vis_a - left_trim, vis_b - left_trim)
        if pp.event_index is None or pp.event_index in demoted_events:
            new_event_index: int | None = None
            new_tpos: int | None = None
        else:
            new_event_index = event_index_remap[pp.event_index]
            new_tpos = pp.template_position
        new_placed.append(
            PlacedPrimitive(
                primitive=Primitive(
                    type=pp.primitive.type,
                    values=list(new_values),
                    definitive=pp.primitive.definitive,
                    random=pp.primitive.random,
                    derived=pp.primitive.derived,
                ),
                span=new_span,
                event_index=new_event_index,
                template_position=new_tpos,
            )
        )

    new_event_spans: list[tuple[int, int]] = []
    for orig_ev_idx in surviving_orig:
        if orig_ev_idx >= len(raw.event_spans):
            continue
        s, e = raw.event_spans[orig_ev_idx]
        vis_s = max(s, left_trim)
        vis_e = min(e, win_end)
        if vis_s > vis_e:
            continue
        new_event_spans.append((vis_s - left_trim, vis_e - left_trim))

    return ConcatResult(
        signal=list(new_signal),
        placed=new_placed,
        layout=list(raw.layout),
        event_spans=new_event_spans,
    )
