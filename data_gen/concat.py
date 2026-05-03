from __future__ import annotations

import numpy as np

from data_gen.types import ConcatResult, Event, PlacedPrimitive, Primitive


def _flatten(layout: list[Primitive | Event]) -> list[tuple[Primitive, int | None]]:
    """Expand events into their constituent primitives, tagging each with its event_index."""
    flat: list[tuple[Primitive, int | None]] = []
    event_index = 0
    for item in layout:
        if isinstance(item, Event):
            for p in item.primitives:
                flat.append((p, event_index))
            event_index += 1
        elif isinstance(item, Primitive):
            flat.append((item, None))
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

    for i, (prim, event_idx) in enumerate(flat):
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
