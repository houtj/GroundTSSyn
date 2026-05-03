from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_gen.features import extract
from data_gen.global_features import compute_globals
from data_gen.layout import Layout
from data_gen.types import ConcatResult, Event, PrimitiveType


@dataclass
class NormedFeatures:
    """
    Per-primitive features recomputed from the z-normed signal (step 12).

    `in_event[occurrence_idx][template_pos]` gives the full feature dict for
    the template-position primitive within that occurrence. Pre-jitter, level-
    independent features are identical across occurrences; level-dependent
    features (`*_value`) differ because of the upstream shift.

    `inter_event[slot_idx][position_in_slot]` mirrors the layout's inter-event
    structure. Included for completeness; not used for constraint generation.

    `globals` holds the `global.*` pseudo-primitive features.
    """

    in_event: list[list[dict[str, Any]]] = field(default_factory=list)
    inter_event: list[list[dict[str, Any]]] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)


def _merge_random(
    feats: dict[str, Any], type_: PrimitiveType, random: dict[str, Any]
) -> dict[str, Any]:
    """Merge non-recoverable random fields back from the source primitive."""
    if type_ == PrimitiveType.OSCILLATION:
        for key in ("waveform_shape", "decay", "resonance", "regularity"):
            if random.get(key) is not None:
                feats[key] = random[key]
    elif type_ in (PrimitiveType.INCREASE, PrimitiveType.DECREASE):
        if random.get("shape") is not None:
            feats["shape"] = random["shape"]
    return feats


def recompute_features(
    raw: ConcatResult,
    z_signal: list[float],
    layout: Layout,
    event_template: Event,
) -> NormedFeatures:
    """Step 12: re-extract features from the z-normed signal slices."""
    nb_events = layout.nb_events
    nb_in_event = len(event_template.primitives)

    in_event: list[list[dict[str, Any]]] = [
        [dict() for _ in range(nb_in_event)] for _ in range(nb_events)
    ]
    inter_event: list[list[dict[str, Any]]] = [
        [dict() for _ in range(k)] for k in layout.inter_event_counts
    ]

    slot_idx = 0
    pos_in_slot = 0
    pos_in_event = 0
    last_event_index: int | None = None

    all_p2ps: list[float] = []

    for placed in raw.placed:
        s, e = placed.span
        z_slice = z_signal[s : e + 1]
        feats = extract(z_slice, placed.primitive.type)
        _merge_random(feats, placed.primitive.type, placed.primitive.random)

        p2p = feats.get("peak2peak")
        if p2p is not None:
            all_p2ps.append(float(p2p))

        if placed.event_index is not None:
            ev = placed.event_index
            if ev != last_event_index:
                pos_in_event = 0
                last_event_index = ev
            in_event[ev][pos_in_event] = feats
            pos_in_event += 1
        else:
            if last_event_index is not None:
                slot_idx = last_event_index + 1
                pos_in_slot = 0
                last_event_index = None
            inter_event[slot_idx][pos_in_slot] = feats
            pos_in_slot += 1

    globals_ = compute_globals(z_signal, all_p2ps)
    return NormedFeatures(in_event=in_event, inter_event=inter_event, globals=globals_)
