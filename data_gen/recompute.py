from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_gen.features import extract
from data_gen.global_features import compute_globals
from data_gen.layout import Layout
from data_gen.types import ConcatResult, Event, PrimitiveType
from data_gen.visibility import compute_cp_visibility, compute_visibility


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
        for key in ("waveform_shape", "envelope_behavior", "envelope_asymmetry", "regularity"):
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
    sigma: float = 0.0,
) -> NormedFeatures:
    """Step 12: re-extract features from the z-normed signal slices.

    `sigma` is the noise std on the z-normed scale; passed through to the
    per-primitive visibility computation. Pass 0 when no noise is added.

    In periodical mode, the first/last edge events may be partial after
    truncation. Per the grounding rule, an event with its first or last
    template primitive missing is dropped from `in_event` so that downstream
    consumers (verify_constraints, etc.) see only events that match the
    grounding's `event_spans`.
    """
    nb_in_event = len(event_template.primitives)

    # Group placed in-event primitives by event_index, indexed by template
    # position, so we can detect events whose first/last template position is
    # missing (periodical edge truncation).
    by_event: dict[int, dict[int, dict[str, Any]]] = {}

    inter_event: list[list[dict[str, Any]]] = [
        [dict() for _ in range(k)] for k in layout.inter_event_counts
    ]

    slot_idx = 0
    pos_in_slot = 0
    last_event_index: int | None = None

    all_p2ps: list[float] = []
    flat_feats: list[dict[str, Any]] = []

    for placed in raw.placed:
        s, e = placed.span
        z_slice = z_signal[s : e + 1]
        feats = extract(z_slice, placed.primitive.type)
        _merge_random(feats, placed.primitive.type, placed.primitive.random)

        p2p = feats.get("peak2peak")
        if p2p is not None:
            all_p2ps.append(float(p2p))

        flat_feats.append(feats)

        if placed.event_index is not None:
            ev = placed.event_index
            last_event_index = ev
            tpos = placed.template_position
            if tpos is not None and 0 <= tpos < nb_in_event:
                by_event.setdefault(ev, {})[tpos] = feats
        else:
            if last_event_index is not None:
                slot_idx = last_event_index + 1
                pos_in_slot = 0
                last_event_index = None
            if slot_idx < len(inter_event) and pos_in_slot < len(inter_event[slot_idx]):
                inter_event[slot_idx][pos_in_slot] = feats
            pos_in_slot += 1

    # Keep only events with both first and last template positions present.
    # Aperiodic events always have both; periodical edge events may not.
    in_event: list[list[dict[str, Any]]] = []
    for ev_idx in sorted(by_event):
        positions = by_event[ev_idx]
        if 0 not in positions or (nb_in_event - 1) not in positions:
            continue
        in_event.append(
            [positions.get(pos, dict()) for pos in range(nb_in_event)]
        )

    n = len(flat_feats)
    for i, placed in enumerate(raw.placed):
        left = flat_feats[i - 1] if i > 0 else None
        right = flat_feats[i + 1] if i + 1 < n else None
        flat_feats[i]["visibility"] = compute_visibility(
            flat_feats[i], placed.primitive.type, left, right, sigma
        )

    # Pass 3: per-change-point visibility, then per-primitive aggregate.
    # Signal-edge CPs are skipped — edge primitives use their inner CP only.
    cp_vis: list[float | None] = [None] * (n + 1)  # cp_vis[i] = CP between i-1 and i
    for i in range(1, n):
        cp_vis[i] = compute_cp_visibility(
            flat_feats[i - 1], raw.placed[i - 1].primitive.type,
            flat_feats[i],     raw.placed[i].primitive.type,
            sigma,
        )

    for i in range(n):
        left_cp = cp_vis[i] if i > 0 else None
        right_cp = cp_vis[i + 1] if i + 1 < n else None
        if left_cp is None and right_cp is None:
            flat_feats[i]["cp_visibility"] = 1.0
        elif left_cp is None:
            flat_feats[i]["cp_visibility"] = right_cp
        elif right_cp is None:
            flat_feats[i]["cp_visibility"] = left_cp
        else:
            flat_feats[i]["cp_visibility"] = min(left_cp, right_cp)

    globals_ = compute_globals(z_signal, all_p2ps)
    return NormedFeatures(in_event=in_event, inter_event=inter_event, globals=globals_)
