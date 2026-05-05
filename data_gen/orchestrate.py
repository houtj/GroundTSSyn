from __future__ import annotations

import numpy as np

from data_gen.config import Config
from data_gen.layout import Layout
from data_gen.realize import generate
from data_gen.sampling import sample_uniform_continuous
from data_gen.schema import Schema
from data_gen.types import Event, Primitive, PrimitiveType


class RealizationError(RuntimeError):
    """Raised when slope-validation re-sampling exhausts max_attempts."""


def _slope_pair(prim: Primitive) -> tuple[float, float]:
    if prim.type == PrimitiveType.PLATEAU:
        return 0.0, 0.0
    v = prim.values
    return v[1] - v[0], v[-1] - v[-2]


def _junction_interior_angle(s1: float, s2: float) -> float:
    """Interior angle at the junction between two slope directions (degrees).

    Treats each slope as the 2-D direction (1, s) in time-value space.
    180° = smooth continuation (same slope), 0° = sharp reversal.
    """
    cos_theta = -(1.0 + s1 * s2) / (
        np.sqrt(1.0 + s1 * s1) * np.sqrt(1.0 + s2 * s2)
    )
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def _junction_angle_ok(
    prev: Primitive | None,
    current: Primitive,
    next_: Primitive | None,
    max_angle_deg: float,
) -> bool:
    """Return False when a non-plateau junction has interior angle >= max_angle_deg.

    Plateau neighbors are skipped — a flat-to-slope transition is always visible.
    """
    if current.type == PrimitiveType.PLATEAU:
        return True
    cur_start, cur_end = _slope_pair(current)
    if prev is not None and prev.type != PrimitiveType.PLATEAU:
        _, prev_end = _slope_pair(prev)
        if _junction_interior_angle(prev_end, cur_start) >= max_angle_deg:
            return False
    if next_ is not None and next_.type != PrimitiveType.PLATEAU:
        next_start, _ = _slope_pair(next_)
        if _junction_interior_angle(cur_end, next_start) >= max_angle_deg:
            return False
    return True


def validate_slope(
    prev: Primitive | None, current: Primitive, next_: Primitive | None
) -> bool:
    """
    Strict-inequality slope check at boundaries. If the current primitive is
    a plateau, skip — we can't re-sample it (no random features). The
    non-plateau neighbors will have non-zero slope almost surely.
    """
    if current.type == PrimitiveType.PLATEAU:
        return True
    cur_start, cur_end = _slope_pair(current)
    if prev is not None:
        _, prev_end = _slope_pair(prev)
        if cur_start == prev_end:
            return False
    if next_ is not None:
        next_start, _ = _slope_pair(next_)
        if cur_end == next_start:
            return False
    return True


def _build_definitive(
    t: PrimitiveType, peak2peak: float, duration: int
) -> dict[str, float | int]:
    base: dict[str, float | int] = {"start_value": 0.0, "duration": duration}
    if t != PrimitiveType.PLATEAU:
        base["peak2peak"] = peak2peak
    return base


def _sample_peak2peak(cfg: Config, in_event: bool, rng: np.random.Generator) -> float:
    low, high = cfg.peak2peak_range(in_event)
    return sample_uniform_continuous(low, high, rng)


def _realize_with_retry(
    t: PrimitiveType,
    in_event: bool,
    duration: int,
    cfg: Config,
    schema: Schema,
    rng: np.random.Generator,
    prev: Primitive | None,
    next_: Primitive | None,
    *,
    max_attempts: int,
) -> Primitive:
    """
    Repeatedly call `generate()` until the slope check passes. plateaus
    short-circuit — they have no random features, so the first realization
    is the only realization.
    """
    if t == PrimitiveType.PLATEAU:
        peak2peak = _sample_peak2peak(cfg, in_event, rng)
        definitive = _build_definitive(t, peak2peak, duration)
        return generate(t, definitive, schema, rng)

    for _ in range(max_attempts):
        peak2peak = _sample_peak2peak(cfg, in_event, rng)
        definitive = _build_definitive(t, peak2peak, duration)
        prim = generate(t, definitive, schema, rng)
        if validate_slope(prev, prim, next_) and _junction_angle_ok(
            prev, prim, next_, cfg.max_junction_angle_deg
        ):
            return prim
    raise RealizationError(
        f"slope validation failed for {t.value} after {max_attempts} attempts"
    )


def _realize_event_template_drift_limited(
    cfg: Config,
    layout: Layout,
    schema: Schema,
    rng: np.random.Generator,
) -> Event:
    """
    Drift-constrained event template realization (used when cfg.drift_limit).

    Realize spike/dip/oscillation/plateau in their positions first, then
    realize increase/decrease in random order; the last inc/dec's peak2peak
    is solved so total event drift falls in [-drift_threshold, +drift_threshold].
    Slope validation is dropped in this flow since realization is no longer
    in temporal order and the forced last position cannot be re-rolled.
    """
    types = layout.event_template_types
    durations = layout.event_template_durations
    n = len(types)
    primitives: list[Primitive | None] = [None] * n

    fixed_pos = [
        i for i, t in enumerate(types)
        if t not in (PrimitiveType.INCREASE, PrimitiveType.DECREASE)
    ]
    var_pos = [
        i for i, t in enumerate(types)
        if t in (PrimitiveType.INCREASE, PrimitiveType.DECREASE)
    ]

    for i in fixed_pos:
        p2p = _sample_peak2peak(cfg, in_event=True, rng=rng)
        definitive = _build_definitive(types[i], p2p, durations[i])
        primitives[i] = generate(types[i], definitive, schema, rng)

    d_other = sum(
        primitives[i].values[-1] - primitives[i].values[0] for i in fixed_pos
    )

    var_order = list(var_pos)
    rng.shuffle(var_order)

    d_partial = 0.0
    for i in var_order[:-1]:
        p2p = _sample_peak2peak(cfg, in_event=True, rng=rng)
        definitive = _build_definitive(types[i], p2p, durations[i])
        primitives[i] = generate(types[i], definitive, schema, rng)
        d_partial += primitives[i].values[-1] - primitives[i].values[0]

    if var_order:
        last = var_order[-1]
        D = d_other + d_partial
        thres = cfg.drift_threshold

        chosen_type = types[last]
        sign = 1.0 if chosen_type == PrimitiveType.INCREASE else -1.0
        low = (-thres - D) * sign
        high = (thres - D) * sign
        if low > high:
            low, high = high, low

        if high < 0.0:
            # No direction-preserving peak2peak exists; flip type so the
            # contribution sign reverses and the feasible window is positive.
            chosen_type = (
                PrimitiveType.DECREASE
                if chosen_type == PrimitiveType.INCREASE
                else PrimitiveType.INCREASE
            )
            sign = -sign
            low = (-thres - D) * sign
            high = (thres - D) * sign
            if low > high:
                low, high = high, low
            types[last] = chosen_type

        v_last = float(rng.uniform(max(low, 0.0), high))
        definitive = _build_definitive(chosen_type, v_last, durations[last])
        primitives[last] = generate(chosen_type, definitive, schema, rng)

    return Event(primitives=[p for p in primitives if p is not None])


def realize_event_template(
    cfg: Config,
    layout: Layout,
    schema: Schema,
    rng: np.random.Generator,
    *,
    max_attempts: int = 200,
    template_max_attempts: int = 50,
) -> Event:
    """
    Step 9 (event template). Realize each in-event primitive sequentially,
    validating start_slope against the previous template primitive's
    end_slope. After the loop, also enforce that the template loops cleanly:
    template[-1].end_slope != template[0].start_slope (because k=0
    inter-event slots cause two template copies to abut).
    """
    types = layout.event_template_types
    durations = layout.event_template_durations
    if not types:
        return Event(primitives=[])

    if cfg.drift_limit:
        return _realize_event_template_drift_limited(cfg, layout, schema, rng)

    for _ in range(template_max_attempts):
        primitives: list[Primitive] = []
        for i, (t, d) in enumerate(zip(types, durations)):
            prev = primitives[-1] if i > 0 else None
            prim = _realize_with_retry(
                t,
                in_event=True,
                duration=d,
                cfg=cfg,
                schema=schema,
                rng=rng,
                prev=prev,
                next_=None,
                max_attempts=max_attempts,
            )
            primitives.append(prim)

        # template self-loop check (handles k=0 slots that abut event copies)
        if primitives[0].type == PrimitiveType.PLATEAU or primitives[-1].type == PrimitiveType.PLATEAU:
            return Event(primitives=primitives)
        first_start, _ = _slope_pair(primitives[0])
        _, last_end = _slope_pair(primitives[-1])
        if first_start == last_end:
            continue
        if _junction_interior_angle(last_end, first_start) >= cfg.max_junction_angle_deg:
            continue
        return Event(primitives=primitives)

    raise RealizationError(
        f"event template self-loop check failed after {template_max_attempts} attempts"
    )


def realize_inter_event_primitives(
    cfg: Config,
    layout: Layout,
    event_template: Event,
    schema: Schema,
    rng: np.random.Generator,
    *,
    max_attempts: int = 200,
) -> list[list[Primitive]]:
    """
    Step 9 (inter-event). Realize inter-event primitives slot-by-slot in
    temporal order. Two-sided slope check fires for the LAST primitive of
    any non-trailing slot (k < N), against the event template's first
    primitive.
    """
    nb_events = layout.nb_events
    template_first = event_template.primitives[0] if event_template.primitives else None
    template_last = event_template.primitives[-1] if event_template.primitives else None

    out: list[list[Primitive]] = []
    for k, slot_types in enumerate(layout.inter_event_types):
        slot_durations = layout.inter_event_durations[k]
        slot: list[Primitive] = []
        for p, (t, d) in enumerate(zip(slot_types, slot_durations)):
            # left neighbor
            if p > 0:
                left = slot[-1]
            elif k > 0:
                left = template_last
            else:
                left = None
            # right neighbor (two-sided): only when last in non-trailing slot
            right = template_first if (p == len(slot_types) - 1 and k < nb_events) else None
            prim = _realize_with_retry(
                t,
                in_event=False,
                duration=d,
                cfg=cfg,
                schema=schema,
                rng=rng,
                prev=left,
                next_=right,
                max_attempts=max_attempts,
            )
            slot.append(prim)
        out.append(slot)
    return out


def assemble_layout(
    layout: Layout,
    event_template: Event,
    inter_event: list[list[Primitive]],
) -> list[Primitive | Event]:
    """
    Build the temporal-order layout to feed `concatenate()`:
    [*inter[0], Event(template), *inter[1], Event(template), …,
     Event(template), *inter[N]]
    """
    items: list[Primitive | Event] = []
    for k, slot in enumerate(inter_event):
        items.extend(slot)
        if k < layout.nb_events:
            items.append(event_template)
    return items
