from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from data_gen.config import Config
from data_gen.sampling import (
    sample_linear_integer,
    sample_skewed_normal_integer,
    sample_skewed_normal_integer_around,
    sample_uniform_integer,
    sample_weighted_categorical,
)
from data_gen.types import PrimitiveType


class LayoutInfeasible(RuntimeError):
    """Raised when the budget arithmetic cannot satisfy per-primitive minima."""


# Adjacent type pairs forbidden during sequence sampling. Re-rolled at the
# sample loop. plateau↔plateau merges into a longer flat; spike↔oscillation
# is too tangled to model in change-point visibility cleanly.
DISALLOWED_ADJACENCY: set[tuple[PrimitiveType, PrimitiveType]] = {
    (PrimitiveType.PLATEAU,     PrimitiveType.PLATEAU),
    (PrimitiveType.SPIKE,       PrimitiveType.OSCILLATION),
    (PrimitiveType.OSCILLATION, PrimitiveType.SPIKE),
}


def _disallowed(left: PrimitiveType | None, right: PrimitiveType | None) -> bool:
    if left is None or right is None:
        return False
    return (left, right) in DISALLOWED_ADJACENCY


class Layout(BaseModel):
    nb_events: int
    event_template_types: list[PrimitiveType]
    event_template_durations: list[int]
    event_length: int
    inter_event_counts: list[int]
    inter_event_types: list[list[PrimitiveType]]
    inter_event_durations: list[list[int]]
    signal_length: int
    # Periodical mode: number of leading samples trimmed from the full M-event
    # tile to fit `signal_length`. Always 0 in aperiodic mode. The trimmed
    # samples on the right side equal full_tile_len - signal_length - tile_offset.
    tile_offset: int = 0
    periodical: bool = False


def _per_type_bounds(
    t: PrimitiveType, cfg: Config
) -> tuple[int, int | None]:
    """Static per-type duration bounds: (low, high). high=None means unbounded."""
    if t in (PrimitiveType.SPIKE, PrimitiveType.DIP):
        return cfg.min_spike_length, cfg.max_spike_length
    return cfg.min_primitive_length, None


def sample_primitive_types(
    cfg: Config,
    count: int,
    rng: np.random.Generator,
    *,
    prev_type: PrimitiveType | None = None,
    next_type: PrimitiveType | None = None,
) -> list[PrimitiveType]:
    """Weighted-categorical sampling with disallowed-adjacency suppression via resampling.

    prev_type: type immediately before this sequence (e.g. last event-template
               primitive for non-leading inter-event slots).
    next_type: type immediately after this sequence (e.g. first event-template
               primitive for non-trailing inter-event slots).
    """
    types, weights = cfg.primitive_weights()
    out: list[PrimitiveType] = []
    for i in range(count):
        is_last = i == count - 1
        eff_prev = prev_type if i == 0 else out[-1]
        while True:
            t = sample_weighted_categorical(types, weights, rng)
            if _disallowed(eff_prev, t):
                continue
            if is_last and _disallowed(t, next_type):
                continue
            break
        out.append(t)
    return out


def _sample_sequential_durations(
    types: list[PrimitiveType],
    total_budget: int,
    cfg: Config,
    rng: np.random.Generator,
) -> list[int]:
    """
    Sequentially sample durations summing exactly to `total_budget`.
    Per-step bounds account for the per-type min and the need to leave
    `min_primitive_length` for each remaining primitive.
    """
    n = len(types)
    if n == 0:
        if total_budget != 0:
            raise LayoutInfeasible(
                f"empty type list but non-zero budget {total_budget}"
            )
        return []

    # Quick infeasibility check on the total min budget.
    sum_min = sum(_per_type_bounds(t, cfg)[0] for t in types)
    if sum_min > total_budget:
        raise LayoutInfeasible(
            f"sum of per-type minima {sum_min} exceeds budget {total_budget}"
        )

    durations: list[int] = []
    remaining = total_budget
    for i, t in enumerate(types):
        n_remaining = n - i
        per_type_low, per_type_high = _per_type_bounds(t, cfg)
        # leave min_primitive_length per remaining-after-this primitive
        future_min = (n_remaining - 1) * cfg.min_primitive_length
        high = remaining - future_min
        if per_type_high is not None:
            high = min(high, per_type_high)
        low = per_type_low

        if i == n - 1:
            # last primitive: take the remainder, validate against bounds
            d = remaining
            if d < low or (per_type_high is not None and d > per_type_high):
                raise LayoutInfeasible(
                    f"last primitive {t.value} needs {d} but bounds are "
                    f"[{low}, {per_type_high}]"
                )
        else:
            if high < low:
                raise LayoutInfeasible(
                    f"primitive {i} ({t.value}): high {high} < low {low}"
                )
            center = remaining / n_remaining
            d = sample_skewed_normal_integer_around(
                low,
                int(high),
                center,
                cfg.primitive_average_length_weight,
                cfg.primitive_extreme_length_weight,
                rng,
            )
        durations.append(d)
        remaining -= d

    if remaining != 0:
        raise LayoutInfeasible(f"residual budget {remaining} after sampling")
    return durations


def _plan_once(cfg: Config, rng: np.random.Generator) -> Layout:
    # Step 1: number of event occurrences
    nb_events = sample_uniform_integer(
        cfg.nb_event_occurrence[0], cfg.nb_event_occurrence[1], rng
    )

    # Step 2: number of inter-event primitives per slot (length = N+1)
    inter_event_counts = [
        sample_linear_integer(
            cfg.nb_inter_event_primitives[0],
            cfg.nb_inter_event_primitives[1],
            cfg.min_nb_weight,
            cfg.max_nb_weight,
            rng,
        )
        for _ in range(nb_events + 1)
    ]

    # Step 3: number of in-event primitives
    nb_in_event = sample_uniform_integer(
        cfg.nb_primitives_in_event[0], cfg.nb_primitives_in_event[1], rng
    )

    # Step 4: in-event primitive types
    event_template_types = sample_primitive_types(cfg, nb_in_event, rng)

    # Step 5: inter-event primitive types per slot
    # Pass boundary types so the first/last primitive of each slot cannot form
    # an adjacent plateau pair with the event template primitive it abuts.
    inter_event_types = []
    for k, count in enumerate(inter_event_counts):
        p = event_template_types[-1] if k > 0 and event_template_types else None
        n = event_template_types[0] if k < nb_events and event_template_types else None
        inter_event_types.append(sample_primitive_types(cfg, count, rng, prev_type=p, next_type=n))

    # Step 6: event length (corrected formula)
    sum_inter_min = sum(inter_event_counts) * cfg.min_primitive_length
    min_event_length = nb_in_event * cfg.min_primitive_length
    if nb_events <= 0:
        raise LayoutInfeasible(f"nb_events must be >= 1, got {nb_events}")
    max_event_length = (cfg.signal_length - 1 - sum_inter_min) // nb_events
    if max_event_length < min_event_length:
        raise LayoutInfeasible(
            f"max_event_length {max_event_length} < min_event_length "
            f"{min_event_length} (signal_length={cfg.signal_length}, "
            f"nb_events={nb_events}, sum_inter_min={sum_inter_min})"
        )
    event_length = sample_skewed_normal_integer(
        min_event_length,
        max_event_length,
        cfg.event_average_length_weight,
        cfg.event_extreme_length_weight,
        rng,
    )

    # Step 7: in-event primitive durations
    event_template_durations = _sample_sequential_durations(
        event_template_types, event_length, cfg, rng
    )

    # Step 8: inter-event primitive durations (corrected formula)
    inter_budget = cfg.signal_length - 1 - nb_events * event_length
    flat_types = [t for slot in inter_event_types for t in slot]
    flat_durations = _sample_sequential_durations(flat_types, inter_budget, cfg, rng)
    # split back into slot-shaped lists
    inter_event_durations: list[list[int]] = []
    cursor = 0
    for k in inter_event_counts:
        inter_event_durations.append(flat_durations[cursor : cursor + k])
        cursor += k

    return Layout(
        nb_events=nb_events,
        event_template_types=event_template_types,
        event_template_durations=event_template_durations,
        event_length=event_length,
        inter_event_counts=inter_event_counts,
        inter_event_types=inter_event_types,
        inter_event_durations=inter_event_durations,
        signal_length=cfg.signal_length,
    )


def _periodical_event_length_bounds(
    n_target: int, signal_length: int, min_event_length: int
) -> tuple[int, int]:
    """L range for periodical mode.

    n_target is the sampled "intended full event count". Per the spec:
        L ∈ [signal_length / (n+2), signal_length / (n-1)]
    For n_target == 1 the upper bound diverges; fall back to [S/3, S-1].
    The lower bound is also clamped to `min_event_length` so the per-primitive
    minima can always be satisfied.
    """
    S = signal_length
    if n_target <= 1:
        low = S // 3
        high = S - 1
    else:
        low = S // (n_target + 2)
        high = S // (n_target - 1)
    low = max(low, min_event_length, 1)
    high = max(high, low)
    return int(low), int(high)


def _periodical_template_can_be_drift_neutral(
    types: list[PrimitiveType],
) -> bool:
    """A periodical event template must be drift-neutral: end_value of the
    event must equal start_value, otherwise tiled events accumulate drift
    across the signal (and the jitter step would have to produce inverted
    inc/dec primitives to compensate, breaking type↔shape consistency).

    Among the primitive types, only INCREASE and DECREASE contribute non-zero
    drift (peak2peak ≥ 0 always; spike/dip/oscillation/plateau all return to
    their start_value). So zero net drift is achievable iff:
      - the template has no inc/dec, OR
      - it has at least one of each so positive and negative contributions
        can be balanced by the drift_limit logic.
    A template with only-increase or only-decrease cannot balance.
    """
    has_inc = any(t == PrimitiveType.INCREASE for t in types)
    has_dec = any(t == PrimitiveType.DECREASE for t in types)
    return has_inc == has_dec


def _plan_once_periodical(cfg: Config, rng: np.random.Generator) -> Layout:
    # Step 1: target full-event count
    n_target = sample_uniform_integer(
        cfg.nb_event_occurrence[0], cfg.nb_event_occurrence[1], rng
    )

    # Step 3: number of in-event primitives
    nb_in_event = sample_uniform_integer(
        cfg.nb_primitives_in_event[0], cfg.nb_primitives_in_event[1], rng
    )

    # Step 4: in-event primitive types
    event_template_types = sample_primitive_types(cfg, nb_in_event, rng)

    if cfg.drift_limit and not _periodical_template_can_be_drift_neutral(
        event_template_types
    ):
        raise LayoutInfeasible(
            f"periodical template {[t.value for t in event_template_types]} "
            f"cannot achieve zero net drift (only inc or only dec): would "
            f"force per-event drift to accumulate across tiled occurrences"
        )

    # Step 6: event length, sampled uniformly in the periodical bound formula.
    min_event_length = nb_in_event * cfg.min_primitive_length
    L_low, L_high = _periodical_event_length_bounds(
        n_target, cfg.signal_length, min_event_length
    )
    if L_high < min_event_length:
        raise LayoutInfeasible(
            f"periodical L_high={L_high} < min_event_length={min_event_length} "
            f"(signal_length={cfg.signal_length}, n_target={n_target}, "
            f"nb_in_event={nb_in_event})"
        )
    event_length = sample_uniform_integer(L_low, L_high, rng)

    # Step 7: in-event primitive durations
    event_template_durations = _sample_sequential_durations(
        event_template_types, event_length, cfg, rng
    )

    # Determine actual number of tiled events M so the full tile covers the
    # signal. With M events of duration L tiled back-to-back, the full tile
    # has M*L+1 samples; we need M*L >= signal_length - 1.
    full_min = cfg.signal_length - 1
    M = (full_min + event_length - 1) // event_length  # ceil
    if M < 1:
        M = 1
    full_tile_len = M * event_length + 1
    overshoot = full_tile_len - cfg.signal_length

    # Random integer offset; allows one edge to be a full event and the other
    # truncated, or both truncated.
    tile_offset = int(rng.integers(0, overshoot + 1)) if overshoot > 0 else 0

    # Step 2/5/8: no inter-event primitives in periodical mode.
    inter_event_counts = [0] * (M + 1)
    inter_event_types: list[list[PrimitiveType]] = [[] for _ in range(M + 1)]
    inter_event_durations: list[list[int]] = [[] for _ in range(M + 1)]

    return Layout(
        nb_events=M,
        event_template_types=event_template_types,
        event_template_durations=event_template_durations,
        event_length=event_length,
        inter_event_counts=inter_event_counts,
        inter_event_types=inter_event_types,
        inter_event_durations=inter_event_durations,
        signal_length=cfg.signal_length,
        tile_offset=tile_offset,
        periodical=True,
    )


def plan_layout(
    cfg: Config, rng: np.random.Generator, *, max_attempts: int = 10
) -> Layout:
    """Steps 1-8. Retries on `LayoutInfeasible` up to `max_attempts` times."""
    plan_fn = _plan_once_periodical if cfg.periodical else _plan_once
    last_err: Exception | None = None
    for _ in range(max_attempts):
        try:
            return plan_fn(cfg, rng)
        except LayoutInfeasible as e:
            last_err = e
            continue
    raise LayoutInfeasible(
        f"plan_layout failed after {max_attempts} attempts; last error: {last_err}"
    )
