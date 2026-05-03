from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data_gen.config import Config
from data_gen.layout import Layout
from data_gen.orchestrate import validate_slope
from data_gen.realize import REALIZERS
from data_gen.sampling import sample_uniform_categorical
from data_gen.schema import FeatureSpec, Schema
from data_gen.types import Event, Primitive, PrimitiveType


@dataclass
class JitterFallback:
    """Records a position where slope retries were exhausted and we fell back."""

    event_index: int
    template_index: int
    attempts: int


@dataclass
class JitterReport:
    """Diagnostics for the step-17 jitter pass."""

    fallbacks: list[JitterFallback] = field(default_factory=list)


def jitter_random(
    random_dict: dict[str, Any],
    duration: int,
    ptype: PrimitiveType,
    schema: Schema,
    cfg: Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Per-feature jitter on a primitive's `random` dict. Magnitude scales with
    `cfg.jitter_randomness`. Returns a NEW dict; does not mutate input.
    """
    out: dict[str, Any] = {}
    feats = schema.features(ptype)
    for fname, fval in random_dict.items():
        spec = feats.get(fname)
        if spec is None or spec.scope is None:
            out[fname] = fval
            continue
        out[fname] = _jitter_one(fval, spec, cfg.jitter_randomness, rng)
    if ptype == PrimitiveType.SPIKE and "rise_duration" in out:
        out["rise_duration"] = int(np.clip(int(out["rise_duration"]), 1, duration - 1))
    elif ptype == PrimitiveType.DIP and "fall_duration" in out:
        out["fall_duration"] = int(np.clip(int(out["fall_duration"]), 1, duration - 1))
    return out


def _jitter_one(
    value: Any, spec: FeatureSpec, magnitude: float, rng: np.random.Generator
) -> Any:
    if magnitude <= 0:
        return value
    if spec.type == "categorical":
        if rng.random() < magnitude:
            return sample_uniform_categorical(list(spec.scope), rng)
        return value
    low = float(spec.scope[0])
    high = float(spec.scope[1])
    width = max(high - low, 1e-9)
    sigma = magnitude * width
    if spec.type == "integer":
        sigma = max(sigma, 1.0)
        new = int(round(float(value) + rng.normal() * sigma))
        return int(np.clip(new, int(low), int(high)))
    new = float(value) + rng.normal() * sigma
    return float(np.clip(new, low, high))


def realize_with_random(
    template_prim: Primitive, jittered_random: dict[str, Any]
) -> Primitive:
    """
    Re-realize a primitive with the same `definitive` triple but a (possibly
    jittered) random dict. Bypasses `generate()` so we can inject specific
    random values.
    """
    realizer = REALIZERS[template_prim.type]
    new_values = realizer(**template_prim.definitive, **jittered_random)
    return Primitive(
        type=template_prim.type,
        values=new_values,
        definitive=dict(template_prim.definitive),
        random=dict(jittered_random),
    )


def jitter_event_occurrence(
    template: Event,
    schema: Schema,
    cfg: Config,
    rng: np.random.Generator,
    *,
    boundary_prev: Primitive | None,
    boundary_next: Primitive | None,
    max_attempts: int = 200,
    event_index: int = 0,
    fallbacks: list[JitterFallback] | None = None,
) -> Event:
    """
    Build one event occurrence by jittering each template primitive's random
    dict, re-realizing, and validating the slope rule against in-event
    neighbours plus the supplied boundary_prev/next (the inter-event
    primitives flanking the event).

    Plateaus pass through unchanged (no random features). On retry exhaustion,
    falls back to the template's primitive at that position and records the
    fallback in `fallbacks`.
    """
    primitives: list[Primitive] = []
    template_prims = template.primitives
    for pos, tprim in enumerate(template_prims):
        prev = primitives[-1] if pos > 0 else boundary_prev
        next_ = template_prims[pos + 1] if pos + 1 < len(template_prims) else boundary_next

        if tprim.type == PrimitiveType.PLATEAU or not tprim.random:
            primitives.append(
                realize_with_random(tprim, dict(tprim.random))
            )
            continue

        chosen: Primitive | None = None
        attempts_used = 0
        for attempt in range(max_attempts):
            attempts_used = attempt + 1
            jittered = jitter_random(
                tprim.random, int(tprim.definitive["duration"]),
                tprim.type, schema, cfg, rng,
            )
            try:
                cand = realize_with_random(tprim, jittered)
            except Exception:
                continue
            if validate_slope(prev, cand, next_):
                chosen = cand
                break
        if chosen is None:
            if fallbacks is not None:
                fallbacks.append(
                    JitterFallback(
                        event_index=event_index,
                        template_index=pos,
                        attempts=attempts_used,
                    )
                )
            chosen = realize_with_random(tprim, dict(tprim.random))
        primitives.append(chosen)
    return Event(primitives=primitives)


def build_jittered_occurrences(
    cfg: Config,
    schema: Schema,
    layout: Layout,
    event_template: Event,
    inter_event: list[list[Primitive]],
    rng: np.random.Generator,
    *,
    max_attempts: int = 200,
) -> tuple[list[Event], JitterReport]:
    """
    Step 17: produce N event occurrences from the shared template by jittering
    each occurrence's random features. Inter-event primitives are unchanged.
    """
    nb_events = layout.nb_events
    occurrences: list[Event] = []
    report = JitterReport()

    for k in range(nb_events):
        # Determine boundary neighbours.
        # Left: last primitive of inter_event[k] if non-empty, else last primitive of the
        # previously-built occurrence (or None for the very first event).
        if inter_event[k]:
            left = inter_event[k][-1]
        elif occurrences:
            left = occurrences[-1].primitives[-1] if occurrences[-1].primitives else None
        else:
            left = None
        # Right: first primitive of inter_event[k+1] if non-empty, else first primitive of
        # the template (next occurrence will start with the same shape).
        if inter_event[k + 1]:
            right = inter_event[k + 1][0]
        elif event_template.primitives:
            right = event_template.primitives[0]
        else:
            right = None

        occ = jitter_event_occurrence(
            event_template,
            schema,
            cfg,
            rng,
            boundary_prev=left,
            boundary_next=right,
            max_attempts=max_attempts,
            event_index=k,
            fallbacks=report.fallbacks,
        )
        occurrences.append(occ)
    return occurrences, report


def assemble_with_occurrences(
    layout: Layout,
    occurrences: list[Event],
    inter_event: list[list[Primitive]],
) -> list[Primitive | Event]:
    """N possibly-different Events interleaved with inter_event slots."""
    items: list[Primitive | Event] = []
    for k, slot in enumerate(inter_event):
        items.extend(slot)
        if k < layout.nb_events:
            items.append(occurrences[k])
    return items
