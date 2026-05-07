"""Pre-compute merge / split / reinterpret candidates from visibility metadata.

These shortlists are *guidelines* for the variant agent, not directives. The
agent gets to override them based on the visualization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_gen.features import extract
from data_gen.types import PrimitiveType

from .config import CandidatesConfig


def _first_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    return events[0]


def merge_candidates(
    events: list[dict[str, Any]],
    cfg: CandidatesConfig,
) -> list[dict[str, Any]]:
    """Adjacent in-event positions whose boundary cp_visibility is below threshold.

    Reads from the first event's primitives (cp_visibility is level-independent
    so it's representative). Returns a list of:
        {target_template_indices: [k, k+1], left_type, right_type,
         boundary_cp_visibility, left_visibility, right_visibility}
    """
    ev = _first_event(events)
    if ev is None:
        return []
    prims = ev.get("primitives", [])
    out: list[dict[str, Any]] = []
    for i in range(len(prims) - 1):
        right = prims[i + 1]
        cp = right.get("cp_visibility")
        if not isinstance(cp, (int, float)):
            continue
        if cp < cfg.merge_cp_max:
            out.append(
                {
                    "target_template_indices": [
                        prims[i].get("template_index"),
                        right.get("template_index"),
                    ],
                    "left_type": prims[i].get("type"),
                    "right_type": right.get("type"),
                    "boundary_cp_visibility": float(cp),
                    "left_visibility": prims[i].get("visibility"),
                    "right_visibility": right.get("visibility"),
                }
            )
    return out


def split_candidates(
    events: list[dict[str, Any]],
    cfg: CandidatesConfig,
) -> list[dict[str, Any]]:
    """Spike or dip whose change-point on a side is below threshold.

    Indicates the spike/dip blends visually with a neighbor and could be
    described as e.g. increase+decrease (spike) or decrease+increase (dip).
    """
    ev = _first_event(events)
    if ev is None:
        return []
    prims = ev.get("primitives", [])
    out: list[dict[str, Any]] = []
    for i, p in enumerate(prims):
        ptype = p.get("type")
        if ptype not in ("spike", "dip"):
            continue
        left_cp = prims[i].get("cp_visibility")  # this is the left boundary cp
        right_cp = prims[i + 1].get("cp_visibility") if i + 1 < len(prims) else None
        low_left = isinstance(left_cp, (int, float)) and left_cp < cfg.split_cp_max
        low_right = isinstance(right_cp, (int, float)) and right_cp < cfg.split_cp_max
        if low_left or low_right:
            strategy = "increase+decrease" if ptype == "spike" else "decrease+increase"
            out.append(
                {
                    "target_template_indices": [p.get("template_index")],
                    "type": ptype,
                    "split_strategy": strategy,
                    "left_cp_visibility": left_cp,
                    "right_cp_visibility": right_cp,
                    "visibility": p.get("visibility"),
                }
            )
    return out


_ALT_TYPE_MAP: dict[str, list[str]] = {
    # Common perceptual reinterpretations under noise:
    "increase": ["plateau"],            # very flat increase reads as plateau
    "decrease": ["plateau"],
    "plateau": [],                      # plateau is the "least informative" reinterpretation
    "spike": ["increase", "decrease"],  # very small spike could read as a step
    "dip": ["decrease", "increase"],
    "oscillation": ["plateau"],         # tiny oscillation reads as flat
}


def reinterpret_candidates(
    events: list[dict[str, Any]],
    signal_noisy: list[float],
    cfg: CandidatesConfig,
) -> list[dict[str, Any]]:
    """Primitives whose `visibility` is low; propose plausible alt-types.

    For each candidate alt-type, re-extracts features on the noisy span via
    `data_gen.features.extract` so the variant agent can see whether the
    alternative reading is feature-consistent.
    """
    ev = _first_event(events)
    if ev is None:
        return []
    prims = ev.get("primitives", [])
    out: list[dict[str, Any]] = []
    for p in prims:
        vis = p.get("visibility")
        if not isinstance(vis, (int, float)):
            continue
        if vis >= cfg.reinterpret_vis_max:
            continue
        ptype = p.get("type")
        alts = _ALT_TYPE_MAP.get(ptype, [])
        if not alts:
            continue
        span = p.get("span", [0, 0])
        s, e = int(span[0]), int(span[1])
        if e <= s:
            continue
        slice_vals = signal_noisy[s : e + 1]
        alt_summaries = []
        for alt in alts:
            try:
                extracted = extract(slice_vals, PrimitiveType(alt))
                alt_summaries.append({"alt_type": alt, "features": _to_jsonable(extracted)})
            except Exception:
                continue
        if not alt_summaries:
            continue
        out.append(
            {
                "target_template_indices": [p.get("template_index")],
                "current_type": ptype,
                "current_visibility": float(vis),
                "alternatives": alt_summaries,
            }
        )
    return out


def _to_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out
