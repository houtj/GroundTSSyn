"""Deterministic verification & string-assembly helpers.

Three jobs:
  1. Assemble the canonical `primitive[: c, c]; primitive[: c, c]` string from
     a list of (primitive_phrase, constraint_phrases) chunks, AND record the
     character span of each chunk inside that string.
  2. Verify a single constraint against a freshly-extracted feature dict on
     a signal slice (used to gate variant chunks before sending to the critic).
  3. Verify a negative description's underlying (types, constraints) pattern
     does not match anywhere in the signal — reuses the semantics of
     `data_gen.negation._matches_anywhere`, but rebuilds the placed-primitive
     sequence from `whole_time_series_layout` so we don't need the live
     ConcatResult object.
"""

from __future__ import annotations

import re
from typing import Any

from data_gen.constraints import (
    ComparableConstraint,
    Constraint,
    NumericalConstraint,
    SelfConstraint,
    evaluate_constraint,
)
from data_gen.features import extract
from data_gen.types import PrimitiveType


# ---------------------------------------------------------------------------
# String assembly + grounding spans
# ---------------------------------------------------------------------------


def assemble_string_from_chunks(
    chunks: list[tuple[str, list[str]]],
) -> tuple[str, list[tuple[int, int]]]:
    """Join chunks into a single string and return per-chunk char spans.

    Each chunk is (primitive_phrase, [constraint_phrase, ...]).
    Output format:
      * with constraints:    `primitive: c, c`
      * without constraints: `primitive`            (no trailing colon)
    Chunks are joined with `; `.
    Returns (full_text, [(char_start, char_end) for each chunk]).
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, (prim, cons) in enumerate(chunks):
        prim = prim.strip()
        cons_clean = [c.strip() for c in cons if c and c.strip()]
        if cons_clean:
            chunk = f"{prim}: {', '.join(cons_clean)}"
        else:
            chunk = prim
        if i > 0:
            sep = "; "
            parts.append(sep)
            cursor += len(sep)
        chunk_start = cursor
        parts.append(chunk)
        cursor += len(chunk)
        chunk_end = cursor
        spans.append((chunk_start, chunk_end))
    return "".join(parts), spans


# A chunk is either `primitive` (no constraints) or `primitive: c, c`.
# Chunks are separated by `;` (with optional whitespace), trailing `;` allowed.
_CHUNK_RE = r"[^:;]+(:[^;]*)?"
_FORMAT_RE = re.compile(rf"^{_CHUNK_RE}(;\s*{_CHUNK_RE})*;?$")


def matches_format(text: str) -> bool:
    return bool(_FORMAT_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Constraint verification on a slice
# ---------------------------------------------------------------------------


def _to_constraint(raw: dict[str, Any]) -> Constraint | None:
    kind = raw.get("kind")
    try:
        if kind == "numerical":
            return NumericalConstraint.model_validate(raw)
        if kind == "comparable":
            return ComparableConstraint.model_validate(raw)
        if kind == "self":
            return SelfConstraint.model_validate(raw)
    except Exception:
        return None
    return None


def verify_constraint_on_features(
    constraint_dict: dict[str, Any],
    feats_by_pos: list[dict[str, Any]],
    globals_: dict[str, Any],
) -> bool | None:
    c = _to_constraint(constraint_dict)
    if c is None:
        return None
    return evaluate_constraint(c, feats_by_pos, globals_)


def extract_features_for_window(
    type_sequence: list[str],
    span_sequence: list[tuple[int, int]],
    signal: list[float],
) -> list[dict[str, Any]]:
    """Re-extract features for a list of (type, span) — used for negative
    verification and variant constraint re-checking.
    """
    out: list[dict[str, Any]] = []
    for ptype, (s, e) in zip(type_sequence, span_sequence):
        if e <= s:
            out.append({})
            continue
        try:
            feats = extract(signal[s : e + 1], PrimitiveType(ptype))
        except Exception:
            feats = {}
        out.append(feats)
    return out


# ---------------------------------------------------------------------------
# Negative verification — slide pattern over whole_time_series_layout
# ---------------------------------------------------------------------------


def verify_no_match(
    candidate_types: list[str],
    candidate_constraints_by_pos: list[list[dict[str, Any]]],
    layout: list[dict[str, Any]],
    signal: list[float],
    globals_: dict[str, Any],
) -> bool:
    """Return True when no window over `layout` matches both the candidate
    type sequence AND every candidate constraint (strictly).

    Mirrors `data_gen.negation._matches_anywhere` semantics on the JSON layout
    rather than the in-memory ConcatResult.
    """
    M = len(candidate_types)
    if M == 0:
        return False  # vacuous match — treat as not-a-valid-negative
    n = len(layout)
    if n < M:
        return True
    constraint_objs = [
        [_to_constraint(c) for c in lst]
        for lst in candidate_constraints_by_pos
    ]
    for start in range(n - M + 1):
        window = layout[start : start + M]
        if [w.get("type") for w in window] != candidate_types:
            continue
        spans = [(int(w["span"][0]), int(w["span"][1])) for w in window]
        feats_by_pos = extract_features_for_window(candidate_types, spans, signal)
        all_ok = True
        for pos_constraints in constraint_objs:
            for c in pos_constraints:
                if c is None:
                    all_ok = False
                    break
                sat = evaluate_constraint(c, feats_by_pos, globals_)
                if sat is not True:
                    all_ok = False
                    break
            if not all_ok:
                break
        if all_ok:
            return False
    return True
