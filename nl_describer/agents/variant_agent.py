"""Agent 2 — VariantAgent.

Two phases:
  Turn 1 (observe): vision-capable LLM looks at the global plot + a zoomed
    event plot and produces a list of perceptual observations (merge / split
    / reinterpret) that a human might naturally make.
  Turn 2..N (generate): for each sampled subset of observations, the LLM
    produces one VariantOutput. Code resolves spans and constraints
    afterwards.

Observation subset sampling is performed by code (numpy RNG) so variants are
reproducible per seed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from openai import AzureOpenAI

from ..bucketize import feature_summary
from ..config import Config
from ..llm import make_vision_message, structured_call
from ..schemas import (
    BaseChunk,
    Observation,
    ObservationsOutput,
    VariantChunk,
    VariantOutput,
)


_OBSERVE_SYSTEM = """You are a perceptual reviewer of a 1D time-series. You will see two PNGs:
  * global_plot — full noisy signal with primitive boundaries colored by type.
  * event_plot — a zoomed view of one randomly chosen event occurrence.
You also receive JSON with per-primitive `visibility` and `cp_visibility` scores in [0, 1] and pre-computed merge / split / reinterpret candidate shortlists.

Important: the visibility scores are SNR-based proxies. They are guidelines, NOT ground truth. When the plot disagrees with a visibility score, trust the plot.

Your job: list perceptual reinterpretations a human looking at the noisy data might make.
- merge: two adjacent in-event primitives whose boundary is visually invisible could be described as a single primitive (typically the longer / more visible neighbor's type, OR a shape-based rename like plateau+increase -> increase).
- split: a spike or dip whose change-point on a side blends into the neighbor could be described as increase+decrease (spike) or decrease+increase (dip).
- reinterpret: a primitive whose internal shape is barely visible could be described as a different type (e.g. very flat increase -> plateau, increase+decrease that already merged -> spike).

Rules:
- Only propose changes that are visually defensible — do NOT propose changes contradicted by the raw shape.
- Use confidence "high" only if you are visually certain.
- Each observation lists exactly one operation. Multiple observations CAN target overlapping positions, but the orchestrator will pick a non-overlapping subset.
- Return between 0 and 8 observations. Empty is acceptable if nothing is perceptually ambiguous.
"""

_GENERATE_SYSTEM = """You are a time-series annotator producing one variant description by applying a list of perceptual observations to a base description.

You will receive:
  * the base description as a list of (template_index, primitive_phrase, constraint_phrases) chunks (the positive_base produced earlier),
  * a small list of observations to apply (merge / split / reinterpret), each carrying its target_template_indices, new_type, and a short rationale,
  * for each new span produced by the observations, a feature_summary computed by code that you MUST use to phrase quantitative claims.

Your job: emit one ordered list of VariantChunks (in temporal/template order) covering ALL template positions exactly once (with merges collapsing positions and splits expanding them):
  * For positions not touched by any observation, copy the base chunk verbatim into a VariantChunk with op_applied="none", new_type=original_type, derived_from=[that_index].
  * For a merge over positions [k, k+1, ...], emit ONE VariantChunk with derived_from=[k, k+1, ...], op_applied="merge", new_type=<the chosen type>, and constraint_phrases derived from the provided merged feature_summary.
  * For a split of position k into two pieces, emit TWO VariantChunks each with derived_from=[k], op_applied="split", and the appropriate sub_span_fraction (e.g. [0.0, 0.4] then [0.4, 1.0]). The new_type for each piece should match split_strategy.
  * For a reinterpret on position k, emit ONE VariantChunk with derived_from=[k], op_applied="reinterpret", and new_type set to the alternative.

Critical:
- Format is `primitive: c, c; primitive; primitive: c` (the colon and constraint list appear ONLY when there are constraints). Joining is done by code; you only emit phrases. Do NOT include ":" or ";" inside any phrase.
- Wording must be varied (do not just append "(merged)" — fully rewrite the chunk in natural language).
- Operator semantics on constraints must be preserved.
- If a constraint can no longer be expressed for a merged/split/reinterpreted span (e.g. plateau has no slope), OMIT it; do not invent a substitute.
"""


def _length_budget_block(cfg: Config) -> str:
    """Length-budget snippet appended to the generate system prompt.

    Same numbers as the base describer; informs the LLM that the assembled
    variant text must fit a 512-token BERT window.
    """
    return (
        "\n\nLength budget (CRITICAL):\n"
        f"- The full assembled variant description (every primitive_phrase + its constraint_phrases, "
        f"joined as `primitive: c, c; primitive`) must stay under ~{cfg.length_limit.soft_word_budget} words "
        f"and ~{cfg.length_limit.soft_char_budget} characters in total, so it fits a "
        f"{cfg.length_limit.max_bert_tokens}-token BERT input window.\n"
        "- Prefer terse, direct phrasing; avoid decorative synonyms when many constraints stack up.\n"
        "- Keep individual chunks 3-10 words; lean to the lower end when many positions are present.\n"
        "- Numbers, hyphenated terms, and rare words count as multiple tokens. Round numbers and avoid "
        "compound jargon when the same claim can be made plainly."
    )


def _build_observe_payload(
    basic_output: dict[str, Any],
    merge_cands: list[dict[str, Any]],
    split_cands: list[dict[str, Any]],
    reinterpret_cands: list[dict[str, Any]],
) -> str:
    template = basic_output.get("template", {})
    template_prims = template.get("primitives", [])
    template_types = [p.get("type") for p in template_prims]
    events = basic_output.get("events", [])
    first_event = events[0] if events else {}
    prim_summaries = []
    for p in first_event.get("primitives", []):
        prim_summaries.append(
            {
                "template_index": p.get("template_index"),
                "type": p.get("type"),
                "span": p.get("span"),
                "visibility": p.get("visibility"),
                "cp_visibility": p.get("cp_visibility"),
            }
        )
    payload = {
        "template_types": template_types,
        "first_event_primitives": prim_summaries,
        "candidates_guideline": {
            "merge": merge_cands,
            "split": split_cands,
            "reinterpret": reinterpret_cands,
        },
    }
    return (
        "You see two attached PNGs (global_plot and event_plot). JSON context:\n"
        + json.dumps(payload, indent=2, default=str)
    )


def run_observe(
    basic_output: dict[str, Any],
    global_png: bytes,
    event_png: bytes,
    merge_cands: list[dict[str, Any]],
    split_cands: list[dict[str, Any]],
    reinterpret_cands: list[dict[str, Any]],
    client: AzureOpenAI,
    deployment: str,
    cfg: Config,
) -> ObservationsOutput:
    text = _build_observe_payload(
        basic_output, merge_cands, split_cands, reinterpret_cands
    )
    user_msg = make_vision_message(
        text, [("global_plot", global_png), ("event_plot", event_png)]
    )
    messages = [
        {"role": "system", "content": _OBSERVE_SYSTEM},
        user_msg,
    ]
    return structured_call(
        client,
        deployment,
        messages,
        ObservationsOutput,
        cfg=cfg,
        agent_name="variant_observe",
        temperature=cfg.llm.variant_observe.temperature,
    )


# ---------------------------------------------------------------------------
# Subset sampling — pick non-overlapping observation sets
# ---------------------------------------------------------------------------


def _observation_targets(obs: Observation) -> set[int]:
    return set(obs.target_template_indices)


def sample_observation_subsets(
    observations: list[Observation],
    n_variants: int,
    cfg: Config,
    rng: np.random.Generator,
) -> list[list[Observation]]:
    """Return up to n_variants non-overlapping observation subsets.

    Each subset has between cfg.generation.variant_observation_subset.min and
    cfg.generation.variant_observation_subset.max observations whose target
    template indices are pairwise disjoint. The first variant always gets at
    least one observation (when any are available); duplicates across variants
    are allowed when the observation pool is small.
    """
    if not observations or n_variants <= 0:
        return [[] for _ in range(n_variants)]
    lo = max(1, int(cfg.generation.variant_observation_subset.min))
    hi = max(lo, int(cfg.generation.variant_observation_subset.max))
    out: list[list[Observation]] = []
    for _ in range(n_variants):
        target_size = int(rng.integers(lo, hi + 1))
        shuffled = list(observations)
        rng.shuffle(shuffled)
        chosen: list[Observation] = []
        used: set[int] = set()
        for obs in shuffled:
            t = _observation_targets(obs)
            if t & used:
                continue
            chosen.append(obs)
            used |= t
            if len(chosen) >= target_size:
                break
        out.append(chosen)
    return out


# ---------------------------------------------------------------------------
# Generate one variant
# ---------------------------------------------------------------------------


def _resolve_merged_span_per_occurrence(
    derived_from: list[int],
    grounding: list[dict[str, Any]],
) -> list[list[int]]:
    """Union the spans of the merged template positions, per occurrence."""
    by_pos = {int(g["template_index"]): g for g in grounding}
    if not derived_from:
        return []
    pos_spans = [by_pos[i].get("spans", []) for i in derived_from if i in by_pos]
    if not pos_spans:
        return []
    n_occ = min(len(s) for s in pos_spans)
    out: list[list[int]] = []
    for occ in range(n_occ):
        starts = [int(p[occ][0]) for p in pos_spans]
        ends = [int(p[occ][1]) for p in pos_spans]
        out.append([min(starts), max(ends)])
    return out


def _resolve_split_span_per_occurrence(
    template_index: int,
    sub_span_fraction: list[float] | None,
    grounding: list[dict[str, Any]],
) -> list[list[int]]:
    by_pos = {int(g["template_index"]): g for g in grounding}
    g = by_pos.get(int(template_index))
    if g is None:
        return []
    spans = g.get("spans", [])
    if sub_span_fraction is None or len(sub_span_fraction) != 2:
        return [[int(s[0]), int(s[1])] for s in spans]
    lo, hi = float(sub_span_fraction[0]), float(sub_span_fraction[1])
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if hi <= lo:
        return [[int(s[0]), int(s[1])] for s in spans]
    out: list[list[int]] = []
    for s in spans:
        s0, s1 = int(s[0]), int(s[1])
        L = s1 - s0
        sub_s = s0 + int(round(lo * L))
        sub_e = s0 + int(round(hi * L))
        if sub_e <= sub_s:
            sub_e = sub_s + 1
        out.append([sub_s, sub_e])
    return out


def resolve_variant_chunk_spans(
    chunks: list[VariantChunk],
    grounding: list[dict[str, Any]],
) -> list[list[list[int]]]:
    """Compute per-occurrence ts_spans for each VariantChunk."""
    out: list[list[list[int]]] = []
    for c in chunks:
        if c.op_applied == "merge":
            out.append(_resolve_merged_span_per_occurrence(c.derived_from, grounding))
        elif c.op_applied == "split":
            if not c.derived_from:
                out.append([])
                continue
            out.append(
                _resolve_split_span_per_occurrence(
                    c.derived_from[0], c.sub_span_fraction, grounding
                )
            )
        else:
            # "none" or "reinterpret" — the original primitive's spans
            if not c.derived_from:
                out.append([])
                continue
            by_pos = {int(g["template_index"]): g for g in grounding}
            g = by_pos.get(int(c.derived_from[0]))
            if g is None:
                out.append([])
            else:
                out.append([[int(s[0]), int(s[1])] for s in g.get("spans", [])])
    return out


def _build_generate_payload(
    base_chunks: list[BaseChunk],
    observations: list[Observation],
    basic_output: dict[str, Any],
    cfg: Config,
) -> str:
    grounding = basic_output.get("grounding", [])
    globals_ = basic_output.get("globals", {})
    signal_length = int(basic_output.get("signal_length", len(basic_output.get("signal", []))))
    signal = basic_output.get("signal_noisy") or basic_output.get("signal", [])

    # For each observation, compute the merged/split/reinterpreted span on the
    # FIRST occurrence and produce a feature_summary the LLM can read.
    obs_payload = []
    by_pos = {int(g["template_index"]): g for g in grounding}
    from data_gen.features import extract  # local to avoid circulars
    from data_gen.types import PrimitiveType

    for obs in observations:
        targets = obs.target_template_indices
        if obs.op == "merge":
            spans = [by_pos[t].get("spans", [])[0] for t in targets if t in by_pos]
            if not spans:
                continue
            s = min(int(p[0]) for p in spans)
            e = max(int(p[1]) for p in spans)
        elif obs.op == "split":
            t = targets[0]
            if t not in by_pos:
                continue
            sp = by_pos[t].get("spans", [])
            if not sp:
                continue
            s = int(sp[0][0])
            e = int(sp[0][1])
        else:  # reinterpret
            t = targets[0]
            if t not in by_pos:
                continue
            sp = by_pos[t].get("spans", [])
            if not sp:
                continue
            s = int(sp[0][0])
            e = int(sp[0][1])
        try:
            new_type_obj = PrimitiveType(getattr(obs.new_type, "value", obs.new_type))
            feats = extract(signal[s : e + 1], new_type_obj)
        except Exception:
            feats = {}
        obs_payload.append(
            {
                "op": obs.op,
                "target_template_indices": targets,
                "new_type": getattr(obs.new_type, "value", obs.new_type),
                "split_strategy": obs.split_strategy,
                "rationale": obs.rationale,
                "confidence": obs.confidence,
                "first_occurrence_span": [s, e],
                "feature_summary_for_new_span": feature_summary(
                    feats, globals_, signal_length, cfg.bucketize
                ),
            }
        )

    base_payload = [
        {
            "template_index": c.template_index,
            "primitive_phrase": c.primitive_phrase,
            "constraint_phrases": c.constraint_phrases,
        }
        for c in base_chunks
    ]
    template_prims = basic_output.get("template", {}).get("primitives", [])
    types_by_pos = {int(p["template_index"]): p.get("type") for p in template_prims}

    payload = {
        "template_types_by_index": types_by_pos,
        "base_chunks": base_payload,
        "observations_to_apply": obs_payload,
    }
    return (
        "Apply the listed observations and emit a VariantOutput.chunks list "
        "covering every template position exactly once (merges collapse, "
        "splits expand). Use the feature_summary_for_new_span when phrasing "
        "constraints on a merged/split/reinterpreted chunk.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


def run_generate_variant(
    base_chunks: list[BaseChunk],
    observations: list[Observation],
    basic_output: dict[str, Any],
    client: AzureOpenAI,
    deployment: str,
    cfg: Config,
    *,
    extra_instruction: str | None = None,
) -> VariantOutput:
    """Generate one variant.

    `extra_instruction`, when provided, is prepended to the user payload —
    used by the pipeline to inject a length-budget retry hint.
    """
    text = _build_generate_payload(base_chunks, observations, basic_output, cfg)
    if extra_instruction:
        text = extra_instruction.strip() + "\n\n" + text
    system = _GENERATE_SYSTEM + _length_budget_block(cfg)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    return structured_call(
        client,
        deployment,
        messages,
        VariantOutput,
        cfg=cfg,
        agent_name="variant_generate",
        temperature=cfg.llm.variant_generate.temperature,
    )
