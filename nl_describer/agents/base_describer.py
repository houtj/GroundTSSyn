"""Agent 1 — BaseDescriber.

Text-only paraphrasing of the positive base + every negative seed. The LLM
emits structured `(primitive_phrase, [constraint_phrase, ...])` chunks; code
joins them into the canonical `primitive[: c, c]; primitive[: c, c]` format
(constraint-less primitives have no trailing colon).
"""

from __future__ import annotations

import json
from typing import Any

from openai import AzureOpenAI

from ..bucketize import constraint_summary, feature_summary
from ..config import Config
from ..llm import structured_call
from ..schemas import BaseAgentOutput

_SYSTEM = """You are a meticulous time-series annotator. You will receive an event template extracted from a 1D signal: each template position has a primitive type and zero or more constraints. You will also receive a list of "negative-grounding seeds" — variant templates that do NOT match the signal anywhere.

Your job:
1. For every template position, produce ONE varied human paraphrase of the primitive type (`primitive_phrase`) plus ONE paraphrase per constraint (`constraint_phrase`). The constraint phrases must appear in the same order as the input constraints.
2. For each negative seed, produce the same kind of phrases. Negatives describe events NOT in the signal but you should still write fluent, plausible-sounding descriptions — pretend you are the discriminator and you are stating what is absent.

Critical rules:
- Operator semantics MUST be preserved. `>` reads as "greater than / above / exceeds / longer than", NEVER "around" or "approximately".
- Do NOT invent constraints. Do NOT change the type at any position. Do NOT skip a constraint.
- Vary your wording across positions and across calls. For each item pick a paraphrase strategy from {type-synonym, comparative, bucketed-quantitative, metaphor}.
- Keep each phrase short (3-10 words). Do not include the literal characters ":" or ";" inside a phrase, since they are reserved as our format separators.
- Duration phrasing: NEVER state an exact sample count for any duration-related constraint. Use qualitative / bucketed wording instead (e.g. "short", "long", "brief", "extended", "very brief", or relative comparisons like "lasting about as long as the spike"). Forbidden examples: "lasting 12 samples", "for 30 samples", "about 14 samples", "~12 samples". This applies to BOTH positive and negative descriptions, and to BOTH `duration` numerical constraints AND `duration`-vs-`duration` comparable constraints.

Examples of varied primitive phrases:
- spike: "abrupt overshoot", "sharp upward jolt", "sudden peak", "transient pulse"
- dip: "brief downward dip", "sharp valley", "momentary trough"
- increase: "gradual rise", "steady ramp up", "ascending trend"
- decrease: "downward slope", "gradual decline", "ramp down"
- plateau: "flat segment", "level period", "rest"
- oscillation: "rapid fluctuation", "wavy segment", "oscillating section"

Examples of varied constraint phrases:
- `peak2peak > 7` -> "with a high amplitude exceeding 7", "noticeably tall (over 7)"
- `duration == another.duration` -> "lasting the same number of samples as the second decrease"
- `shape == convex` -> "with a convex profile"
- `recovery_type == partial` -> "that does not fully return to baseline"
"""


def _length_budget_line(cfg: Config) -> str:
    """One-line length budget reminder, appended to the user message.

    Replaces the older multi-bullet block in the system prompt: same hard
    numbers (`cfg.length_limit`) but ~10x shorter. Phrased to remind the
    model of the constraint without re-explaining the assembly format,
    which is already covered in the system prompt.
    """
    return (
        f"Length budget: full assembled description (every primitive_phrase + constraint_phrases joined) "
        f"<= ~{cfg.length_limit.soft_word_budget} words / "
        f"~{cfg.length_limit.soft_char_budget} chars, fits a "
        f"{cfg.length_limit.max_bert_tokens}-token BERT window. Prefer terse phrasing."
    )


def _slim_constraint(cs: dict[str, Any], *, for_negative: bool) -> dict[str, Any]:
    """Drop fields the LLM doesn't need to paraphrase the constraint.

    Conservative slim: keep `op` + `threshold` (semantics-preserving) and
    `realized_bucket` (magnitude hint). Drop the redundant raw
    `realized_value` for positives. For negatives, drop both the raw value
    AND the bucket — negatives don't realize anywhere meaningful, so any
    realized-* field is misleading noise to the paraphraser.
    """
    out = dict(cs)
    kind = cs.get("kind")
    if kind == "numerical":
        out.pop("realized_value", None)
        if for_negative:
            out.pop("realized_bucket", None)
    elif kind == "comparable":
        for side in ("lhs", "rhs"):
            v = out.get(side)
            if isinstance(v, dict):
                out[side] = {k: vv for k, vv in v.items() if k != "realized_value"}
    return out


def _build_user_payload(
    basic_output: dict[str, Any],
    cfg: Config,
) -> str:
    template = basic_output.get("template", {})
    template_prims = template.get("primitives", [])
    template_types = [p.get("type") for p in template_prims]
    grounding = basic_output.get("grounding", [])
    globals_ = basic_output.get("globals", {})
    signal_length = int(basic_output.get("signal_length", len(basic_output.get("signal", []))))

    # First-event features power the bucketed feature_summary.
    events = basic_output.get("events", [])
    first_event_prims = events[0].get("primitives", []) if events else []
    feats_by_pos: dict[int, dict[str, Any]] = {}
    for p in first_event_prims:
        feats_by_pos[int(p["template_index"])] = p.get("features", {})

    positive_payload = []
    for g in grounding:
        idx = int(g["template_index"])
        feats = feats_by_pos.get(idx, {})
        positive_payload.append(
            {
                "template_index": idx,
                "type": g.get("type"),
                "n_event_occurrences": len(g.get("spans", [])),
                "feature_summary": feature_summary(
                    feats, globals_, signal_length, cfg.bucketize
                ),
                "constraints": [
                    _slim_constraint(
                        constraint_summary(
                            c, template_types, globals_, signal_length, cfg.bucketize
                        ),
                        for_negative=False,
                    )
                    for c in g.get("constraints", [])
                ],
            }
        )

    raw_negs = basic_output.get("neg_groundings", []) or []
    capped = raw_negs[: cfg.generation.max_negatives]
    negatives_payload = []
    for neg in capped:
        items = []
        for entry in neg.get("grounding", []):
            items.append(
                {
                    "template_index": entry.get("template_index"),
                    "type": entry.get("type"),
                    "constraints": [
                        _slim_constraint(
                            constraint_summary(
                                c, template_types, globals_, signal_length, cfg.bucketize
                            ),
                            for_negative=True,
                        )
                        for c in entry.get("constraints", [])
                    ],
                }
            )
        negatives_payload.append({"mutation": neg.get("mutation"), "grounding": items})

    # Static-prefix-friendly key order: signal_length, then positive/negatives
    # which depend on it. `globals` and `template_types` were dropped — the
    # bucketized feature/constraint summaries already encode magnitude info,
    # and per-position `type` covers the template type list.
    payload = {
        "signal_length": signal_length,
        "positive": positive_payload,
        "negatives": negatives_payload,
    }
    instruction = (
        "Produce structured output with two fields:\n"
        "  positive_chunks: one BaseChunk per template position (in template order). "
        "Each BaseChunk has template_index, primitive_phrase, and constraint_phrases (one per input constraint, same order).\n"
        "  negatives: one NegativeDescription per negative seed (same order as input). "
        "Each NegativeDescription has chunks: list[NegativeChunk], one per position in that seed's grounding.\n\n"
        + _length_budget_line(cfg)
        + "\n\nInput data follows as JSON:\n"
    )
    return instruction + json.dumps(payload, separators=(",", ":"), default=str)


def run_base_describer(
    basic_output: dict[str, Any],
    client: AzureOpenAI,
    deployment: str,
    cfg: Config,
    *,
    extra_instruction: str | None = None,
) -> BaseAgentOutput:
    """Run the base describer.

    `extra_instruction`, when provided, is prepended to the user payload —
    used by the pipeline to inject a "your previous output exceeded the
    length budget; be more concise" hint on retry.
    """
    user_text = _build_user_payload(basic_output, cfg)
    if extra_instruction:
        user_text = extra_instruction.strip() + "\n\n" + user_text
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_text},
    ]
    return structured_call(
        client,
        deployment,
        messages,
        BaseAgentOutput,
        cfg=cfg,
        agent_name="base_describer",
        temperature=cfg.llm.base_describer.temperature,
    )
