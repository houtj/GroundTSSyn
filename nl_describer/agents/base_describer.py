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


def _length_budget_block(cfg: Config) -> str:
    """Length-budget snippet appended to the system prompt.

    The numbers come from `cfg.length_limit` so the LLM is informed of the
    same budget that code will hard-enforce (with headroom for sub-token
    expansion in WordPiece).
    """
    return (
        "\n\nLength budget (CRITICAL):\n"
        f"- The full assembled description for each item (positive base, each negative) must stay under "
        f"~{cfg.length_limit.soft_word_budget} words and ~{cfg.length_limit.soft_char_budget} characters total, "
        f"so it fits a {cfg.length_limit.max_bert_tokens}-token BERT input window.\n"
        "- The 'full assembled description' is every primitive_phrase + its constraint_phrases, joined as "
        "`primitive: c, c; primitive: c, c`. Count them all together, not per-chunk.\n"
        "- Prefer terse, direct phrasing over decorative synonyms when many constraints stack up.\n"
        "- Keep individual chunks 3-10 words (already required above); when many positions or constraints "
        "are present, lean to the lower end of that range.\n"
        "- Numbers, hyphenated terms, and rare words count as multiple tokens, not one. Round numbers "
        "and avoid compound jargon when you can express the same claim plainly."
    )


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
                    constraint_summary(
                        c, template_types, globals_, signal_length, cfg.bucketize
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
                        constraint_summary(
                            c, template_types, globals_, signal_length, cfg.bucketize
                        )
                        for c in entry.get("constraints", [])
                    ],
                }
            )
        negatives_payload.append({"mutation": neg.get("mutation"), "grounding": items})

    payload = {
        "signal_length": signal_length,
        "globals": {
            k: globals_.get(k)
            for k in (
                "min",
                "max",
                "mean",
                "std",
                "peak2peak",
                "frequency",
                "histogram_peak",
            )
        },
        "template_types": template_types,
        "positive": positive_payload,
        "negatives": negatives_payload,
    }
    instruction = (
        "Produce structured output with two fields:\n"
        "  positive_chunks: one BaseChunk per template position (in template order). "
        "Each BaseChunk has template_index, primitive_phrase, and constraint_phrases (one per input constraint, same order).\n"
        "  negatives: one NegativeDescription per negative seed (same order as input). "
        "Each NegativeDescription has chunks: list[NegativeChunk], one per position in that seed's grounding.\n\n"
        "Input data follows as JSON:\n"
    )
    return instruction + json.dumps(payload, indent=2, default=str)


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
    system = _SYSTEM + _length_budget_block(cfg)
    messages = [
        {"role": "system", "content": system},
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
