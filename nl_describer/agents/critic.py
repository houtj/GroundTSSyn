"""Agent 3 — Critic.

Batched vision verifier. Sees the global plot + a batch of descriptions with
their grounding, returns accept/borderline/reject per item with optional
free-zoom requests. Bounded one zoom retry per item.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AzureOpenAI

from ..config import Config
from ..llm import make_vision_message, structured_call
from ..plotting import render_slice_plot
from ..schemas import CritiqueItem, CritiqueOutput, Description

_SYSTEM = """You are an expert reviewer of natural-language annotations of a 1D time series. You will see the global signal plot (with primitive boundaries colored by type) and a batch of candidate descriptions. Each description has:
  * kind: positive_base | positive_variant | negative
  * text: the natural-language description
  * grounding: list of {text_span, ts_spans} mapping each `primitive: c, c` chunk to the time-series spans it claims to ground
  * event_spans: occurrences of the event in the signal (positive_base / positive_variant only)
  * applied_observations: only on positive_variant — what merges / splits / reinterpretations were applied

General checks, for EACH description (regardless of kind):
  1. Check that the primitive type named in each chunk actually looks like that shape inside the claimed ts_spans on the global plot (correct type, correct direction, correct location).
  2. Check that the quantitative / qualitative claims (high amplitude, fast oscillation, "spans most of the signal", numeric thresholds, "convex", "partial recovery", "longer than X", etc.) are visually consistent with the signal inside the grounded spans.
  3. Check that the chunk ordering in the text matches the temporal ordering of the grounded ts_spans within each event.

Per-kind guidance — apply ALL three of these, do NOT skip positive_base:
  * `positive_base` — this is the canonical paraphrase of the extracted template. You MUST critique it with the same rigor as the others. Common failure modes to look for:
      - the primitive_phrase in a chunk uses a word that implies the WRONG type or wrong direction (e.g. saying "rise" or "spike" for a chunk that grounds to a decrease/dip; saying "plateau" for a sloped segment; calling a short blip a "long" event).
      - a constraint paraphrase flips the operator semantics ("around 7" instead of "greater than 7"; "shorter than" instead of "longer than"; dropping a negation).
      - a constraint paraphrase silently drops a constraint or invents one not in the input.
      - the grounded ts_spans for a chunk visibly contain a DIFFERENT primitive type than the text claims.
    Treat positive_base as a first-class candidate for `borderline` or `reject` whenever any of these occur. Do NOT default to `accept` just because it is the base description.
  * `positive_variant` — additionally validate that the `applied_observations` (merge / split / reinterpret tags) are consistent with what the text and ts_spans show; reject if a "merge" chunk grounds to clearly disjoint segments of different shapes, a "split" chunk's two halves do not actually look like the two claimed sub-types, or a "reinterpret" chunk's new type does not visually fit.
  * `negative` — the structural pattern was already verified by code as not matching anywhere. Your only extra job here is to catch paraphrases that accidentally describe a DIFFERENT but real pattern in the signal.
    HIGH BAR FOR REJECTING NEGATIVES (read carefully — the default is `accept`):
      - The whole point of a negative is that some structural detail (a primitive type, a direction, an ordering, or a constraint) is wrong. Such a description will naturally still SHARE most of its primitives with the real motif — that overlap alone is NOT grounds to reject.
      - Reject ONLY when the FULL ordered chunk sequence in the negative text — every primitive type, in the stated order, with the stated directions and constraints — can be traced contiguously over a single window of the signal. "Most chunks line up but one or two are reordered / wrong type / wrong direction" is exactly what a valid negative looks like; accept it.
      - Apply these concrete tests before rejecting:
          (a) Pick the candidate window in the signal that best matches the negative.
          (b) Walk the negative's chunks left-to-right and check that EACH chunk's primitive type AND its stated direction/constraint is visually present in the corresponding sub-window in that exact order. If ANY chunk fails (wrong type, wrong direction, wrong relative ordering vs its neighbors, or a constraint that does not hold there), the negative is valid — return `accept`.
          (c) A reordering of two adjacent primitives (e.g. `flat → rise` swapped to `rise → flat`) IS a real structural difference. Do not dismiss it as "middle ordering differs somewhat" — that swap alone is enough to keep the negative valid.
          (d) A type substitution that changes the shape family (e.g. "sharp valley" instead of "small trough" when no deep valley exists in that position; "downward slope" instead of "brief dip" when the signal does not contain an extended slope there) IS a real structural difference. Accept the negative.
      - Phrases like "overall it still describes a real recurring pattern", "broadly similar to the motif", or "most of the sequence is visible" are NOT sufficient justification to reject. The matching must be exact-in-order across the full chunk list.
      - When in doubt about a negative, prefer `accept` (or `borderline` with a note) over `reject`. The downstream code-level gate already filtered structural matches; your job is only to catch the rare case where the paraphrase exactly traces a different real pattern.

Verdicts:
  * accept — the description is faithful and correctly grounded (for positives), OR (for negatives) the chunk sequence is NOT exactly traceable in order anywhere in the signal.
  * borderline — minor issue (slight wording mismatch, mild over-claim); usable but flag the issue in `issues`.
  * reject — at least one wrong type, wrong direction, dropped/invented constraint, flipped operator, or grounded-to-wrong-region claim (for positives), OR (for negatives) the full chunk sequence in order is exactly traceable on a window of the signal.

Always populate `issues` with a short, concrete reason whenever the verdict is `borderline` or `reject` (e.g. "chunk 2 says 'spike' but ts_span [120,138] is a downward dip"). For `accept`, leave `issues` empty. For a `reject` on a negative, your `issues` MUST cite the specific signal window (approximate sample range) where the full ordered chunk sequence is visible; if you cannot name such a window, do not reject.

If a span is too small / overlapping to judge from the global plot, set free_zoom_request=[s, e] for ONE slice you want zoomed in. Use this sparingly — only when you genuinely cannot decide.
"""


def _slim_description(desc: Description, idx: int) -> dict[str, Any]:
    return {
        "description_index": idx,
        "kind": desc.kind,
        "text": desc.text,
        "grounding": [
            {"text_span": gc.text_span, "ts_spans": gc.ts_spans}
            for gc in desc.grounding
        ],
        "event_spans": desc.event_spans,
        "applied_observations": desc.applied_observations,
    }


def run_critic(
    descriptions: list[Description],
    global_png: bytes,
    signal_noisy: list[float],
    client: AzureOpenAI,
    deployment: str,
    cfg: Config,
) -> list[CritiqueItem]:
    if not descriptions:
        return []

    payload = [_slim_description(d, i) for i, d in enumerate(descriptions)]
    kinds_summary = ", ".join(
        f"#{i}={d.kind}" for i, d in enumerate(descriptions)
    )
    text = (
        "Review every description below using the attached global_plot, "
        "INCLUDING any with kind='positive_base'. Apply the same critique "
        "rigor to positive_base as to positive_variant and negative — do "
        "not auto-accept it. For each description, return a CritiqueItem "
        "with description_index, verdict (accept|borderline|reject), and "
        "concrete `issues` strings whenever verdict is borderline or "
        "reject. Return a CritiqueOutput with one CritiqueItem per "
        f"description (you must produce {len(descriptions)} items, one "
        f"for each of: {kinds_summary}).\n\n"
        + json.dumps(payload, separators=(",", ":"), default=str)
    )
    user_msg = make_vision_message(text, [("global_plot", global_png)])
    messages = [
        {"role": "system", "content": _SYSTEM},
        user_msg,
    ]
    parsed: CritiqueOutput = structured_call(
        client,
        deployment,
        messages,
        CritiqueOutput,
        cfg=cfg,
        agent_name="critic",
        temperature=cfg.llm.critic.temperature,
    )
    items: list[CritiqueItem] = list(parsed.items)

    # Handle one-zoom-retry: re-issue any item that requested a zoom, with the
    # zoomed slice attached.
    final_items: list[CritiqueItem] = []
    for it in items:
        if it.free_zoom_request and it.verdict != "accept":
            try:
                s, e = int(it.free_zoom_request[0]), int(it.free_zoom_request[1])
                _, png_bytes = render_slice_plot(
                    signal_noisy, s, e, cfg.vision.slice_plot
                )
                desc = descriptions[it.description_index]
                zoom_payload = {
                    "description": _slim_description(desc, it.description_index),
                    "previous_verdict": it.verdict,
                    "previous_issues": it.issues,
                }
                zoom_text = (
                    "You requested a zoom on this slice. The zoomed plot is "
                    "attached. Re-decide the verdict for this single description. "
                    "Return a CritiqueOutput.items with EXACTLY ONE CritiqueItem "
                    f"having description_index={it.description_index}.\n\n"
                    + json.dumps(zoom_payload, separators=(",", ":"), default=str)
                )
                zoom_msg = make_vision_message(
                    zoom_text,
                    [
                        ("global_plot", global_png),
                        (f"zoom_{s}_{e}", png_bytes),
                    ],
                )
                zoom_messages = [
                    {"role": "system", "content": _SYSTEM},
                    zoom_msg,
                ]
                zoom_out: CritiqueOutput = structured_call(
                    client,
                    deployment,
                    zoom_messages,
                    CritiqueOutput,
                    cfg=cfg,
                    agent_name="critic_zoom",
                    temperature=cfg.llm.critic.temperature,
                )
                if zoom_out.items:
                    final_items.append(zoom_out.items[0])
                    continue
            except Exception:
                pass
        final_items.append(it)
    return final_items
