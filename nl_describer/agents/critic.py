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

Your job, for EACH description:
  1. Check the primitive types named in the text actually look like that shape inside the claimed ts_spans on the global plot.
  2. Check the quantitative claims (high amplitude, fast oscillation, "spans most of the signal", numeric thresholds) are visually consistent.
  3. For `negative` descriptions, the structural pattern was already verified by code as not matching anywhere. Your job is only to catch paraphrases that accidentally describe a DIFFERENT but real pattern in the signal.

Verdicts:
  * accept — the description is faithful and correctly grounded.
  * borderline — minor issue (slight wording mismatch); usable.
  * reject — at least one wrong type, wrong direction, or grounded-to-wrong-region claim.

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
    text = (
        "Review every description below using the attached global_plot. "
        "Return a CritiqueOutput with one CritiqueItem per description.\n\n"
        + json.dumps(payload, indent=2, default=str)
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
                    + json.dumps(zoom_payload, indent=2, default=str)
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
