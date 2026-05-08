"""End-to-end orchestrator: BasicOutput JSON -> DescribedSample JSON.

Pipeline:
  1. Load BasicOutput JSON.
  2. Compute candidates (merge / split / reinterpret) and bucketized feature
     summaries; render global + event PNGs.
  3. Agent 1 (BaseDescriber) -> assemble base positive + negatives.
  4. Code gate on negatives via verify_no_match (drop negatives that
     accidentally match in the signal).
  5. Agent 2 (VariantAgent observe) -> observations.
  6. Sample non-overlapping subsets -> Agent 2 (VariantAgent generate, looped).
  7. Resolve variant chunk spans, assemble strings, code-gate constraints.
  8. Agent 3 (Critic) batched verification with one-zoom retry.
  9. Emit DescribedSample.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .agents.base_describer import run_base_describer
from .agents.critic import run_critic
from .agents.variant_agent import (
    resolve_variant_chunk_spans,
    run_generate_variant,
    run_observe,
    sample_observation_subsets,
)
from .candidates import (
    merge_candidates,
    reinterpret_candidates,
    split_candidates,
)
from .config import Config
from .llm import make_client
from .plotting import render_event_plot, render_global_plot
from .schemas import (
    BaseAgentOutput,
    Description,
    DescribedSample,
    GroundingChunk,
    VariantOutput,
)
from .token_limit import count_bert_tokens
from .verifier import (
    assemble_string_from_chunks,
    extract_features_for_window,
    matches_format,
    verify_constraint_on_features,
    verify_no_match,
)

logger = logging.getLogger("nl_describer.pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_spans_for_position(
    grounding: list[dict[str, Any]], template_index: int
) -> list[list[int]]:
    for g in grounding:
        if int(g.get("template_index", -1)) == int(template_index):
            return [[int(s[0]), int(s[1])] for s in g.get("spans", [])]
    return []


def _build_base_descriptions(
    base_out: BaseAgentOutput,
    basic_output: dict[str, Any],
) -> tuple[Description, list[Description]]:
    grounding = basic_output.get("grounding", [])
    event_spans = [
        [int(s[0]), int(s[1])] for s in basic_output.get("event_spans", [])
    ]

    # Positive base
    sorted_chunks = sorted(base_out.positive_chunks, key=lambda c: c.template_index)
    pairs = [(c.primitive_phrase, list(c.constraint_phrases)) for c in sorted_chunks]
    text, char_spans = assemble_string_from_chunks(pairs)
    grounding_chunks: list[GroundingChunk] = []
    for c, span in zip(sorted_chunks, char_spans):
        grounding_chunks.append(
            GroundingChunk(
                text_span=[span[0], span[1]],
                ts_spans=_ts_spans_for_position(grounding, c.template_index),
            )
        )
    positive_base = Description(
        kind="positive_base",
        text=text,
        grounding=grounding_chunks,
        event_spans=event_spans,
    )

    # Negatives
    negatives_in: list[Description] = []
    raw_negs = basic_output.get("neg_groundings", []) or []
    for i, neg_desc in enumerate(base_out.negatives):
        if i >= len(raw_negs):
            break
        seed = raw_negs[i]
        sorted_neg_chunks = sorted(
            neg_desc.chunks, key=lambda c: c.template_index
        )
        pairs = [
            (c.primitive_phrase, list(c.constraint_phrases))
            for c in sorted_neg_chunks
        ]
        ntext, _ = assemble_string_from_chunks(pairs)
        negatives_in.append(
            Description(
                kind="negative",
                text=ntext,
                grounding=[],
                event_spans=[],
                source_neg_index=i,
                mutation=seed.get("mutation"),
            )
        )
    return positive_base, negatives_in


def _gate_negatives(
    negatives: list[Description],
    basic_output: dict[str, Any],
) -> tuple[list[Description], list[Description]]:
    """Drop any negative whose underlying (types, constraints) pattern
    accidentally matches in the signal. Returns (kept, dropped)."""
    raw_negs = basic_output.get("neg_groundings", []) or []
    layout = basic_output.get("whole_time_series_layout", [])
    signal = basic_output.get("signal", [])
    globals_ = basic_output.get("globals", {})
    kept: list[Description] = []
    dropped: list[Description] = []
    for d in negatives:
        if d.source_neg_index is None or d.source_neg_index >= len(raw_negs):
            kept.append(d)
            continue
        seed = raw_negs[d.source_neg_index]
        types = [g.get("type") for g in seed.get("grounding", [])]
        cons_by_pos = [g.get("constraints", []) for g in seed.get("grounding", [])]
        ok = verify_no_match(types, cons_by_pos, layout, signal, globals_)
        if ok:
            kept.append(d)
        else:
            d.critic_verdict = "reject"
            d.critic_issues = ["paraphrase pattern unexpectedly matches in signal"]
            dropped.append(d)
    return kept, dropped


def _build_variant_descriptions(
    variants: list[tuple[VariantOutput, list[str]]],
    basic_output: dict[str, Any],
) -> list[Description]:
    grounding = basic_output.get("grounding", [])
    event_spans = [
        [int(s[0]), int(s[1])] for s in basic_output.get("event_spans", [])
    ]
    out: list[Description] = []
    for variant_out, applied_obs in variants:
        chunks = list(variant_out.chunks)
        per_chunk_spans = resolve_variant_chunk_spans(chunks, grounding)
        pairs = [
            (c.primitive_phrase, list(c.constraint_phrases)) for c in chunks
        ]
        text, char_spans = assemble_string_from_chunks(pairs)
        grounding_chunks = [
            GroundingChunk(text_span=[s[0], s[1]], ts_spans=ts)
            for s, ts in zip(char_spans, per_chunk_spans)
        ]
        out.append(
            Description(
                kind="positive_variant",
                text=text,
                grounding=grounding_chunks,
                event_spans=event_spans,
                applied_observations=applied_obs,
            )
        )
    return out


def _bert_tokens(text: str, cfg: Config) -> int:
    return count_bert_tokens(text, cfg.length_limit.tokenizer_model)


def _within_token_budget(text: str, cfg: Config) -> bool:
    return _bert_tokens(text, cfg) <= cfg.length_limit.max_bert_tokens


def _length_overflow_hint(cfg: Config) -> str:
    """Extra-instruction text injected on retry after a length overflow."""
    target_words = max(50, cfg.length_limit.soft_word_budget // 2)
    return (
        f"NOTE: A previous attempt produced one or more descriptions that exceeded the "
        f"{cfg.length_limit.max_bert_tokens}-token BERT budget. Be substantially more concise "
        f"this time: aim for ~{target_words} words per assembled description, drop decorative "
        "synonyms, and shorten or simplify constraint phrases. Do NOT change semantics, but "
        "prefer the plainest wording that still preserves operator meaning."
    )


def _gate_variant_constraints(
    variant: Description,
    variant_out: VariantOutput,
    basic_output: dict[str, Any],
) -> bool:
    """Best-effort sanity check: re-extract features on the chunk's first
    occurrence span and ensure no obviously-wrong claim. We only check by
    confirming feature extraction works; deep claim-vs-feature checking is
    deferred to the critic."""
    signal = basic_output.get("signal", [])
    chunks = variant_out.chunks
    for chunk, gc in zip(chunks, variant.grounding):
        if not gc.ts_spans:
            return False
        s, e = gc.ts_spans[0][0], gc.ts_spans[0][1]
        if e <= s or e >= len(signal):
            return False
        try:
            feats_list = extract_features_for_window(
                [getattr(chunk.new_type, "value", chunk.new_type)],
                [(s, e)],
                signal,
            )
            if not feats_list or not feats_list[0]:
                return False
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def describe_sample(
    in_path: str | Path,
    cfg: Config | None = None,
    *,
    plots_dir: str | Path | None = None,
    seed: int = 0,
) -> DescribedSample:
    in_path = Path(in_path)
    cfg = cfg or Config.load(Config.default_path())
    rng = np.random.default_rng(seed)

    with open(in_path, "r") as f:
        basic_output: dict[str, Any] = json.load(f)

    signal_noisy = basic_output.get("signal_noisy") or basic_output.get("signal", [])
    layout = basic_output.get("whole_time_series_layout", [])
    event_spans_raw = basic_output.get("event_spans", [])
    events = basic_output.get("events", [])

    # Step 1: candidates + plots
    merge_cands = merge_candidates(events, cfg.candidates)
    split_cands = split_candidates(events, cfg.candidates)
    reinterpret_cands = reinterpret_candidates(
        events, signal_noisy, cfg.candidates
    )

    _, global_png = render_global_plot(
        signal_noisy, layout, event_spans_raw, cfg.vision.global_plot
    )
    if events:
        chosen_event_idx = int(rng.integers(0, len(events)))
        _, event_png = render_event_plot(
            signal_noisy, events[chosen_event_idx], cfg.vision.event_plot
        )
    else:
        event_png = global_png

    if plots_dir is not None:
        pdir = Path(plots_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "global.png").write_bytes(global_png)
        (pdir / "event.png").write_bytes(event_png)

    # Step 2: build LLM client
    client, deployment = make_client(cfg.azure)
    logger.info("Azure OpenAI client ready (deployment=%s)", deployment)

    # Step 3: BaseDescriber (with one retry if the assembled text overflows
    # the BERT-token budget; the retry is on the whole agent call because
    # positive_base + all negatives come from a single structured response).
    base_out = run_base_describer(basic_output, client, deployment, cfg)
    positive_base, negatives = _build_base_descriptions(base_out, basic_output)
    logger.info(
        "Base describer: positive_base + %d negatives produced", len(negatives)
    )

    if cfg.length_limit.retry_on_overflow:
        over: list[str] = []
        if not _within_token_budget(positive_base.text, cfg):
            over.append(f"positive_base({_bert_tokens(positive_base.text, cfg)}t)")
        for i, n in enumerate(negatives):
            if not _within_token_budget(n.text, cfg):
                over.append(f"negative[{i}]({_bert_tokens(n.text, cfg)}t)")
        if over:
            logger.warning(
                "base_describer over BERT-token budget %d: %s; retrying once",
                cfg.length_limit.max_bert_tokens,
                over,
            )
            try:
                base_out = run_base_describer(
                    basic_output,
                    client,
                    deployment,
                    cfg,
                    extra_instruction=_length_overflow_hint(cfg),
                )
                positive_base, negatives = _build_base_descriptions(
                    base_out, basic_output
                )
            except Exception as e:
                logger.warning("base_describer retry failed: %s", e)

    # Step 4a: hard length gate (drop survivors that still exceed)
    if not _within_token_budget(positive_base.text, cfg):
        logger.error(
            "positive_base still over BERT-token budget after retry: %dt > %d; dropping",
            _bert_tokens(positive_base.text, cfg),
            cfg.length_limit.max_bert_tokens,
        )
        positive_base = None  # type: ignore[assignment]
    kept_negatives: list[Description] = []
    for i, n in enumerate(negatives):
        if _within_token_budget(n.text, cfg):
            kept_negatives.append(n)
        else:
            logger.warning(
                "negative[%d] dropped: over BERT-token budget (%dt > %d)",
                i,
                _bert_tokens(n.text, cfg),
                cfg.length_limit.max_bert_tokens,
            )
    negatives = kept_negatives

    # Step 4b: gate negatives against signal pattern (existing semantic gate)
    negatives, _dropped = _gate_negatives(negatives, basic_output)
    logger.info(
        "Negative gate: %d kept, %d dropped (matched somewhere in signal)",
        len(negatives),
        len(_dropped),
    )

    # Step 5: VariantAgent observe
    obs_out = run_observe(
        basic_output,
        global_png,
        event_png,
        merge_cands,
        split_cands,
        reinterpret_cands,
        client,
        deployment,
        cfg,
    )
    observations = list(obs_out.observations)[: cfg.generation.n_observations_max]
    logger.info("Observations produced: %d", len(observations))

    # Step 6: sample subsets and generate variants
    subsets = sample_observation_subsets(
        observations,
        cfg.generation.n_positive_variants,
        cfg,
        rng,
    )
    variants_descs: list[Description] = []
    accepted_variant_outs: list[VariantOutput] = []
    for subset in subsets:
        if not subset:
            # Skip empty subsets — no point regenerating positive_base.
            continue
        try:
            variant_out = run_generate_variant(
                base_out.positive_chunks,
                subset,
                basic_output,
                client,
                deployment,
                cfg,
                previous_variants=accepted_variant_outs,
            )
        except Exception as e:
            logger.warning("variant_generate failed: %s", e)
            continue
        applied_tags = [
            f"{o.op}@{','.join(map(str, o.target_template_indices))}->{getattr(o.new_type, 'value', o.new_type)}"
            for o in subset
        ]
        built = _build_variant_descriptions([(variant_out, applied_tags)], basic_output)
        if not built:
            continue
        variant_desc = built[0]

        # Length gate with optional one-retry. Re-call the agent with a
        # "be more concise" hint, then re-build and re-check; drop on
        # second failure.
        if not _within_token_budget(variant_desc.text, cfg):
            tokens_before = _bert_tokens(variant_desc.text, cfg)
            if cfg.length_limit.retry_on_overflow:
                logger.warning(
                    "variant over BERT-token budget (%dt > %d); retrying once",
                    tokens_before,
                    cfg.length_limit.max_bert_tokens,
                )
                try:
                    variant_out = run_generate_variant(
                        base_out.positive_chunks,
                        subset,
                        basic_output,
                        client,
                        deployment,
                        cfg,
                        extra_instruction=_length_overflow_hint(cfg),
                        previous_variants=accepted_variant_outs,
                    )
                    rebuilt = _build_variant_descriptions(
                        [(variant_out, applied_tags)], basic_output
                    )
                    if rebuilt:
                        variant_desc = rebuilt[0]
                except Exception as e:
                    logger.warning("variant retry failed: %s", e)
            if not _within_token_budget(variant_desc.text, cfg):
                logger.warning(
                    "variant dropped: still over BERT-token budget (%dt > %d)",
                    _bert_tokens(variant_desc.text, cfg),
                    cfg.length_limit.max_bert_tokens,
                )
                continue

        if not matches_format(variant_desc.text):
            logger.warning("variant rejected: bad format -> %s", variant_desc.text)
            continue
        if not _gate_variant_constraints(variant_desc, variant_out, basic_output):
            logger.warning("variant rejected: constraint gate failed")
            continue
        variants_descs.append(variant_desc)
        accepted_variant_outs.append(variant_out)

    # Format check on positive_base too
    if positive_base is not None and not matches_format(positive_base.text):
        logger.warning(
            "positive_base failed format regex; keeping anyway: %s", positive_base.text
        )

    all_descriptions: list[Description] = []
    if positive_base is not None:
        all_descriptions.append(positive_base)
    all_descriptions.extend(variants_descs)
    all_descriptions.extend(negatives)

    # Step 8: Critic
    try:
        critiques = run_critic(
            all_descriptions, global_png, signal_noisy, client, deployment, cfg
        )
        for c in critiques:
            if 0 <= c.description_index < len(all_descriptions):
                d = all_descriptions[c.description_index]
                d.critic_verdict = c.verdict
                d.critic_issues = list(c.issues)
    except Exception as e:
        logger.warning("critic failed: %s", e)

    return DescribedSample(
        source_path=str(in_path), descriptions=all_descriptions
    )
