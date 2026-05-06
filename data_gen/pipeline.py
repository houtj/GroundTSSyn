from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_gen.concat import concatenate
from data_gen.config import Config
from data_gen.constraints import ConstraintsResult, sample_constraints
from data_gen.grounding import (
    BasicOutput,
    OccurrenceVerification,
    apply_extra_occurrences,
    build_basic_output,
    filter_failed_constraints,
    verify_constraints,
)
from data_gen.jitter import (
    JitterReport,
    assemble_with_occurrences,
    build_jittered_occurrences,
)
from data_gen.layout import Layout, plan_layout
from data_gen.negation import NegativeGrounding, build_negative_groundings
from data_gen.noise import add_white_noise, sample_noise_level
from data_gen.normalize import z_normalize_with_stats
from data_gen.orchestrate import (
    assemble_layout,
    realize_event_template,
    realize_inter_event_primitives,
)
from data_gen.post_match import ExtraOccurrence, find_extra_occurrences
from data_gen.priors import EmpiricalPriors
from data_gen.recompute import NormedFeatures, recompute_features
from data_gen.schema import Schema
from data_gen.types import ConcatResult, Event, Primitive


@dataclass
class PipelineResult:
    layout: Layout
    event_template: Event
    inter_event: list[list[Primitive]]
    raw: ConcatResult
    signal: list[float]
    mean: float
    std: float


@dataclass
class ExtendedPipelineResult:
    base: PipelineResult
    normed: NormedFeatures
    constraints: ConstraintsResult


@dataclass
class FinalPipelineResult:
    extended: ExtendedPipelineResult
    occurrences: list[Event]
    jitter_report: JitterReport
    raw_jittered: ConcatResult
    z_signal: list[float]
    z_mean: float
    z_std: float
    z_signal_noisy: list[float]
    noise_sigma: float
    jittered_features: NormedFeatures
    basic_output: BasicOutput
    verification: list[OccurrenceVerification]
    post_match_extras: list[ExtraOccurrence]
    post_match_indices: list[int]
    neg_groundings: list[NegativeGrounding]


def run_steps_1_to_11(
    cfg: Config | None = None,
    schema: Schema | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> PipelineResult:
    cfg = cfg or Config.load()
    schema = schema or Schema.load()
    rng = rng or np.random.default_rng()

    layout = plan_layout(cfg, rng)
    template = realize_event_template(cfg, layout, schema, rng)
    inter = realize_inter_event_primitives(cfg, layout, template, schema, rng)
    items = assemble_layout(layout, template, inter)
    raw = concatenate(items)
    signal, mean, std = z_normalize_with_stats(raw.signal)

    assert len(raw.signal) == cfg.signal_length, (
        f"signal length mismatch: got {len(raw.signal)}, expected {cfg.signal_length}"
    )
    return PipelineResult(
        layout=layout,
        event_template=template,
        inter_event=inter,
        raw=raw,
        signal=signal,
        mean=mean,
        std=std,
    )


def run_steps_1_to_16(
    cfg: Config | None = None,
    schema: Schema | None = None,
    *,
    rng: np.random.Generator | None = None,
    prior_K: int = 200,
) -> ExtendedPipelineResult:
    cfg = cfg or Config.load()
    schema = schema or Schema.load()
    rng = rng or np.random.default_rng()

    base = run_steps_1_to_11(cfg, schema, rng=rng)
    normed = recompute_features(base.raw, base.signal, base.layout, base.event_template)
    priors = EmpiricalPriors.build(
        cfg, schema, rng, K=prior_K, signal_std=base.std
    )
    constraints = sample_constraints(
        cfg, schema, base.event_template, normed, priors, rng
    )
    return ExtendedPipelineResult(base=base, normed=normed, constraints=constraints)


def run_steps_1_to_19(
    cfg: Config | None = None,
    schema: Schema | None = None,
    *,
    rng: np.random.Generator | None = None,
    prior_K: int = 200,
    jitter_max_attempts: int = 200,
) -> FinalPipelineResult:
    cfg = cfg or Config.load()
    schema = schema or Schema.load()
    rng = rng or np.random.default_rng()

    extended = run_steps_1_to_16(cfg, schema, rng=rng, prior_K=prior_K)
    base = extended.base

    occurrences, jitter_report = build_jittered_occurrences(
        cfg,
        schema,
        base.layout,
        base.event_template,
        base.inter_event,
        rng,
        extended.constraints,
        max_attempts=jitter_max_attempts,
    )

    items = assemble_with_occurrences(base.layout, occurrences, base.inter_event)
    raw_jittered = concatenate(items)
    z_signal, z_mean, z_std = z_normalize_with_stats(raw_jittered.signal)

    assert len(z_signal) == cfg.signal_length, (
        f"jittered signal length mismatch: got {len(z_signal)}, "
        f"expected {cfg.signal_length}"
    )

    # White-noise injection: σ in z-normed units (clean signal has unit std).
    # Clean z_signal is preserved as ground truth and used by feature
    # extraction + post-matching + negation; z_signal_noisy is the delivered
    # output for downstream consumers.
    noise_sigma = sample_noise_level(cfg, rng)
    z_signal_noisy = add_white_noise(z_signal, noise_sigma, rng)

    reference_event = occurrences[0] if occurrences else base.event_template
    jittered_features = recompute_features(
        raw_jittered, z_signal, base.layout, reference_event, sigma=noise_sigma
    )

    template_types = [p.type for p in base.event_template.primitives]
    basic_output = build_basic_output(
        z_signal, z_signal_noisy, raw_jittered, template_types, extended.constraints
    )
    verification = verify_constraints(extended.constraints, jittered_features)
    basic_output, verification = filter_failed_constraints(basic_output, verification)

    extras: list[ExtraOccurrence] = []
    extra_indices: list[int] = []
    if cfg.post_match.enabled:
        extras = find_extra_occurrences(
            raw_jittered,
            z_signal,
            base.event_template,
            [g.constraints for g in basic_output.grounding],
            jittered_features.globals,
        )
        basic_output, extra_indices = apply_extra_occurrences(basic_output, extras)

    neg_groundings = []
    if cfg.negation.enabled:
        neg_groundings = build_negative_groundings(
            [g.type for g in basic_output.grounding],
            [list(g.constraints) for g in basic_output.grounding],
            raw_jittered,
            z_signal,
            jittered_features.globals,
            schema,
            rng,
            max_attempts=cfg.negation.max_attempts,
            target_n=cfg.negation.target_n,
        )

    return FinalPipelineResult(
        extended=extended,
        occurrences=occurrences,
        jitter_report=jitter_report,
        raw_jittered=raw_jittered,
        z_signal=z_signal,
        z_mean=z_mean,
        z_std=z_std,
        z_signal_noisy=z_signal_noisy,
        noise_sigma=noise_sigma,
        jittered_features=jittered_features,
        basic_output=basic_output,
        verification=verification,
        post_match_extras=extras,
        post_match_indices=extra_indices,
        neg_groundings=neg_groundings,
    )
