# Pipeline overview

The pipeline is implemented as three nested entry points in `pipeline.py`, each extending the previous:

- `run_steps_1_to_11` → layout planning + realization + concatenation + z-norm (steps 1-11)
- `run_steps_1_to_16` → adds feature recomputation + constraint sampling (steps 12-16)
- `run_steps_1_to_19` → adds jitter/multi-occurrence assembly, noise, grounding output, post-matching, negatives (steps 17+)

Two config flags change the shape of steps 1-9: `cfg.periodical` (tiled/periodic layout instead of randomly-spaced events) and `cfg.drift_limit` (constrains in-event net drift instead of validating slopes for increase/decrease primitives). Both are noted inline where they diverge.

## Layout planning (`layout.py`)

1. Define the number of event occurrences (`nb_events`) from `nb_event_occurrence` / `pdf_nb_event_occurrence`. The same event template is reused across all occurrences (steps 3, 4, 7, 9 are sampled once per signal); per-occurrence variation comes only from the jitter step (step 17).
   - Periodical mode: this instead samples a *target* full-event count `n_target`; the actual tiled count `nb_events = M` is derived later in step 6 from the sampled event length so that `M` tiles cover `signal_length`.

2. Define the number of inter-event primitives: a vector of `nb_events + 1` elements, one per gap (including before the first and after the last event). Sampled with `sample_linear_integer` using `nb_inter_event_primitives`, `pdf_nb_inter_event_primitives`, `min_nb_weight`, `max_nb_weight` (linear ramp over the integer range, normalized to a probability distribution).
   - Periodical mode: skipped — there are no inter-event primitives; all gap counts are 0.

3. Define the number of primitives inside an event (`nb_primitives_in_event`, `pdf_nb_primitives_in_event`). Sampled once per signal and shared across all occurrences.

4. (after 3) Sample the in-event primitive type sequence via weighted-categorical sampling (`primitive_type_samping_weight`). Adjacent-pair resampling suppresses `DISALLOWED_ADJACENCY` (plateau→plateau, spike↔oscillation). Sampled once per signal and shared across all occurrences.
   - Drift-limit mode: additionally requires the template be "drift-neutral capable" — either no increase/decrease primitives, or at least one of each — otherwise the layout attempt is rejected and retried (periodical mode only checks this explicitly; non-periodical relies on the solve in step 9).

5. (after 2) Sample inter-event primitive types per slot, one sublist per gap, using the same weighted-categorical + disallowed-adjacency logic. The first/last primitive of each slot is checked against the event template's boundary primitive it abuts, so no plateau-plateau or spike-oscillation pair forms across the event boundary.
   - Periodical mode: skipped (no inter-event primitives).

6. (after 3, 4) Define the event length (in DURATION units — duration = samples − 1; a primitive of duration *d* contributes *d+1* samples but adjacent primitives share their boundary sample, so `total_samples = sum(durations) + 1` holds across the whole signal):
   - Non-periodical: `min_event_length = nb_primitives_in_event * min_primitive_length`; `max_event_length = (signal_length − 1 − sum_of_inter_event_min_durations) // nb_events`. Sampled with `sample_skewed_normal_integer` using `pdf_event_length`, `event_average_length_weight`, `event_extreme_length_weight`.
   - Periodical: bounds are `L ∈ [signal_length / (n_target+2), signal_length / (n_target-1)]` (clamped to `min_event_length`), sampled uniformly. The actual tile count `M = ceil((signal_length − 1) / event_length)` is then derived, a random `tile_offset ∈ [0, overshoot]` is chosen so the tiled signal can be truncated on either edge, and `nb_events` is overwritten to `M`.

7. (after 6) Sample in-event primitive durations by sequential budget-tracked sampling so the total equals `event_length` exactly. Per-step bounds respect `min_primitive_length` (and `min_spike_length`/`max_spike_length` when the type is spike/dip), reserving enough budget for remaining primitives. Sampling uses `sample_skewed_normal_integer_around` with `pdf_primitives_length`, `primitive_average_length_weight`, `primitive_extreme_length_weight` (average-length weight biases toward `remaining_length / nb_remaining_primitives`).
   Note: each primitive's realized values array has length `duration + 1`. Concatenation shares one sample at every boundary, so the event spans `event_length + 1` samples after concatenation.

8. (after 6, non-periodical only) Sample inter-event primitive durations the same way, with total budget `signal_length − 1 − nb_events * event_length`, then split back into per-slot lists. (Periodical mode: no inter-event primitives, so this step is skipped.)

`plan_layout` retries the whole steps 1-8 procedure (via `_plan_once` / `_plan_once_periodical`) up to `max_attempts` times if a `LayoutInfeasible` is raised (e.g. budget can't satisfy per-primitive minima, or a periodical template can't be drift-neutral).

## Realization (`orchestrate.py`)

9. Realize each primitive in temporal order, first the event template, then the inter-event primitives slot-by-slot. Each primitive has two kinds of generative inputs:
   - **definitive**: `start_value = 0` for every primitive, `peak2peak` sampled uniformly from `event_primitive_peak_2_peak` / `inter_event_primitive_peak_2_peak` (plateau has no `peak2peak`), and `duration` from the layout planner.
   - **random**: sampled from the per-feature `scope` + `pdf` declared in `feature_schema.yaml` (e.g. spike's `rise_duration`/`recovery`; oscillation's `frequency`/`decay`/`resonance`/`regularity`/`waveform_shape`; increase/decrease's `shape`; plateau has none).

   Change points must be visible: `validate_slope` requires the boundary slope of a primitive to differ (strict inequality) from its neighbor's boundary slope, and `_junction_angle_ok` additionally rejects junctions whose interior angle (treating each slope as a 2-D direction) is at or above `cfg.max_junction_angle_deg` — i.e. a junction that is *visually* too close to a straight line, even if the slopes aren't bit-for-bit equal. Plateau neighbors are exempt (a flat-to-slope transition is always visible). If validation fails, `_realize_with_retry` re-samples the random inputs and re-realizes (up to `max_attempts`).
   - The event template additionally self-validates its loop-closure junction (`template[-1]` end vs `template[0]` start), since a `k=0` inter-event slot makes two template copies abut directly. `realize_event_template` retries the whole template (up to `template_max_attempts`) if this fails.
   - Inter-event slots use a two-sided check for the last primitive of any non-trailing slot, validating against the following event template's first primitive.
   - **Drift-limit mode** (`cfg.drift_limit`): bypasses slope validation for the event template. All non-increase/decrease primitives are realized first (in random relative order for inc/dec), then increase/decrease primitives are realized in a shuffled order; the *last* one's `peak2peak` is solved (not sampled) so the total event drift (`end_value − start_value` summed across primitives) falls in `[-drift_threshold, +drift_threshold]`. If no direction-preserving solution exists, the primitive's type is flipped (increase↔decrease) to reverse the contribution sign.

10. (after 9) Concatenate all primitives (`concat.py`) in `[*inter[0], event, *inter[1], event, …, event, *inter[N]]` order. Concatenation shifts each primitive so its start level matches the previous primitive's end level, then merges the shared boundary sample.
    - Periodical mode: the full M-event tile is built and then truncated by `tile_offset` on the left and the remainder on the right to hit exactly `signal_length` samples (`truncate_concat_result`).

11. (after 10) Z-normalize the concatenated signal (`z_normalize_with_stats`), producing `signal`, `mean`, `std`. This is the end of `run_steps_1_to_11` (`PipelineResult`).

## Feature recomputation & constraint sampling (`recompute.py`, `constraints.py`, `priors.py`)

12. (after 11) Because shifting + z-norm changed the primitives' realized values, recompute features from the z-normed signal slices (`recompute_features`). Features split into level-dependent (any `*_value`: `start_value`, `end_value`, `baseline_value`, `mean_value`, etc. — differ per occurrence) and level-independent (`slope`, `duration`, `amplitude`/`peak2peak`, `sharpness`, `asymmetry`, `frequency`, `period`, `n_cycles`, `shape`, `waveform_shape` — identical across occurrences pre-jitter), per the `level_dependent` flag in `feature_schema.yaml`. This step also:
    - re-merges non-recoverable random fields (`waveform_shape`, `envelope_behavior`, `envelope_asymmetry`, `regularity`, `shape`) that can't be derived from realized values alone;
    - computes per-primitive `visibility` (amplitude/duration/neighbor-masking SNR vs. noise σ — see `visibility.py`) and per-boundary `cp_visibility` (slope + trend-level mismatch SNR at each change point), aggregated per-primitive as the min of its two boundary visibilities;
    - computes signal-level `global.*` statistics (`global_features.py`: min, max, mean, std, peak2peak, etc.);
    - in periodical mode, drops events whose first or last template-position primitive was truncated away, so downstream consumers only see fully-realized events.
    These are the "normed features".

13-16. (after 12) Sample constraints per in-event template position (`sample_constraints`, using only the first occurrence's normed features as the reference since level-independent features are identical across occurrences pre-jitter):
    - **13.** Sample the requested constraint count per primitive from `nb_constraints` / `pdf_nb_constraints`.
    - **14.** For each candidate slot, the constraint *kind* (numerical / comparable / self) is chosen implicitly by pooling all three candidate types and sampling by weight (see below) — `numerical_constraint_weight` and `comparable_constraint_weight` set the pool-level mixing weight (self shares the comparable weight).
    - **15. Numerical**: for every schema feature tagged `numerical` in `constraint`, generate round-ish candidate thresholds near the realized value (`_candidate_thresholds`: nearest integer, half-integer, 1-sig-fig, and coarser round numbers), keep only (op, threshold) pairs the realized value currently satisfies, and score by distinctiveness (below).
    - **16. Comparable**: for every schema feature tagged `comparable` with a non-null `comparable_domain`, find peer features sharing that domain (`schema.comparable_peers`) across other template positions of the same type, or `global.*`. For each peer, pick the operator(s) currently true for the realized (lhs, rhs) pair, and score by distinctiveness.
    - **Self** (schema-driven, not in the original numbered steps): for every feature tagged `self` in `constraint` (categorical features, e.g. `shape`, `waveform_shape`), emit an equality constraint on the realized categorical value if its prior probability is distinctive enough. Self constraints are informational-only — `jitter.py` classifies them as `"defer"` (not enforced during jitter) since they have no operator to re-satisfy.

    **Distinctiveness scoring** (steps 15-16 and self): for each candidate constraint `c`, compute `P(c)` — the probability it holds under the generation prior (`EmpiricalPriors`, built via Monte-Carlo sampling of `prior_K` synthetic primitives per type from the schema's `scope`+`pdf`, cached per `(feature, op, threshold)` or `(p1, f1, p2, f2, op)`/`(feature, value)`). Candidates with `p <= P_LOW` or `p >= P_HIGH` (0.01 / 0.99) are dropped as trivial. `info(c) = −log(max(P(c), ε))`; `weight(c) = salience(f1) · info(c)` where salience is the per-feature `[0,1]` prior in `feature_schema.yaml` (default 0.5). The final pool (across numerical/comparable/self) is sampled without replacement, weighted by `weight(c)`, until `requested_count` constraints are picked or the pool is exhausted.

    This produces `ConstraintsResult` (`run_steps_1_to_16`'s `ExtendedPipelineResult`).

## Multi-occurrence jitter (`jitter.py`)

17. (after 16) Generate the N event occurrences from the shared template by jittering, enforcing every level-independent (non-global) constraint from step 13-16 that jitter can act on. Constraints referencing `global.*` or any level-dependent feature (or `self` constraints) are classified `"defer"` — not enforced here, only checked afterward by `verify_constraints`/`filter_failed_constraints`. The pipeline runs in three stages:
    1. **Duration matrix** (`jitter_durations_constant_sum`): jitter each in-event primitive's duration by ±`jitter_randomness`% per occurrence while keeping the *sum* across all occurrences equal to `nb_events * sum(template_durations)` (per-element deltas re-centered to zero-mean, then rounded with floor-respecting redistribution of rounding error). Retried until any duration-only constraints are satisfied for every occurrence.
    2. **Per-occurrence build** (`jitter_event_occurrence`): for each occurrence, walk template positions in order; at each position, jitter the `definitive` dict (duration fixed from stage 1; other level-independent numeric fields ±`jitter_randomness`%, clipped to schema `scope`) and the `random` dict (same uniform-% rule), re-realize, and retry (up to `max_attempts`) until `validate_slope` passes against already-built neighbors and any intra-primitive constraints (both sides on this position) hold.
       - Drift-limit mode: one inc/dec position per occurrence is deferred (chosen at random); every other position is jittered normally, then the deferred primitive's `peak2peak` is solved so total occurrence drift stays within `drift_threshold` (slope/intra-constraint checks are skipped for the deferred position, mirroring template realization).
    3. **Cross-primitive retry**: once an occurrence is fully built, evaluate cross-primitive constraints (both sides within this occurrence, different positions) against the full feature set; if any fail, rebuild the whole occurrence (durations stay fixed) up to `max_attempts` times.
    Raises `ConstraintRetryExhausted` if any stage exhausts its attempt budget.

18. (after 17) Concatenate all `N` (possibly-different) occurrences with the inter-event primitives (`assemble_with_occurrences` → `concatenate`), truncate if periodical, and z-normalize the whole signal again (new `z_signal`/`z_mean`/`z_std`). Recompute features from this final signal (`recompute_features`, using occurrence 0 as the reference event) — these are the "jittered features", now including the real per-occurrence visibility/cp_visibility values.

19. (after 18) Inject noise: sample `noise_sigma` from `noise_level`/`pdf_noise_level` (0 if `cfg.add_noise` is False) and add i.i.d. `N(0, σ²)` per sample (`add_white_noise`) to get `signal_noisy`. The clean `z_signal` is preserved as ground truth for feature extraction, post-matching, and negation; `signal_noisy` is the delivered input signal for downstream consumers.

## Grounding, verification, post-matching, negation, output (`grounding.py`, `post_match.py`, `negation.py`, `serialization.py`)

20. Build the basic grounding output (`build_basic_output`): one span list per template position across all kept occurrences (inter-event primitives are never grounded), paired with that position's sampled constraints. Periodical mode drops any event whose first or last template-position primitive was truncated away (and its `event_span`).

21. Verify constraints against the final jittered features (`verify_constraints`): every `(occurrence, template_position, constraint)` triple is checked (True / False / None-if-unevaluable). Any constraint that fails for *any* occurrence is dropped from the grounding entirely (`filter_failed_constraints`) — the delivered grounding only contains constraints that hold for every kept occurrence.

22. Post-match pass (optional, `cfg.post_match.enabled`): scan the inter-event primitive runs for windows whose primitive-type sequence matches the event template and whose features satisfy every surviving constraint (`find_extra_occurrences`). These are naturally-occurring, unplanned instances of the pattern; they're appended to the grounding's spans/event_spans (`apply_extra_occurrences`), re-sorted temporally, and tagged as `source="post_match"` in the enriched output.

23. Negative-grounding generation (optional, `cfg.negation.enabled`): produce up to `target_n` structurally-plausible negative groundings (`build_negative_groundings`) by applying one mutation to the (type sequence, constraints) pair — `remove_middle` (drop an interior position), `switch_type` (replace one position's type), `swap_positions` (exchange two differently-typed positions), or `invert_operator` (flip one constraint's comparison). A mutation is only accepted as a negative if sliding it over every window of the realized `placed` sequence finds **no** window that matches both the type sequence and all constraints (unevaluable constraints count as failing, i.e. as support for the negative).

24. Enrich the output (`enrich_basic_output`): populate `signal_length`, `noise_sigma`, `globals` (signal-level stats), `template` (abstract event definition: type + duration per position), `whole_time_series_layout` (every primitive in temporal order, tagged `in_event`/`inter_event`, with span/visibility/cp_visibility/features), and `events` (in-event primitives grouped by occurrence, including post-match extras, each tagged `source="template"|"post_match"`).

Open: text description generation. Not yet implemented in `data_gen`. Planned approach — signal-relative templating: constraints are raw thresholds (e.g. `peak2peak > 7.3`); at template-rendering time, map raw thresholds to natural-language buckets that reference the signal's globals (e.g. "high amplitude" when `peak2peak / global.peak2peak` is in the top tertile of the in-event primitives' values; "spans most of the signal" when `duration / signal_length` is large; "fast oscillation" when `frequency` exceeds k × the median oscillation frequency of the signal). Bucket boundaries would be derived per-signal from `global.*` and per-signal aggregates rather than baked into `feature_schema.yaml`, so they adapt across signals and survive z-norm.

## Final output shape (`BasicOutput` in `grounding.py`)

- `signal`: list[float] — clean z-normed signal (ground truth).
- `signal_noisy`: list[float] — `signal` + white noise (`noise_sigma`); the delivered input.
- `signal_length`, `noise_sigma`.
- `globals`: dict — signal-level statistics (min, max, mean, std, peak2peak, etc.).
- `template`: the abstract event definition (primitive types + durations per template position).
- `events`: list, one per kept occurrence (template-derived or post-match), each with `event_index`, `span`, `source`, and its per-position primitives (`type`, `span`, `visibility`, `cp_visibility`, `features`).
- `whole_time_series_layout`: every primitive in the signal in temporal order (`in_event` or `inter_event`), with span/visibility/cp_visibility/features.
- `grounding`: list, one per in-event template position — `type`, list of `constraint` (numerical / comparable / self, each with operator, threshold/peer, and the distinctiveness-scoring fields), and `spans` (inclusive `[start, end]` per occurrence, template + post-match).
- `event_spans`: list of inclusive `[start, end]` spans, one per kept event occurrence.
- `neg_groundings`: list of mutated (type sequence, constraints) pairs verified to have no match anywhere in the realized signal.
