1. Define the number of event occurrences based on nb_event_occurrence and pdf_nb_event_occurrence. The same template will be reused across all occurrences (steps 3, 4, 7, 9 are sampled once per signal); per-occurrence variation comes only from the jitter step (step 15).

2. Once the number of event occurrences is defined, define the number of inter-event primitives. It should be a vector having nb_event_occurrence+1 elements. Each element represents the number of primitives between two adjacent events (including before the first and after the last). Use nb_inter_event_primitives, pdf_nb_inter_event_primitives, min_nb_weight, max_nb_weight for sampling the numbers. For the "linear" pdf, min_nb_weight and max_nb_weight are the endpoint weights of a linear ramp over the integer range, normalized to a probability distribution.

3. Define the number of primitives inside an event, based on nb_primitives_in_event and pdf_nb_primitives_in_event. Sampled once per signal and shared across all occurrences.

4. (after 3) Create a list of event primitive types. The number of elements should equal to the number of primitives inside an event. Use primitive_type_samping_weight for sampling the primitive types. Sampled once per signal and shared across all occurrences.

5. (after 2) Create a list of inter-event primitive types(p): [(p1), (p2, p3), (p4), (p5, p6, p7)]. Each element is a sublist. The length of the sublist should be the same as defined in step 2. Use primitive_type_samping_weight for sampling the primitive types.

6. (after 3) Define the length of event. Using min_primitive_length, and the nb_primitives_in_event the min length can be deduced (event_min_length = nb_primitives_in_event * min_primitive_length). The max is deduced as max_event_length = (signal_length - sum_of_inter_event_min_lengths) / nb_event_occurrences, where sum_of_inter_event_min_lengths uses min_primitive_length per inter-event primitive. Use pdf_event_length, event_average_length_weight, event_extreme_length_weight to sample the length.

7. (after 6) Define the in-event primitives length. Use sequential sampling with budget tracking because the total of the primitives' durations should equal event_length. During the scope calculation, use min_primitive_length, min_spike_length, max_spike_length (the spike bounds apply only when the primitive type is spike or dip). For sampling, use pdf_primitives_length, primitive_average_length_weight, primitive_extreme_length_weight. Average length weight means length = remaining length / nb of remaining primitives.
Note 1: each primitive's realized values array has length `duration + 1`. Concatenation in step 10 shares one sample at every boundary, so the event spans `event_length + 1` samples after concatenation (n primitives contribute `sum(duration_i + 1) − (n − 1) = event_length + 1`).

8. (after 6) Define the inter-event primitives length. Use sequential sampling with budget tracking. The total inter-event duration budget is `signal_length − 1 − sum_of_event_lengths_after_concat = signal_length − 1 − nb_event_occurrences * (event_length + 1)`, accounting for Note 1 and for the final shared boundaries between events and inter-event regions. During the scope calculation, use min_primitive_length, min_spike_length, max_spike_length (the spike bounds apply only when the primitive type is spike or dip). For sampling, use pdf_primitives_length, primitive_average_length_weight, primitive_extreme_length_weight. Average length weight means length = remaining length / nb of remaining primitives.

9. (after 7 and 8) Realize each primitive in temporal (left-to-right) order, first in-event, then inter-event. Each primitive has two kinds of generative inputs: (a) **definitive** inputs, fixed before this step — start_value=0 for every primitive, peak2peak from config (event_primitive_peak_2_peak / inter_event_primitive_peak_2_peak), and duration from the layout planner in steps 6/7/8; (b) **random** inputs, sampled here from the per-feature `scope` and `pdf` declared in feature_schema.yaml (e.g. spike's rise_duration and recovery; oscillation's frequency, decay, resonance, regularity, waveform_shape; increase/decrease's shape; plateau has no random inputs). Once all generative inputs are set, realize the values and compute the derived features. The change point needs to be visible: the slope at the starting point of the next primitive should be different from the slope at the ending point of the previous primitive. The generated primitive therefore needs a validation. If the slope rule is not met, re-sample the random inputs and re-realize until a sample meeting the requirement is generated. When an inter-event primitive is placed before an event (the event already being defined), validate against both sides. The current rule is strict inequality: end_slope != start_slope, with slope computed as values[t+1] - values[t] at the boundary samples. A tolerance range will be added in the future.
Note 2.1: Here we got the first version of features. Let's call it raw features.
Note 2.2 (resolved): The schema extensions called out here are now applied in feature_schema.yaml. The `generative` field is categorical: `definitive` (pre-set externally, no scope/pdf) | `random` (sampled with scope+pdf) | `derived` (computed from realized values). The definitive triple is {start_value (fixed=0), peak2peak (config), duration (layout)} for every primitive (plateau omits peak2peak). start_slope and end_slope are derived on every primitive.

10. (after 9) Concatenate all the generated primitives. The concatenation should be done by first shifting the primitive - bring the end point of the previous primitive and the start point of the next primitive to the same level, then merge the end point and the start point at the SAME indice.

11. (after 10) Perform Z-normalization of the concatenated signal

12. (after 11) Because of the shifting and the z-norm, the features are changed. Recalculate the features. Since the occurrences are at different levels, split the in-event primitive features into level-dependent and level-independent. Level-dependent features are any *_value (start_value, end_value, baseline_value, mean_value, etc.) — these differ per occurrence. Level-independent features (slope, duration, amplitude/peak2peak, sharpness, asymmetry, frequency, period, n_cycles, shape, waveform_shape) remain identical across occurrences.
Note 3.1: Let's call the features "normed features".
Note 3.2 (resolved): feature_schema.yaml now carries a `level_dependent: true|false` flag per feature.

13. (after 12) Randomly generate the number of constraints (c) for each primitive in the event, using nb_constraints, pdf_nb_constraints.

14. (after 13) For each constraint, choose numerical constraint or comparable constraint using numerical_constraint_weight, comparable_constraint_weight.

15. (after 14) If numerical, choose feature (f) 1 from features that are numerical-constraintable. Choose an operation and a value based on the current feature value. See "Distinctiveness scoring" below for how to pick f1 and the threshold.

16. (after 14) If comparable, choose feature (f) 1 from features that are comparable-constraintable. Choose primitive 2 and feature 2 from feature 1's `comparable_features` list (which may include `global.*` references — global is a pseudo-primitive whose features are signal-level statistics: min, max, mean, std, peak2peak, etc.). Choose an operation based on the feature values. See "Distinctiveness scoring" below.
Note 4 (resolved): feature_schema.yaml now adds a `comparable_features` list per constraintable feature, and a `global` pseudo-primitive section.

Distinctiveness scoring (steps 15 and 16):
For each candidate constraint c on a primitive instance, compute the **information content** under the generation prior:
  - P(c) = probability that c holds, evaluated over the prior distributions of the feature(s) involved.
    - For a random feature, the prior is the per-feature `scope` + `pdf` declared in feature_schema.yaml. Closed-form for uniform/linear; numeric integration / Monte Carlo (~1k samples) for skewed_normal.
    - For a definitive feature, the prior is its config-bound or layout-bound distribution.
    - For a derived feature, the prior is computed empirically by sampling N=1k realizations from the underlying generative inputs and applying the derivation.
    - For comparable constraints `p1.f1 op p2.f2`, sample (x, y) from the marginal priors of f1 and f2 (assumed independent at generation time) and estimate P(x op y).
    - All P(c) values are cached per (feature, op, threshold) or (p1, f1, p2, f2, op).
  - info(c) = −log(max(P(c), ε)) where ε is a small floor to bound the score.
  - weight(c) = salience(f1) · info(c), where salience is the per-feature [0, 1] prior declared in feature_schema.yaml (default 0.5).
Sample `nb_constraints` constraints from the candidate pool weighted by weight(c), without replacement.

Threshold selection for numerical constraints (step 15):
Given the realized value v of f1, generate candidate (op, T) pairs by rounding v to a small set of round-ish boundaries (e.g., 1 sig fig, half-integer, integer) in the satisfying direction. Evaluate info(c) for each candidate; the candidate pool feeds the weighted sampling above.

Constraint-diversity (avoiding redundant correlated picks across multiple constraints on the same primitive) is intentionally NOT modeled in v1 — revisit if multiple constraints frequently end up redundant in practice.

17. (after 14) Using jitter_randomness to add random jitter to the generative features (normed features) of the in-event primitives, generate new event occurrences. For each occurrence: jitter generative features (in normed space), realize values, fit into the global signal via shift+merge. Check that the new occurrence meets the constraints; resample until it does. The jitter magnitude should depend on the peak2peak value and the duration of the primitive.

18. (after 17) Concatenate all the generated primitives, z-norm the whole signal, re-calculate the features.
Note 5: let's call the features "jittered features".

19. (after 18) Calculate the position of each primitive and each event occurrence. Spans use the inclusive convention [start, end]; adjacent primitives share a sample at their boundary index because of the shift+merge in step 10.

20. (after 19) Generate text description of the event.
Open: signal-relative templating. Constraints are emitted as raw thresholds (e.g., `peak2peak > 7.3`). At template-rendering time, map raw thresholds to natural-language buckets that reference the signal's globals (e.g., "high amplitude" when `peak2peak / global.peak2peak` is in the top tertile of the in-event primitives' values; "spans most of the signal" when `duration / signal_length` is large; "fast oscillation" when `frequency` exceeds k × the median oscillation frequency of the signal). Bucket boundaries are derived per-signal from `global.*` and per-signal aggregates rather than baked into feature_schema.yaml — they adapt across signals and survive z-norm. To be designed when the templating module is built.

The basic output:
- signal: list[number]
- grounding: list (in-event primitives only; inter-event primitives are not grounded)
    For each element:
        - primitive type
        - list of constraint
            For each constraint: p1, f1 operation p2, f2
        - spans: list of span [start, end] (inclusive). The start and end indices of each occurrence of the primitive in the time series.
- event_spans: list of span [start, end] (inclusive). The start and end indices of each occurrence of the event.
    





