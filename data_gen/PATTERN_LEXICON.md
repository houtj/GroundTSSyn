# Time-Series Pattern Lexicon (v3 — dimension/term model)

A domain-invariant vocabulary of terms describing the **morphology** (shape)
of univariate time series. This is a design document for the "Open: text
description generation" item in `pipline.md` — the missing layer between
`feature_schema.yaml`'s 6 generative primitives and natural-language
pattern descriptions.

Scope is unchanged from v1/v2: a term is included **iff its meaning is
fully determined by curve shape**, independent of domain semantics (see v1
history for excluded examples: "anomaly", "crash", "fever spike", etc.).

---

## 1. The model

### 1.1 Node type

Everything is one recursive node type:

```
PrimitiveNode
  ├─ constituents: ordered list of PrimitiveNode | wildcard-slot   (empty ⇒ "basic")
  ├─ features: generative/derived fields (as in feature_schema.yaml)
  └─ repeat: optional (sub-sequence, count) — for unbounded recurrence
```

### 1.2 Dimension vs. feature

- A **feature** is a raw generative or derived field, exactly as in
  `feature_schema.yaml` (`slope`, `duration`, `sharpness`, ...).
- A **dimension** is a named, *typed* (`numerical` | `ordinal` |
  `categorical`) axis used to place a vocabulary term. A dimension may be:
  - **identical to a feature** (1:1, e.g. dimension `duration` ≡ feature
    `duration`), or
  - **a derived view over one feature** (e.g. dimension `direction` =
    `sign(slope)`, categorical; dimension `steepness` = `|slope|`,
    numerical — both over the single feature `slope`), or
  - **a function over multiple features directly** (e.g. dimension
    `sharpness` = `peak2peak / duration`, spanning two features without
    first being staged as its own feature).
  - **a comparative view across constituents** of a compound (e.g.
    dimension `peak_ratio` = `crest₁.peak2peak / crest₂.peak2peak`) — this
    kind only exists at the compound level, never on a basic primitive.
- A **term** (a vocabulary word) = a node identity + a list of
  `(dimension: constraint)` pairs. A term with one pair is what earlier
  drafts called a "dependent primitive"; a term with several pairs is what
  earlier drafts called a "conjunction" — same representation, no special
  case needed.

### 1.3 Compound identity and constituents

A **compound**'s identity is its **ordered list of constituent terms** —
there is no separate display name for the identity itself; it is always
referenced by that list (e.g. `[increase, decrease]`), never given an
extra label like "crest transient."

Constituents are normally already-**named terms** (e.g. `increase`,
`decrease`, or even another compound term, since nesting is allowed:
`double-top = [spike, gap, spike]`). A constituent slot may instead be
**explicitly unconstrained** — written with a wildcard suffix, e.g.
`ramp[*]` (direction free) or `gap[*]` (either `plateau` or a low-slope
`ramp` — filler content, unconstrained) — used sparingly, only where a
term's meaning genuinely does not depend on that slot's value. Wildcard
slots are always visually marked; they are never silently omitted.

Forming a compound generates **emergent dimensions/features** (`recovery`,
`sharpness`, `peak_ratio`, ...) — intrinsic, deterministic functions of the
constituent sequence, fixed the moment the identity is fixed, before any
specific term picks which ones to constrain (**Rule F**, below).

A given term's `constraints` column lists **only the new constraint(s) it
adds on top of the constituents' own base constraints** — e.g. `spike`'s
row does not repeat "`slope>0` then `slope<0`" (that's already implied by
its constituents `increase`, `decrease`); it lists only `sharpness: high`.

### 1.4 Rules

- **Rule A — Dimension test.** A term that constrains one or more
  dimensions of an existing node, without changing which other
  dimensions/features remain meaningful, is a term of that node — not a
  new node type. (Subsumes the old "Rule A′ conjunction" case: a
  multi-dimension term is still just a term, with a longer constraint
  list.)
- **Rule B — Compound/sequence test.** A candidate is a compound only if
  it requires ≥ 2 temporally distinct constituents (terms and/or wildcard
  slots) arranged in order.
- **Rule C — Irreducibility test.** A shape requiring *unbounded*
  repetition (count is a free parameter, not a structural choice) is
  either a basic primitive with native repetition features (`oscillation`
  — hence it stays basic), or built with the explicit `repeat(sub-seq, n)`
  operator when the repeated unit isn't already covered by an existing
  basic primitive's own dimensions (§4.4).
- **Rule D — Redundancy test.** If a shape is already reachable as a term
  on an *existing* basic primitive's own dimensions, do not also define it
  via `repeat(...)` of smaller parts (`zigzag`/`sawtooth`/`whipsaw` stay
  under `oscillation`, §3.3, not `repeat(ramp, ramp)`).
- **Rule E — Degeneracy promotion test.** A constrained boundary region of
  a basic primitive is promoted to its **own basic primitive** (overriding
  Rule A) iff, within that region, one or more of the parent's *other*
  dimensions/features become undefined or lose their degrees of freedom —
  not merely take an extreme value. This is why `plateau` (≈ `ramp` with
  `slope = 0`) stays a separate basic primitive: at `slope = 0`, the
  `curvature` dimension becomes meaningless and `peak2peak` stops being a
  free parameter, unlike `increase`/`decrease` where `curvature`,
  `duration`, `peak2peak` all remain fully meaningful on both sides of
  `slope = 0`.
- **Rule F — Emergent-dimension integrity.** Emergent dimensions/features
  of a compound are intrinsic, deterministic functions of the constituent
  sequence, defined once per identity — never redefined per term.

---

## 2. Basic primitives (3) — intrinsic features

| Primitive | Constituents | Common features | Distinctive (intrinsic) features |
|---|---|---|---|
| `plateau` (P) | — | `start_value`, `duration`, `end_value` | *(none beyond common — Rule E)* |
| `ramp` (R) | — | `start_value`, `duration`, `end_value`, `peak2peak` | `slope`, `curvature` (linear/convex/concave) |
| `oscillation` (O) | — | `start_value`, `duration`, `end_value`, `peak2peak` | `frequency`, `period`, `n_cycles`, `envelope_behavior`, `envelope_asymmetry`, `regularity`, `waveform_shape`, `center_trend` |

`spike`/`dip` remain implemented as their own generative primitives in
`feature_schema.yaml` for practical sampling reasons; ontologically they
are compound terms on `[increase, decrease]` / `[decrease, increase]` —
see §4.1. The implementation layer and this ontology layer may diverge.

---

## 3. Basic-primitive terms

`term | primitive | dimension(s) | dimension type | relevant feature(s) | constraint(s) | reference`

### 3.1 `ramp`

| Term | Primitive | Dimension(s) | Type | Feature(s) | Constraint(s) | Reference |
|---|---|---|---|---|---|---|
| increase | ramp | direction | categorical | slope | `slope > 0` | `[KBTA]` |
| decrease | ramp | direction | categorical | slope | `slope < 0` | `[KBTA]` |
| steep / sharp | ramp | steepness | numerical | slope | `\|slope\|` high | `New` |
| gentle / gradual | ramp | steepness | numerical | slope | `\|slope\|` low | `[LingSum]` |
| brief | ramp | duration | numerical | duration | `duration` low | `New` |
| sustained / prolonged | ramp | duration | numerical | duration | `duration` high | `New` |
| linear | ramp | curvature | categorical | curvature | `curvature = linear` | `New` |
| accelerating | ramp | curvature | categorical | curvature | `curvature = convex` | `New` |
| decelerating | ramp | curvature | categorical | curvature | `curvature = concave` | `New` |
| large-magnitude | ramp | magnitude | numerical | peak2peak | `peak2peak` high | `New` |
| small-magnitude | ramp | magnitude | numerical | peak2peak | `peak2peak` low | `New` |
| drift / creep | ramp | steepness, duration | numerical, numerical | slope, duration | `\|slope\|` low; `duration` high | `[KBTA]` |
| surge | ramp | direction, steepness, magnitude | categorical, numerical, numerical | slope, slope, peak2peak | `slope > 0`; `\|slope\|` high; `peak2peak` high | `New` |
| plunge | ramp | direction, steepness, magnitude | categorical, numerical, numerical | slope, slope, peak2peak | `slope < 0`; `\|slope\|` high; `peak2peak` high | `New` |
| tick up / tick down | ramp | direction, magnitude | categorical, numerical | slope, peak2peak | `slope ≷ 0`; `peak2peak` low | `New` |

### 3.2 `plateau`

| Term | Primitive | Dimension(s) | Type | Feature(s) | Constraint(s) | Reference |
|---|---|---|---|---|---|---|
| brief plateau | plateau | duration | numerical | duration | `duration` low | `New` |
| extended plateau / consolidation-base | plateau | duration | numerical | duration | `duration` high | `[TA-chart]` |

### 3.3 `oscillation`

| Term | Primitive | Dimension(s) | Type | Feature(s) | Constraint(s) | Reference |
|---|---|---|---|---|---|---|
| wedge / funnel (narrowing) | oscillation | envelope | categorical | envelope_behavior | `envelope_behavior = decaying` | `[TA-chart]` |
| megaphone / broadening | oscillation | envelope | categorical | envelope_behavior | `envelope_behavior = resonating` | `[TA-chart]` |
| channel | oscillation | envelope | categorical | envelope_behavior | `envelope_behavior = sustained` | `[TA-chart]` |
| high-frequency | oscillation | frequency | numerical | frequency | `frequency` high | `[TA-IEPF]` |
| low-frequency | oscillation | frequency | numerical | frequency | `frequency` low | `[TA-IEPF]` |
| seasonal cycle | oscillation | regularity | categorical | regularity | `regularity = regular` | `[TA-IEPF]`, `[BEDTime/CaTS]` |
| sawtooth pattern | oscillation | waveform | categorical | waveform_shape | `waveform_shape = sawtooth` | schema field |
| square-wave pattern | oscillation | waveform | categorical | waveform_shape | `waveform_shape = square` | schema field |
| consolidation / range-bound | oscillation | magnitude, duration | numerical, numerical | peak2peak, duration | `peak2peak` low; `duration` high | `[TA-chart]` |
| trend-with-cycle | oscillation | center-trend | categorical | center_trend | `center_trend ≠ flat` | `[LingSum]` |
| zigzag | oscillation | regularity, waveform | categorical, categorical | regularity, waveform_shape | `regularity = irregular`; `waveform_shape ≈ triangle` | `New` |
| whipsaw | oscillation | frequency, magnitude, regularity | numerical, numerical, categorical | frequency, peak2peak, regularity | `frequency` high; `peak2peak` high; `regularity = irregular` | `[TA-chart]` |

Per **Rule D**, none of these are modeled as `repeat(ramp, ramp)` compounds
— `oscillation` already owns this dimension space.

---

## 4. Compound primitives

Notation: `R[increase]` / `R[decrease]` = the ramp terms from §3.1;
`P` = a `plateau` term; `*` suffix = explicitly unconstrained slot.

### 4.1 `[increase, decrease]` and `[decrease, increase]` — transient identities

**Constituent terms** (identical shape for both, mirrored):

| Position | Term | Dimension(s) | Type | Feature(s) |
|---|---|---|---|---|
| 1 | increase (or decrease, mirrored) | direction | categorical | slope |
| 2 | decrease (or increase, mirrored) | direction | categorical | slope |

**Emergent dimensions/features** (Rule F, fixed once per identity):
`recovery` (ordinal: none/partial/full — how close `end_value` returns to
`start_value`), `sharpness` (numerical: `peak2peak / duration`),
`shape` (categorical: pointed/rounded — derived from `sharpness` and the
two legs' `curvature`), `asymmetry` (categorical: symmetric/asymmetric —
leg-1 vs. leg-2 duration).

**Term-level — `[increase, decrease]`:**

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| spike | recovery, sharpness | ordinal, numerical | recovery, sharpness | `recovery = full`; `sharpness` high | `[TA-chart]` |
| crest / hump / rounded peak | recovery, shape | ordinal, categorical | recovery, shape | `recovery = full`; `shape = rounded` | `New` |
| (upward) reversal / inverted-V | recovery | ordinal | recovery | `recovery = none` | `New` |
| overshoot | recovery, magnitude | ordinal, numerical | recovery, peak2peak | `recovery = full`; `peak2peak` small (relative to context) | `New` |
| blip / glitch (up) | sharpness, magnitude | numerical, numerical | sharpness, peak2peak | `sharpness` high; `peak2peak` small | `New` |

**Term-level — `[decrease, increase]`** (mirror):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| dip | recovery, sharpness | ordinal, numerical | recovery, sharpness | `recovery = full`; `sharpness` high | `New` |
| bowl / U-shape / cup (bare) | recovery, shape | ordinal, categorical | recovery, shape | `recovery = full`; `shape = rounded` | `[TA-chart]` (cup) |
| (downward) reversal / V-shape | recovery | ordinal | recovery | `recovery = none` | `New` |
| undershoot | recovery, magnitude | ordinal, numerical | recovery, peak2peak | `recovery = full`; `peak2peak` small | `New` |
| blip / glitch (down) | sharpness, magnitude | numerical, numerical | sharpness, peak2peak | `sharpness` high; `peak2peak` small | `New` |
| impulse / point outlier | duration | numerical | duration (total) | `duration ≈ 1` (degenerate) | `New` |

### 4.2 `[ramp, plateau]` family — level-change identities

**Emergent dimensions/features**: `level_shift` (numerical:
`end_value(after) − start_value(before)`), `direction-consistency`
(categorical, for 3-slot variants comparing leg 1 vs. leg 3 direction).

| Identity | Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|---|
| `[increase, plateau]` | step up / jump | steepness (leg 1) | numerical | slope (leg 1) | `\|slope\|` high | `[LingSum]` |
| `[decrease, plateau]` | step down / drop | steepness (leg 1) | numerical | slope (leg 1) | `\|slope\|` high | `[LingSum]` |
| `[ramp[*], plateau]` | leveling off / plateauing | curvature (leg 1) | categorical | curvature (leg 1) | `curvature = decelerating` | `[LingSum]` |
| `[plateau, increase]` | breakout | duration (leg 1) | numerical | duration (leg 1) | `duration` high (established base) | `[TA-chart]` |
| `[plateau, decrease]` | breakdown | duration (leg 1) | numerical | duration (leg 1) | `duration` high | `[TA-chart]` |
| `[ramp[dir], plateau, ramp[dir]*]` | stall / stagnation | duration (leg 2), direction-consistency | numerical, categorical | duration (leg 2) | `duration` (leg 2) low; leg 1/3 same direction | `New` |
| `[plateau, ramp[*], plateau]` | level shift / structural break | duration (legs 1, 3) | numerical | duration (legs 1, 3) | both high (established base before/after) | `[KBTA]`, `[LingSum]` |

### 4.3 Nested compounds (constituents may be compound terms)

**`[spike, gap[*], spike]`** — emergent dimension `peak_ratio` (numerical,
comparative: `crest₁.peak2peak / crest₂.peak2peak`), `spacing` (numerical:
gap duration):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| double top / double peak | peak_ratio | numerical | peak2peak (both legs) | `peak_ratio ≈ 1` | `[TA-chart]`, `[PIP/PDL]` |

**`[dip, gap[*], dip]`** (mirror) — emergent dimension `trough_ratio`:

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| double bottom / double trough | trough_ratio | numerical | peak2peak (both legs) | `trough_ratio ≈ 1` | `[TA-chart]`, `[PIP/PDL]` |

**`[spike, gap[*], spike, gap[*], spike]`** — emergent dimension
`magnitude_pattern` (categorical, comparative across 3 legs):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| triple top | magnitude_pattern | categorical | peak2peak (all legs) | `magnitude_pattern = equal-equal-equal` | `[TA-chart]` |
| head-and-shoulders | magnitude_pattern | categorical | peak2peak (all legs) | `magnitude_pattern = low-high-low` | `[TA-chart]`, `[PIP/PDL]` |

**Raw 4-ramp signatures** (not nested — continuous zigzag reading, no
baseline-return gap between legs) — emergent dimension `trough_ratio` /
`peak_ratio` as above, computed directly from the two inner extrema:

| Identity | Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|---|
| `[decrease, increase, decrease, increase]` | W-shape | trough_ratio | numerical | peak2peak (legs 1, 3) | `trough_ratio ≈ 1` | `New` |
| `[increase, decrease, increase, decrease]` | M-shape | peak_ratio | numerical | peak2peak (legs 1, 3) | `peak_ratio ≈ 1` | `New` |

**`[U-shape, decrease, increase]`** — nested (leg 1 is a compound term),
emergent dimension `pullback_ratio` (numerical: leg-2 `peak2peak` /
leg-1 `peak2peak`):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| cup-and-handle | pullback_ratio | numerical | peak2peak (legs 2, 1) | `pullback_ratio` small | `[TA-chart]` |

**Aliases (Rule D — same identity/dimensions as an existing term, kept only
for vocabulary completeness):**

| Term | Alias of | Note | Reference |
|---|---|---|---|
| rebound / bounce | dip (or `[decrease, increase]` reversal, scale-relative) | same identity, evaluated relative to a larger enclosing decline | `New` |
| recovery (pattern) | `[decrease, gap[*], increase]` | gap-separated variant of dip; `leg3.end_value ≈` pre-decline level | `New` |
| V-shape (large-scale) | (downward) reversal, §4.1 | same identity/dimensions, larger scale | `New` |

**`[ramp[dir], oscillation, ramp[dir]*]`** — emergent dimension
`direction-consistency` (categorical, legs 1 vs. 3):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| flag / pennant | direction-consistency, magnitude, duration (leg 2) | categorical, numerical, numerical | direction (legs 1,3), peak2peak (leg 2), duration (leg 2) | leg 1/3 same direction; leg 2 `peak2peak` small, `duration` short | `[TA-chart]` |

**`[transient-term, oscillation]`** (constituent 1 = any `[increase,
decrease]`/`[decrease, increase]` term, e.g. spike or step):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| ringing | envelope (leg 2) | categorical | envelope_behavior (leg 2) | `envelope_behavior = decaying` | `New` |

**`[decrease, increase, decrease]`** — emergent dimension `magnitude_ratio`
(numerical: leg 3 `peak2peak` / leg 1 `peak2peak`):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| rebound-and-fade | magnitude_ratio | numerical | peak2peak (legs 1, 3) | `magnitude_ratio < 1` | `New` |

**`[plateau[*], increase, increase, plateau]`** — emergent dimension
`curvature-sequence` (categorical over the two middle legs):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| S-curve / sigmoid growth | curvature-sequence | categorical | curvature (legs 2, 3) | `curvature-sequence = convex-then-concave` | `New` |

**`[ramp[*], ramp[dir]]`** — emergent dimension `steepness_jump`
(numerical: leg 2 `\|slope\|` / leg 1 `\|slope\|`):

| Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|
| hockey-stick | steepness_jump | numerical | slope (legs 1, 2) | `steepness_jump` high | `New` |

### 4.4 Repetition compounds (`repeat(...)`, Rule C-(b))

Reserved for repeated structures **not** already covered by
`oscillation`'s own dimensions — the repeated unit is directional (net
drift) or gapped by plateaus, not an alternating continuous waveform.

Emergent dimensions: `n_repeats` (numerical), `step-direction-consistency`
(categorical), `magnitude-trend-across-repeats` (categorical: increasing /
decreasing / stable).

| Identity | Term | Relevant dimension(s) | Type | Relevant feature(s) | New constraint(s) | Reference |
|---|---|---|---|---|---|---|
| `repeat([increase, plateau], n)` | staircase (ascending) | n_repeats, step-direction-consistency | numerical, categorical | — | `n_repeats ≥ 3`; all steps same direction | `New` |
| `repeat([decrease, plateau], n)` | staircase (descending) | n_repeats, step-direction-consistency | numerical, categorical | — | `n_repeats ≥ 3`; all steps same direction | `New` |
| `repeat([decrease, plateau], n)` | cascading decline | n_repeats, magnitude-trend-across-repeats | numerical, categorical | peak2peak (per repeat) | same as staircase-descending, plus step magnitudes vary | `New` (alias, Rule D) |
| `repeat([spike \| dip, plateau[*]], n)` | burst / spike-train | n_repeats, inter-element gap | numerical, numerical | duration (gap) | `n_repeats ≥ 3`; gap `duration` short | `New` |

---

## 5. Global features (of the `global` pseudo-primitive)

Global descriptors are terms whose subject is the whole signal — same
`(dimension, constraint)` mechanism as §3, applied to `global` instead of a
basic primitive.

### 5.1 Signal-statistic globals (already in `feature_schema.yaml`)

`min`, `max`, `mean`, `std`, `peak2peak`, `frequency`, `histogram_peak` —
dimension = feature, 1:1, numerical.

### 5.2 Structural/compositional globals (new `global.*` fields needed)

| Term | Primitive | Dimension(s) | Type | Relevant feature(s) | Constraint(s) | Reference |
|---|---|---|---|---|---|---|
| trend (overall) | global | net-direction | categorical | (new: net direction across whole signal) | — | `[KBTA]`, `[LingSum]` |
| monotonicity | global | reversal-count | numerical | (new: count of direction reversals among ramps) | `reversal-count = 0` | `[KBTA]` |
| volatility | global | volatility | numerical | (new: aggregate short-timescale variability) | high/low | `[LingSum]` |
| stability / stationarity | global | volatility, level-consistency | numerical, categorical | (new) | volatility low; level ≈ constant | `[KBTA]`, `New` |
| cyclicality | global | oscillation-fraction | numerical | (new: fraction of duration inside `oscillation`) | high | `[TA-IEPF]`, `[BEDTime/CaTS]` |
| choppiness | global | reversal-rate | numerical | (new: reversals per unit time, no stable period) | high | `New` |
| smoothness / noisiness | global | noisiness | numerical | noise_sigma | low / high | `New` |
| regime change | global | regime-shift | categorical | (new: change in feature distribution across a boundary) | present | `[KBTA]` |
| mean reversion | global | reversion-tendency | categorical | (new) | present | `New` |
| momentum / persistence | global | direction-autocorrelation | numerical | (new) | high | `New` |
| intermittency | global | activity-sparsity | categorical | (new: long `plateau` stretches + isolated `burst`s) | present | `New` |

---

## 6. Summary

| Category | Count |
|---|---|
| Basic primitives | 3 |
| Basic-primitive terms | 29 |
| Compound identities | 20 |
| Compound terms (incl. aliases) | 30 |
| Global terms | 18 (7 signal-statistic + 11 structural) |
| **Total terms documented** | **89** |

## 7. Changelog from v2 to v3

1. **"Family" renamed to "dimension"**, and decoupled from "feature":
   a dimension is a named, typed (numerical/ordinal/categorical) axis that
   may be 1:1 with a feature, a derived view over one feature (`direction`/
   `steepness` both over `slope`), a function over several features
   directly (`sharpness` over `peak2peak` and `duration`), or — new in v3 —
   a **comparative** view across constituents (`peak_ratio`), which only
   exists at the compound level.
2. **Conjunction is no longer a special rule.** A term is always "a node
   identity + a list of `(dimension: constraint)` pairs"; single- and
   multi-dimension terms use the same representation (old Rule A′ is
   folded into Rule A).
3. **Compound identity has no display name** — always the literal ordered
   constituent-term list, never a label like "crest transient" (dropped
   from v2).
4. **Two-level table structure per compound**: constituent terms (with
   their own dimensions/features, possibly cross-referencing another
   compound term for nested cases) vs. term-level rows (new
   dimensions/constraints only — base constraints from constituents are
   not repeated).
5. **Nesting is explicit and used**: `double-top`/`head-and-shoulders` are
   built from `spike`/`dip` compound terms as constituents;
   `cup-and-handle` nests `U-shape`; `flag` nests an `oscillation` term.
6. **Wildcard constituent slots** (`ramp[*]`, `plateau[*]`, `gap[*]`)
   introduced for positions whose value is genuinely irrelevant to a
   term's meaning (e.g. `leveling off`'s incoming ramp direction), always
   explicitly marked rather than omitted.
7. **Dimension type column added everywhere** (numerical/ordinal/
   categorical), including for global terms.

## 8. Open items for implementation

1. `feature_schema.yaml` needs new `derived` fields for compound-level
   emergent dimensions not already present as `spike`/`dip` fields today
   (`shape`, plus comparative dimensions like `peak_ratio`,
   `trough_ratio`, `pullback_ratio`, `steepness_jump`, `magnitude_ratio`,
   `magnitude_pattern` for nested/multi-constituent compounds).
2. Comparative dimensions (§4.3) require the schema/constraint engine to
   reference **two different constituents' same feature** within one
   compound — check whether `constraints.py`'s existing `comparable`
   machinery (currently peer-features across template positions or vs.
   `global.*`) already covers this, or needs extension to same-compound
   cross-leg comparisons.
3. `global.*` needs the structural/compositional fields listed in §5.2 —
   none exist in the schema today; several (`net-direction`,
   `reversal-count`, `regime-shift`) require scanning the whole
   primitive-type sequence, not just raw sample statistics.
4. The `repeat(sub-sequence, n)` construct (§4.4) has no analogue in
   `layout.py`/`constraints.py` yet — would need either a new
   layout-planning construct or a post hoc pattern-matching pass (similar
   to `post_match.py`'s window search) over an already-realized sequence.
5. Wildcard slot semantics need a concrete generation-time rule: does a
   `ramp[*]` constituent get realized with a uniformly sampled direction,
   or does it inherit whatever value keeps neighboring change-points
   valid (per `validate_slope` in `orchestrate.py`)? This affects sampling
   weight, not just documentation.
6. Validate emergent-dimension thresholds (what counts as "high"
   `sharpness`, `recovery = full` vs. `partial`) empirically against a
   mined vocabulary corpus (e.g. BEDTime-style descriptions) rather than
   asserted cutoffs.
