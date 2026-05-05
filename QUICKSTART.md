# Quick Start

## Requirements

```bash
uv sync
```

## Run the full pipeline (steps 1–19)

```bash
uv run python -m tests.dump_grounding --seed 0
```

Writes two files:

| File | Contents |
|---|---|
| `tests/out/grounding_seed_0.json` | Basic output (signal, grounding, event spans) + diagnostics |
| `tests/out/grounding_seed_0.png` | Annotated plot of the z-normed signal |

Try a few seeds to see different layouts:

```bash
uv run python -m tests.dump_grounding --seed 7
uv run python -m tests.dump_grounding --seed 42
```

## Use the pipeline in code

```python
import numpy as np
from data_gen.config import Config
from data_gen.pipeline import run_steps_1_to_19
from data_gen.schema import Schema

rng    = np.random.default_rng(0)
cfg    = Config.load("config.yaml")
schema = Schema.load("feature_schema.yaml")

result = run_steps_1_to_19(cfg, schema, rng=rng)

out = result.basic_output
print(out.signal)        # list[float], length = cfg.signal_length
print(out.event_spans)   # list[(start, end)]  — one per event occurrence
print(out.grounding)     # list[GroundedPrimitiveOut] — one per in-event template position
```

### `BasicOutput` structure

```
BasicOutput
├── signal          list[float]           z-normed signal (length = signal_length)
├── event_spans     list[(int, int)]      [start, end] of each event occurrence
└── grounding       list[GroundedPrimitiveOut]
    ├── template_index  int               position in the event template (0-based)
    ├── type            PrimitiveType     spike | dip | increase | decrease | oscillation | plateau
    ├── constraints     list[Constraint]  sampled constraints for this position
    └── spans           list[(int, int)]  one [start, end] per event occurrence
```

### Additional results on `FinalPipelineResult`

| Field | Type | Description |
|---|---|---|
| `basic_output` | `BasicOutput` | Main output (see above) |
| `verification` | `list[OccurrenceVerification]` | Per-(occurrence, constraint) pass/fail |
| `jitter_report` | `JitterReport` | Slope-retry fallbacks during jitter |
| `z_signal` | `list[float]` | Same as `basic_output.signal` |
| `jittered_features` | `NormedFeatures` | Re-extracted features after jitter |
| `extended` | `ExtendedPipelineResult` | Steps 1–16 intermediate state |

## Configuration

| File | Controls |
|---|---|
| `config.yaml` | Signal length, event count, primitive count, peak-to-peak ranges, constraint counts, jitter magnitude |
| `feature_schema.yaml` | Per-primitive-type feature scopes, pdfs, salience weights, constraint eligibility |

Key knobs in `config.yaml`:

```yaml
signal_length: 512          # total samples
nb_event_occurrence: [1, 10] # how many times the template repeats
nb_primitives_in_event: [1, 10]
jitter_randomness: 0.1      # 0 = identical occurrences, 1 = full-range jitter
nb_constraints: [0, 3]      # constraints sampled per template position
```

## Batch generation with TensorBoard

Generate N samples (seeds 0 to N−1) and log statistics to TensorBoard:

```bash
uv run python -m tests.batch_generate --n 100
tensorboard --logdir tests/out/tb
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--n` | *(required)* | Number of samples to generate |
| `--tb-dir` | `tests/out/tb` | TensorBoard log directory |
| `--no-json` | off | Skip writing per-sample JSON files |
| `--prior-k` | 200 | Empirical prior sample count |
| `--jitter-max-attempts` | 200 | Max retries per jitter fallback |

**What gets logged:**

- *Scalars* (per seed): event count, event length, template size and per-type primitive counts, inter-event primitive counts, constraint breakdown (numerical / comparable / self), verification pass rate, jitter fallback count, and mean/min/max/std for every numerical feature of every primitive type (in-event and inter-event)
- *Categorical counters* (per seed): per-value counts for string features such as `recovery_type`, `shape`, `waveform_shape`, `envelope_behavior`
- *Histograms* (running, logged every 10 seeds): distributions of event length, template size, constraint info/salience, all primitive features, verification pass rate, jitter fallbacks

## Pipeline steps at a glance

| Steps | Function | Output |
|---|---|---|
| 1–11 | `run_steps_1_to_11` | Layout + one realized signal |
| 1–16 | `run_steps_1_to_16` | + empirical priors + constraints |
| 1–19 | `run_steps_1_to_19` | + N jittered occurrences + basic output |
