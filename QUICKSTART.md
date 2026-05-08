# Quick Start

## Requirements

```bash
uv sync
```

---

## Data generation

### Run the full pipeline (steps 1–19)

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

### Use the pipeline in code

```python
import numpy as np
from data_gen.config import Config
from data_gen.pipeline import run_steps_1_to_19
from data_gen.schema import Schema

rng    = np.random.default_rng(0)
cfg    = Config.load("data_gen/config.yaml")
schema = Schema.load("data_gen/feature_schema.yaml")

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

### Configuration

| File | Controls |
|---|---|
| `data_gen/config.yaml` | Signal length, event count, primitive count, peak-to-peak ranges, constraint counts, jitter magnitude |
| `data_gen/feature_schema.yaml` | Per-primitive-type feature scopes, pdfs, salience weights, constraint eligibility |

Key knobs in `data_gen/config.yaml`:

```yaml
signal_length: 512          # total samples
nb_event_occurrence: [1, 10] # how many times the template repeats
nb_primitives_in_event: [1, 10]
jitter_randomness: 0.1      # 0 = identical occurrences, 1 = full-range jitter
nb_constraints: [0, 3]      # constraints sampled per template position
```

### Batch generation with TensorBoard

Generate N samples and log statistics to TensorBoard:

```bash
uv run python -m tests.batch_generate --n 100
uv run python -m tests.batch_generate --n 100 --plot        # also saves a PNG per sample
uv run python -m tests.batch_generate --n 100 --n-start 200 # seeds 200–299
tensorboard --logdir tests/out/tb
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--n` | *(required)* | Number of samples to generate |
| `--n-start` | `0` | Starting seed (seeds run from `n_start` to `n_start + n − 1`) |
| `--out-dir` | `tests/out/batch` | Output directory for per-sample JSON files |
| `--no-json` | off | Skip writing per-sample JSON files |
| `--tb-dir` | `tests/out/tb` | TensorBoard log directory |
| `--plot` | off | Save a signal plot PNG for each sample |
| `--plot-dir` | `tests/out/plots` | Output directory for plot PNGs (used with `--plot`) |
| `--prior-k` | 200 | Empirical prior sample count |
| `--jitter-max-attempts` | 200 | Max retries per jitter fallback |

### Pipeline steps at a glance

| Steps | Function | Output |
|---|---|---|
| 1–11 | `run_steps_1_to_11` | Layout + one realized signal |
| 1–16 | `run_steps_1_to_16` | + empirical priors + constraints |
| 1–19 | `run_steps_1_to_19` | + N jittered occurrences + basic output |

---

## NL description generation (`nl_describer`)

Uses Azure OpenAI (GPT-5.4) to produce natural-language descriptions of each
generated sample with text-to-time-series grounding.

### Requirements

Add API credentials to `.env` (already present after initial setup):

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

Configuration lives in `nl_describer/config.yaml` (temperatures, variant count,
visibility thresholds, plot sizes).

### Single sample

```bash
uv run python -m nl_describer.cli \
    --in  tests/out/batch/sample_000000.json \
    --out tests/agentic/out/sample_000000_described.json \
    --plots-dir tests/agentic/out/plots/sample_000000
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--in` | *(required)* | Path to a `BasicOutput` JSON from the data-gen pipeline |
| `--out` | *(required)* | Path to write the `DescribedSample` JSON |
| `--plots-dir` | *(none)* | If set, saves `global.png` and `event.png` here |
| `--config` | `nl_describer/config.yaml` | Override the NL-describer config |
| `--seed` | `0` | RNG seed (controls event zoom choice and observation subsets) |
| `--log-level` | `INFO` | Logging verbosity |

### Batch (all samples in `tests/out/batch/`)

```bash
uv run python -m tests.agentic.batch_describe
uv run python -m tests.agentic.batch_describe --plot          # also generate plots
uv run python -m tests.agentic.batch_describe --skip-existing # skip already-done samples
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--in-dir` | `tests/out/batch` | Directory of `BasicOutput` JSON files to process |
| `--out-dir` | `tests/agentic/out` | Output directory for `*_described.json` files |
| `--plot` | off | Generate `global.png` and `event.png` per sample |
| `--plot-dir` | `<out-dir>/plots` | Base directory for per-sample plot subdirs |
| `--workers` | `1` | Parallel threads (increase carefully — Azure rate limits apply) |
| `--skip-existing` | off | Skip samples whose output JSON already exists |
| `--config` | `nl_describer/config.yaml` | Override the NL-describer config |

### Quick two-sample test

```bash
uv run python -m tests.agentic.run_two_samples
```

Processes `sample_000000.json` and `sample_000001.json` and validates the output.

### Output schema (`DescribedSample`)

Each `*_described.json` file contains:

```
DescribedSample
├── source_path       str
└── descriptions      list[Description]
    ├── kind          "positive_base" | "positive_variant" | "negative"
    ├── text          str   — "primitive[: c, c]; primitive[: c, c]; ..."
    ├── grounding     list[GroundingChunk]
    │   ├── text_span [char_start, char_end]  — char offsets of the chunk in text
    │   └── ts_spans  [[s,e], ...]            — one TS span per event occurrence
    ├── event_spans   [[s,e], ...]            — event occurrences (empty for negatives)
    ├── applied_observations  list[str]       — e.g. ["merge@0,1->decrease"] (variant only)
    ├── critic_verdict  "accept" | "borderline" | "reject" | null
    └── critic_issues   list[str]
```

Negatives have empty `grounding` and `event_spans`. The `critic_verdict` field is
informational — all descriptions are retained regardless of verdict so downstream
consumers can filter by quality tier.
