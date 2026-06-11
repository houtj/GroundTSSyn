"""Reusable evaluation harness for nl_describer token-reduction experiments.

Picks N samples from a batch directory, runs the agentic pipeline on each,
captures per-sample token usage + quality metrics, and (optionally) diffs
each result against a baseline `*_described.json` from a reference directory.

Designed to be run repeatedly with different `nl_describer` configs / code
paths to compare token-reduction proposals on a fixed sample set.

Output layout (under `--out-dir`, default `tests/agentic/eval_runs/<label>/`):

    described/sample_XXXXXX_described.json   — pipeline outputs
    plots/sample_XXXXXX/{global,event}.png   — per-sample plots
    per_sample_metrics.csv                   — one row per sample
    summary.json                             — aggregated metrics + baseline diff
    diff/sample_XXXXXX.txt                   — side-by-side text diff vs. baseline

Usage:
    # Run on 5 random samples, compare to existing baseline.
    uv run python -m tests.agentic.eval_run \
        --label post_safe_default --n 5 --seed 42

    # Reuse the same 5 samples with a different config.
    uv run python -m tests.agentic.eval_run \
        --label some_other_proposal --n 5 --seed 42 \
        --config nl_describer/config_alt.yaml

    # Compare against a custom baseline dir (defaults to tests/agentic/out/).
    uv run python -m tests.agentic.eval_run \
        --label foo --seeds 100,200,300 \
        --baseline-dir tests/agentic/out

The harness keeps random seeding deterministic on the picked sample list
(but still uses the per-sample integer seed for the pipeline), so two runs
with `--n 5 --seed 42` compare exactly the same samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from nl_describer.config import Config
from nl_describer.llm import (
    get_token_totals,
    reset_token_counters,
)
from nl_describer.pipeline import describe_sample
from nl_describer.token_limit import count_bert_tokens
from nl_describer.verifier import matches_format

logger = logging.getLogger("nl_describer.eval_run")


# ---------------------------------------------------------------------------
# Sample picking
# ---------------------------------------------------------------------------


def _sample_id_from_path(p: Path) -> str:
    """Extract `sample_NNNNNN` stem from either an input or described path."""
    stem = p.stem
    return stem.removesuffix("_described")


def pick_samples(
    in_dir: Path,
    *,
    n: int | None,
    seed: int,
    explicit_seeds: list[int] | None,
    baseline_dir: Path | None,
    require_baseline: bool,
) -> list[Path]:
    """Pick samples from `in_dir`. Reproducible given (seed, n).

    If `explicit_seeds` is given (e.g. `--seeds 0,1,2`), use those exact
    sample numbers. Otherwise, draw `n` random samples from the directory.

    When `require_baseline=True`, only consider samples that have a
    matching `<id>_described.json` file in `baseline_dir` so we always
    have something to diff against.
    """
    all_files = sorted(in_dir.glob("sample_*.json"))
    if not all_files:
        raise SystemExit(f"No sample_*.json files in {in_dir}")

    if explicit_seeds is not None:
        picked: list[Path] = []
        for s in explicit_seeds:
            cand = in_dir / f"sample_{s:06d}.json"
            if not cand.exists():
                logger.warning("explicit seed %d -> %s not found; skipping", s, cand)
                continue
            picked.append(cand)
        if not picked:
            raise SystemExit("None of the explicit --seeds resolved to existing files")
        return picked

    if require_baseline and baseline_dir is not None and baseline_dir.exists():
        baseline_ids = {
            _sample_id_from_path(p) for p in baseline_dir.glob("sample_*_described.json")
        }
        candidates = [p for p in all_files if p.stem in baseline_ids]
        if not candidates:
            raise SystemExit(
                f"No samples in {in_dir} have a baseline in {baseline_dir}; "
                "pass --no-require-baseline to relax this."
            )
    else:
        candidates = all_files

    rng = np.random.default_rng(seed)
    if n is None or n >= len(candidates):
        return candidates
    idxs = rng.choice(len(candidates), size=n, replace=False)
    idxs.sort()
    return [candidates[int(i)] for i in idxs]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


_TOKENIZE_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


def _lexical_diversity(texts: list[str]) -> float:
    """Type-token ratio over a list of texts; 1.0 = all unique words.

    Used as a coarse proxy for "is wording across descriptions diverse?"
    Returns 0.0 for an empty input.
    """
    all_tokens: list[str] = []
    for t in texts:
        all_tokens.extend(_tokenize(t))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def _mean_pairwise_jaccard(texts: list[str]) -> float:
    """Mean pairwise Jaccard distance over the bag-of-words of each text.

    Higher = variants are lexically more different from one another. Returns
    0.0 when fewer than 2 texts (no pairs to compare).
    """
    if len(texts) < 2:
        return 0.0
    bags = [set(_tokenize(t)) for t in texts]
    pairs = 0
    total = 0.0
    for i in range(len(bags)):
        for j in range(i + 1, len(bags)):
            a, b = bags[i], bags[j]
            if not a and not b:
                continue
            inter = len(a & b)
            uni = len(a | b)
            jacc = inter / uni if uni else 1.0
            total += 1.0 - jacc
            pairs += 1
    return total / pairs if pairs else 0.0


def _metrics_from_described(
    described: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    """Compute the per-sample quality metrics from a DescribedSample dict.

    Works on both the new run output and the baseline file (same schema).
    """
    descs = described.get("descriptions", [])
    n = len(descs)

    # Counts by kind.
    by_kind: dict[str, int] = {"positive_base": 0, "positive_variant": 0, "negative": 0}
    by_verdict: dict[str, int] = {"accept": 0, "borderline": 0, "reject": 0, "unjudged": 0}
    n_format_ok = 0
    n_length_ok = 0
    bert_lens: list[int] = []
    variant_texts: list[str] = []
    all_texts: list[str] = []
    n_grounded_chunks_total = 0
    n_grounded_chunks_with_ts = 0

    max_bert = cfg.length_limit.max_bert_tokens
    tk_model = cfg.length_limit.tokenizer_model

    for d in descs:
        kind = d.get("kind", "?")
        text = d.get("text", "") or ""
        all_texts.append(text)
        if kind in by_kind:
            by_kind[kind] += 1
        verdict = d.get("critic_verdict")
        if verdict in by_verdict:
            by_verdict[verdict] += 1
        else:
            by_verdict["unjudged"] += 1
        if matches_format(text):
            n_format_ok += 1
        try:
            n_bert = count_bert_tokens(text, tk_model)
        except Exception:
            n_bert = 0
        bert_lens.append(n_bert)
        if n_bert <= max_bert:
            n_length_ok += 1
        if kind == "positive_variant":
            variant_texts.append(text)
        for gc in d.get("grounding", []) or []:
            n_grounded_chunks_total += 1
            if gc.get("ts_spans"):
                n_grounded_chunks_with_ts += 1

    return {
        "n_descriptions": n,
        "n_positive_base": by_kind["positive_base"],
        "n_positive_variant": by_kind["positive_variant"],
        "n_negative": by_kind["negative"],
        "n_accept": by_verdict["accept"],
        "n_borderline": by_verdict["borderline"],
        "n_reject": by_verdict["reject"],
        "n_unjudged": by_verdict["unjudged"],
        "accept_rate": (by_verdict["accept"] / n) if n else 0.0,
        "reject_rate": (by_verdict["reject"] / n) if n else 0.0,
        "n_format_ok": n_format_ok,
        "format_ok_rate": (n_format_ok / n) if n else 0.0,
        "n_length_ok": n_length_ok,
        "length_ok_rate": (n_length_ok / n) if n else 0.0,
        "mean_bert_tokens": float(np.mean(bert_lens)) if bert_lens else 0.0,
        "max_bert_tokens": int(max(bert_lens)) if bert_lens else 0,
        "lexical_diversity_all": _lexical_diversity(all_texts),
        "lexical_diversity_variants": _lexical_diversity(variant_texts),
        "mean_pairwise_jaccard_variants": _mean_pairwise_jaccard(variant_texts),
        "n_grounded_chunks_total": n_grounded_chunks_total,
        "n_grounded_chunks_with_ts": n_grounded_chunks_with_ts,
    }


# ---------------------------------------------------------------------------
# Per-sample runner
# ---------------------------------------------------------------------------


def _seed_for_sample(sample_path: Path) -> int:
    """Same seed convention used by `batch_describe.py`."""
    parts = sample_path.stem.split("_")
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0


def _run_one_sample(
    in_path: Path,
    out_dir: Path,
    plots_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Run the pipeline once. Returns a metrics dict (always; status field
    indicates success/failure)."""
    stem = in_path.stem
    described_path = out_dir / "described" / f"{stem}_described.json"
    sample_plots = plots_dir / stem
    described_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset per-sample token counters so we can attribute usage cleanly.
    reset_token_counters()
    t0 = time.perf_counter()
    seed = _seed_for_sample(in_path)
    try:
        result = describe_sample(in_path, cfg, plots_dir=sample_plots, seed=seed)
    except Exception as e:
        in_tok, out_tok = get_token_totals()
        return {
            "sample": stem,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
    dt = time.perf_counter() - t0
    in_tok, out_tok = get_token_totals()

    # Persist described JSON exactly as the production pipeline does.
    described_dict = result.model_dump(mode="json")
    with open(described_path, "w") as f:
        json.dump(described_dict, f, indent=2)

    metrics = _metrics_from_described(described_dict, cfg)
    return {
        "sample": stem,
        "status": "ok",
        "wall_seconds": round(dt, 2),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "described_path": str(described_path),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Baseline diff
# ---------------------------------------------------------------------------


def _diff_against_baseline(
    metrics: dict[str, Any],
    baseline_path: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Compute baseline metrics + per-key delta. Adds keys with `_base` and
    `_delta` suffixes onto a copy of `metrics`."""
    if not baseline_path.exists():
        return {**metrics, "baseline": None}
    try:
        with open(baseline_path, "r") as f:
            base = json.load(f)
    except Exception as e:
        logger.warning("Could not read baseline %s: %s", baseline_path, e)
        return {**metrics, "baseline": None}
    base_metrics = _metrics_from_described(base, cfg)
    diffed: dict[str, Any] = dict(metrics)
    diffed["baseline_path"] = str(baseline_path)
    for k, v_base in base_metrics.items():
        diffed[f"{k}_base"] = v_base
        v_new = metrics.get(k)
        if isinstance(v_base, (int, float)) and isinstance(v_new, (int, float)):
            diffed[f"{k}_delta"] = v_new - v_base
    return diffed


def _write_text_diff(
    sample_id: str,
    described_path: Path,
    baseline_path: Path,
    out_dir: Path,
) -> None:
    """Side-by-side textual diff of description texts (new vs baseline) for
    human inspection."""
    if not baseline_path.exists() or not described_path.exists():
        return
    try:
        with open(described_path, "r") as f:
            new = json.load(f)
        with open(baseline_path, "r") as f:
            base = json.load(f)
    except Exception as e:
        logger.warning("diff: could not load %s/%s: %s", described_path, baseline_path, e)
        return
    diff_dir = out_dir / "diff"
    diff_dir.mkdir(parents=True, exist_ok=True)
    out_path = diff_dir / f"{sample_id}.txt"

    def fmt(d: dict) -> list[str]:
        rows: list[str] = []
        for i, desc in enumerate(d.get("descriptions", [])):
            kind = desc.get("kind", "?")
            verdict = desc.get("critic_verdict") or "-"
            text = desc.get("text", "")
            issues = desc.get("critic_issues") or []
            applied = desc.get("applied_observations") or []
            rows.append(f"[{i:>2}] kind={kind:>16}  verdict={verdict}")
            if applied:
                rows.append(f"     applied={applied}")
            rows.append(f"     {text}")
            if issues:
                rows.append(f"     issues={issues}")
        return rows

    new_rows = fmt(new)
    base_rows = fmt(base)
    with open(out_path, "w") as f:
        f.write(f"=== {sample_id} (NEW vs BASELINE) ===\n\n")
        f.write("--- NEW ---\n")
        f.write("\n".join(new_rows))
        f.write("\n\n--- BASELINE ---\n")
        f.write("\n".join(base_rows))
        f.write("\n")


# ---------------------------------------------------------------------------
# Aggregation + IO
# ---------------------------------------------------------------------------


_NUMERIC_METRIC_KEYS = (
    "input_tokens",
    "output_tokens",
    "wall_seconds",
    "n_descriptions",
    "n_positive_base",
    "n_positive_variant",
    "n_negative",
    "n_accept",
    "n_borderline",
    "n_reject",
    "n_unjudged",
    "accept_rate",
    "reject_rate",
    "format_ok_rate",
    "length_ok_rate",
    "mean_bert_tokens",
    "max_bert_tokens",
    "lexical_diversity_all",
    "lexical_diversity_variants",
    "mean_pairwise_jaccard_variants",
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    agg: dict[str, Any] = {
        "n_samples_total": len(rows),
        "n_samples_ok": len(ok_rows),
        "n_samples_error": len(rows) - len(ok_rows),
    }
    for key in _NUMERIC_METRIC_KEYS:
        vals = [r[key] for r in ok_rows if isinstance(r.get(key), (int, float))]
        agg[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
        agg[f"sum_{key}"] = float(np.sum(vals)) if vals else 0.0

    # Per-sample baseline deltas, if present.
    delta_keys = [k for k in ok_rows[0].keys() if k.endswith("_delta")] if ok_rows else []
    for k in delta_keys:
        vals = [r[k] for r in ok_rows if isinstance(r.get(k), (int, float))]
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
    return agg


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(arg: str | None) -> list[int] | None:
    if not arg:
        return None
    return [int(x) for x in arg.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run nl_describer on N samples and compare to a baseline."
    )
    p.add_argument(
        "--in-dir",
        type=Path,
        default=Path("tests/out/batch"),
        help="Directory of BasicOutput JSON files.",
    )
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("tests/agentic/out"),
        help="Directory with `*_described.json` files to diff against.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to tests/agentic/eval_runs/<label>/. "
            "The harness will NEVER write into --baseline-dir."
        ),
    )
    p.add_argument(
        "--label",
        type=str,
        required=True,
        help="Run label (becomes the eval_runs subdir name).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=5,
        help="How many samples to draw randomly (ignored if --seeds is given).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sample selection (deterministic).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated explicit sample seeds (e.g. 0,1,2). Overrides --n.",
    )
    p.add_argument(
        "--no-require-baseline",
        action="store_true",
        help="Do not restrict random picks to samples that have a baseline.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="nl_describer config path (defaults to package default).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    cfg = Config.load(args.config or Config.default_path())

    out_dir: Path = args.out_dir or (
        Path("tests/agentic/eval_runs") / args.label
    )
    if out_dir.resolve() == args.baseline_dir.resolve():
        raise SystemExit(
            "Refusing to write into --baseline-dir; pick a different --out-dir or --label."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    explicit_seeds = _parse_seeds(args.seeds)
    samples = pick_samples(
        args.in_dir,
        n=args.n,
        seed=args.seed,
        explicit_seeds=explicit_seeds,
        baseline_dir=args.baseline_dir,
        require_baseline=not args.no_require_baseline,
    )
    logger.info("Picked %d samples: %s", len(samples), [s.stem for s in samples])

    rows: list[dict[str, Any]] = []
    for sp in samples:
        logger.info("=== %s ===", sp.stem)
        row = _run_one_sample(sp, out_dir, plots_dir, cfg)
        if row.get("status") == "ok":
            baseline_path = args.baseline_dir / f"{sp.stem}_described.json"
            row = _diff_against_baseline(row, baseline_path, cfg)
            _write_text_diff(
                sp.stem,
                Path(row["described_path"]),
                baseline_path,
                out_dir,
            )
            logger.info(
                "OK %s  in=%s out=%s  acc=%d/%d  fmt_ok=%d/%d  len_ok=%d/%d  "
                "wall=%.1fs",
                sp.stem,
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("n_accept", 0),
                row.get("n_descriptions", 0),
                row.get("n_format_ok", 0),
                row.get("n_descriptions", 0),
                row.get("n_length_ok", 0),
                row.get("n_descriptions", 0),
                row.get("wall_seconds", 0.0),
            )
        else:
            logger.error("FAILED %s: %s", sp.stem, row.get("error"))
        rows.append(row)

    csv_path = out_dir / "per_sample_metrics.csv"
    summary_path = out_dir / "summary.json"
    _write_csv(rows, csv_path)
    summary = {
        "label": args.label,
        "config_path": str(args.config or Config.default_path()),
        "in_dir": str(args.in_dir),
        "baseline_dir": str(args.baseline_dir),
        "out_dir": str(out_dir),
        "selection": {
            "n": args.n,
            "seed": args.seed,
            "seeds": explicit_seeds,
            "require_baseline": not args.no_require_baseline,
        },
        "samples": [s.stem for s in samples],
        "aggregate": _aggregate(rows),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.label} complete ===")
    print(f"  per_sample_metrics: {csv_path}")
    print(f"  summary: {summary_path}")
    agg = summary["aggregate"]
    print(
        f"  ok={agg['n_samples_ok']}/{agg['n_samples_total']}  "
        f"mean in_tok={agg['mean_input_tokens']:,.0f}  "
        f"mean out_tok={agg['mean_output_tokens']:,.0f}  "
        f"mean wall={agg['mean_wall_seconds']:.1f}s"
    )
    print(
        f"  mean accept_rate={agg['mean_accept_rate']:.2f}  "
        f"mean reject_rate={agg['mean_reject_rate']:.2f}  "
        f"mean format_ok={agg['mean_format_ok_rate']:.2f}  "
        f"mean length_ok={agg['mean_length_ok_rate']:.2f}"
    )
    if "mean_n_accept_delta" in agg:
        print("  vs. baseline (positive = new is higher):")
        for k in (
            "n_descriptions",
            "n_accept",
            "n_borderline",
            "n_reject",
            "accept_rate",
            "reject_rate",
            "format_ok_rate",
            "length_ok_rate",
            "lexical_diversity_all",
            "lexical_diversity_variants",
            "mean_pairwise_jaccard_variants",
        ):
            key = f"mean_{k}_delta"
            if key in agg:
                print(f"    Δ {k:36s} = {agg[key]:+.3f}")

    return 0 if agg["n_samples_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
