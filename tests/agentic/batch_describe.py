"""Batch NL-description generation over all JSON files in tests/out/batch/.

Usage:
    uv run python -m tests.agentic.batch_describe
    uv run python -m tests.agentic.batch_describe --plot
    uv run python -m tests.agentic.batch_describe --in-dir tests/out/batch \
        --out-dir tests/agentic/out --workers 4

    # Run only on the validation split listed in a train/test split file
    # (the split file's `data_dir` is used to locate the raw, un-described
    # inputs; `--in-dir` is ignored in this mode):
    uv run python -m tests.agentic.batch_describe \
        --split-file train_test_split.json --split val \
        --out-dir output/description_gpt4_1 --workers 4

Each sample_XXXXXX.json in --in-dir produces:
    <out-dir>/sample_XXXXXX_described.json
    <out-dir>/plots/sample_XXXXXX/global.png   (only with --plot)
    <out-dir>/plots/sample_XXXXXX/event.png    (only with --plot)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nl_describer.config import Config
from nl_describer.pipeline import describe_sample
from nl_describer.verifier import matches_format

logger = logging.getLogger("nl_describer.batch")


def _process_one(
    in_path: Path,
    out_dir: Path,
    plots_dir: Path | None,
    cfg: Config,
) -> dict:
    """Run describe_sample for one file; return a result summary dict."""
    stem = in_path.stem
    out_path = out_dir / f"{stem}_described.json"
    sample_plots = plots_dir / stem if plots_dir is not None else None

    try:
        seed = int(stem.split("_")[-1]) if stem.split("_")[-1].isdigit() else 0
        result = describe_sample(in_path, cfg, plots_dir=sample_plots, seed=seed)
        with open(out_path, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

        # Quick validation
        issues: list[str] = []
        for d in result.descriptions:
            if not matches_format(d.text):
                issues.append(f"format mismatch: {d.text[:60]!r}")
        verdicts = {
            v: sum(1 for d in result.descriptions if d.critic_verdict == v)
            for v in ("accept", "borderline", "reject", None)
        }
        return {
            "sample": stem,
            "status": "ok",
            "n_descriptions": len(result.descriptions),
            "verdicts": verdicts,
            "validation_issues": issues,
            "out_path": str(out_path),
        }
    except Exception as e:
        return {
            "sample": stem,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch NL-description generation for all samples in a directory."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("tests/out/batch"),
        help="Directory of BasicOutput JSON files (default: tests/out/batch). "
        "Ignored when --split-file is given.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to a train/test split JSON (e.g. train_test_split.json) with "
        "fields `data_dir`, `train_ids`, `val_ids`. When given, the input "
        "directory is taken from `data_dir` and only the ids listed under "
        "--split are processed.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Which id list to use from --split-file (default: val)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/agentic/out"),
        help="Output directory for described JSONs (default: tests/agentic/out)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate global.png and event.png for each sample",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Base directory for plots (default: <out-dir>/plots)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1 — sequential; increase carefully due to rate limits)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to nl_describer/config.yaml (defaults to package default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose _described.json already exists in --out-dir",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    cfg_path = args.config or Config.default_path()
    cfg = Config.load(cfg_path)

    out_dir: Path = args.out_dir
    plots_dir: Path | None = (
        (args.plot_dir or out_dir / "plots") if args.plot else None
    )

    if args.split_file is not None:
        if not args.split_file.exists():
            logger.error("--split-file does not exist: %s", args.split_file)
            return 1
        with open(args.split_file) as f:
            split = json.load(f)
        in_dir = Path(split["data_dir"])
        ids = split[f"{args.split}_ids"]
        logger.info(
            "Using split-file %s: data_dir=%s, split=%s, n_ids=%d",
            args.split_file,
            in_dir,
            args.split,
            len(ids),
        )
        all_inputs = [in_dir / f"{sample_id}.json" for sample_id in ids]
        missing = [p for p in all_inputs if not p.exists()]
        if missing:
            logger.warning(
                "%d/%d listed ids have no input file (e.g. %s); skipping them",
                len(missing),
                len(all_inputs),
                missing[0],
            )
            all_inputs = [p for p in all_inputs if p.exists()]
        all_inputs.sort()
    else:
        in_dir = args.in_dir
        if not in_dir.exists():
            logger.error("--in-dir does not exist: %s", in_dir)
            return 1
        all_inputs = sorted(in_dir.glob("*.json"))

    if not all_inputs:
        logger.error("No input JSON files found")
        return 1

    if args.skip_existing:
        all_inputs = [
            p for p in all_inputs
            if not (out_dir / f"{p.stem}_described.json").exists()
        ]
        logger.info("After skip-existing filter: %d samples to process", len(all_inputs))

    out_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Processing %d samples | workers=%d | plot=%s",
        len(all_inputs),
        args.workers,
        args.plot,
    )

    results: list[dict] = []

    if args.workers <= 1:
        for p in all_inputs:
            logger.info("→ %s", p.name)
            r = _process_one(p, out_dir, plots_dir, cfg)
            results.append(r)
            _print_result(r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, p, out_dir, plots_dir, cfg): p
                for p in all_inputs
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                _print_result(r)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    print(f"\n=== Batch complete: {ok}/{len(results)} ok, {err} errors ===")
    if err:
        for r in results:
            if r["status"] == "error":
                print(f"  FAILED {r['sample']}: {r['error']}")
    return 0 if err == 0 else 1


def _print_result(r: dict) -> None:
    if r["status"] == "error":
        logger.error("FAILED %s: %s", r["sample"], r["error"])
        return
    v = r.get("verdicts", {})
    issues = r.get("validation_issues", [])
    logger.info(
        "OK %-20s  n=%d  accept=%s borderline=%s reject=%s%s",
        r["sample"],
        r["n_descriptions"],
        v.get("accept", 0),
        v.get("borderline", 0),
        v.get("reject", 0),
        f"  WARN {issues}" if issues else "",
    )


if __name__ == "__main__":
    sys.exit(main())
