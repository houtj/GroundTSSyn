"""
batch_run.py — Multi-process batch data generation + NL description.

Each row in batch.csv is a "sub-batch" of seeds that share the same generation
parameters.  Workers process sub-batches in parallel.  Progress is persisted in
log.csv so interrupted runs can be resumed without re-doing completed work.

Sub-batch lifecycle
-------------------
Phase 1 – data generation (tests.batch_generate logic, run in-process):
    • First PLOT_FIRST_N seeds of each sub-batch → save a plot PNG.
    • Remaining seeds → JSON only, no plot.

Phase 2 – NL description (nl_describer.cli, run as subprocess):
    • All seeds, no plots (regardless of position in sub-batch).

Existing-result check
---------------------
Before generating or describing any seed the script checks whether the output
file already exists on disk and skips it when found.  This makes the sub-batch
for seeds 0-500 (already on disk) a no-op without any special-casing in the
hot path; log.csv is still pre-filled as complete for that range.

Resume
------
Re-run the same command after an interruption.  log.csv holds per-sub-batch
counters (nb_iter_data_gen, nb_iter_nl_desc) that act as watermarks: seeds
below seed_start + nb_iter are not re-attempted.  The file-existence check
provides a second layer of protection.

Usage
-----
    uv run python batch_run.py                          # 3 workers
    uv run python batch_run.py --workers 5
    uv run python batch_run.py --batch-csv batch.csv --log log.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # must be set before any pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from data_gen.config import Config
from data_gen.pipeline import FinalPipelineResult, run_steps_1_to_19
from data_gen.schema import Schema
from data_gen.serialization import sanitize

logger = logging.getLogger("batch_run")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLOT_FIRST_N = 50           # first N seeds per sub-batch generate data-gen plots
LOG_UPDATE_EVERY = 10       # flush data-gen progress to log every N seeds

DATA_GEN_OUT = Path("tests/out/batch")
DATA_GEN_PLOT_OUT = Path("tests/out/plots")
NL_DESC_OUT = Path("tests/agentic/out")

BASE_CONFIG_PATH = Path("data_gen/config.yaml")
BASE_SCHEMA_PATH = Path("data_gen/feature_schema.yaml")

# Sub-batches already fully generated & described — pre-filled as complete in log.
PREGENERATED: list[tuple[int, int]] = [(0, 500)]

LOG_FIELDS = [
    "seed_start",
    "seed_end",
    "periodical",
    "jitter",
    "noise",
    "drift",
    "event_occurrence",
    "nb_iter_data_gen",
    "nb_iter_nl_desc",
    "list_failed_data_gen",
    "list_failed_nl_desc",
]

# ---------------------------------------------------------------------------
# batch.csv parsing helpers
# ---------------------------------------------------------------------------

def _parse_int_range(s: str) -> list[int]:
    """'2-7' → [2, 7]"""
    lo, hi = s.strip().split("-")
    return [int(lo), int(hi)]


def _parse_noise_col(s: str) -> tuple[bool, list[float]]:
    """'0' → (False, [0,0])   '0.01-0.02' → (True, [0.01, 0.02])"""
    s = s.strip()
    if s == "0":
        return False, [0.0, 0.0]
    lo_s, hi_s = s.split("-")
    return True, [float(lo_s), float(hi_s)]


def _parse_drift_col(s: str) -> tuple[bool, float]:
    """'no' → (False, 0.1 default)   '0.1' → (True, 0.1)"""
    s = s.strip()
    if s.lower() == "no":
        return False, 0.1
    return True, float(s)


def read_batch_csv(path: Path) -> list[dict[str, str]]:
    """Return list of stripped-key row dicts from batch.csv."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


def make_config_for_row(row: dict[str, str], base_cfg: Config) -> Config:
    """Return a new Config with the sub-batch overrides applied."""
    add_noise, noise_level = _parse_noise_col(row["noise"])
    drift_limit, drift_threshold = _parse_drift_col(row["drift"])
    return base_cfg.model_copy(update={
        "periodical": row["periodical"].strip().lower() == "yes",
        "jitter_randomness": float(row["jitter"]),
        "add_noise": add_noise,
        "noise_level": noise_level,
        "drift_limit": drift_limit,
        "drift_threshold": drift_threshold,
        "nb_event_occurrence": _parse_int_range(row["event_occurrence"]),
    })

# ---------------------------------------------------------------------------
# log.csv management
# ---------------------------------------------------------------------------

def _is_pregenerated(row: dict[str, str]) -> bool:
    return (int(row["seed_start"]), int(row["seed_end"])) in PREGENERATED


def init_log(log_path: Path, batch_rows: list[dict[str, str]]) -> None:
    """Write a fresh log.csv.  Pre-fill PREGENERATED ranges as complete."""
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in batch_rows:
            count = int(row["seed_end"]) - int(row["seed_start"])
            done = count if _is_pregenerated(row) else 0
            writer.writerow({
                "seed_start": row["seed_start"],
                "seed_end": row["seed_end"],
                "periodical": row["periodical"],
                "jitter": row["jitter"],
                "noise": row["noise"],
                "drift": row["drift"],
                "event_occurrence": row["event_occurrence"],
                "nb_iter_data_gen": done,
                "nb_iter_nl_desc": done,
                "list_failed_data_gen": "[]",
                "list_failed_nl_desc": "[]",
            })


def _read_log_rows(log_path: Path) -> list[dict[str, str]]:
    with open(log_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_log_rows(log_path: Path, rows: list[dict[str, str]]) -> None:
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_log_row(
    log_path: Path,
    lock: Any,      # mp.managers.AcquirerProxy
    row_idx: int,
    **fields: Any,
) -> None:
    """Atomically update fields in one row of log.csv."""
    with lock:
        rows = _read_log_rows(log_path)
        rows[row_idx].update({k: str(v) for k, v in fields.items()})
        _write_log_rows(log_path, rows)

# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _data_gen_path(seed: int) -> Path:
    return DATA_GEN_OUT / f"sample_{seed:06d}.json"


def _nl_desc_path(seed: int) -> Path:
    return NL_DESC_OUT / f"sample_{seed:06d}_described.json"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _save_plot(result: FinalPipelineResult, seed: int, cfg: Config) -> None:
    DATA_GEN_PLOT_OUT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_GEN_PLOT_OUT / f"sample_{seed:06d}.png"

    base = result.extended.base
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax_top.plot(base.raw.signal, color="black", lw=0.9)
    ax_top.set_title(
        f"Raw signal · signal_length={cfg.signal_length} · "
        f"N_events={base.layout.nb_events} · seed={seed}"
    )
    ax_top.set_ylabel("raw value")
    ax_top.grid(True, alpha=0.3)

    sig = result.z_signal_noisy if result.noise_sigma > 0 else result.z_signal
    label = "Z-normalized signal (noisy)" if result.noise_sigma > 0 else "Z-normalized signal"
    ax_bot.plot(sig, color="black", lw=0.9)
    ax_bot.set_title(label)
    ax_bot.set_ylabel("z-score")
    ax_bot.set_xlabel("sample index")
    ax_bot.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    ax_bot.grid(True, alpha=0.3)

    for ax in (ax_top, ax_bot):
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        for pp in base.raw.placed:
            s, e = pp.span
            is_event = pp.event_index is not None
            ax.axvspan(
                s, e, alpha=0.18,
                facecolor="tab:orange" if is_event else "lightgray",
                hatch="///" if is_event else None,
                edgecolor="tab:orange" if is_event else "gray",
                linewidth=0.0,
            )
        for i, (s, e) in enumerate(base.raw.event_spans):
            ax.axvline(s, color="red", ls="--", lw=0.8)
            ax.axvline(e, color="red", ls="--", lw=0.8)
            ax.text(
                (s + e) / 2, ymin + 0.04 * yspan, f"event #{i}",
                ha="center", va="bottom", fontsize=8, color="red",
                bbox=dict(facecolor="white", edgecolor="red", alpha=0.7, pad=1.5),
            )

    ax_top.legend(
        handles=[
            mpatches.Patch(facecolor="tab:orange", alpha=0.35, hatch="///", label="in-event"),
            mpatches.Patch(facecolor="lightgray", alpha=0.5, label="inter-event"),
            Line2D([0], [0], color="red", ls="--", lw=0.8, label="event boundary"),
        ],
        loc="lower right", fontsize=8, ncol=3,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_data_gen_seed(
    seed: int,
    cfg: Config,
    schema: Schema,
    save_plot: bool,
) -> bool:
    """Generate one sample in-process.  Returns True on success."""
    out_path = _data_gen_path(seed)
    if out_path.exists():
        return True

    try:
        rng = np.random.default_rng(seed)
        result = run_steps_1_to_19(cfg, schema, rng=rng)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(sanitize(result.basic_output.model_dump(mode="json")), indent=2)
        )
        if save_plot:
            _save_plot(result, seed, cfg)
        return True

    except Exception as exc:
        logger.error(f"[data_gen] seed={seed} FAILED: {exc}")
        return False

# ---------------------------------------------------------------------------
# NL description
# ---------------------------------------------------------------------------

# Matches the running-total fields printed by nl_describer.llm.structured_call:
#   [tokens] <agent>: in=X  out=Y  |  total_in=X  total_out=Y
_TOKEN_LINE_RE = re.compile(r"total_in=([\d,]+)\s+total_out=([\d,]+)")


def _parse_token_totals(stdout: str) -> tuple[int, int] | None:
    """Return (total_input, total_output) from the last [tokens] line, or None."""
    last: re.Match | None = None
    for m in _TOKEN_LINE_RE.finditer(stdout):
        last = m
    if last is None:
        return None
    return (
        int(last.group(1).replace(",", "")),
        int(last.group(2).replace(",", "")),
    )


def run_nl_desc_seed(seed: int) -> tuple[bool, int, int]:
    """Run nl_describer for one sample via subprocess.

    Returns (success, total_input_tokens, total_output_tokens).
    Token counts are 0 when the output already existed or the call failed.
    """
    out_path = _nl_desc_path(seed)
    if out_path.exists():
        return True, 0, 0

    in_path = _data_gen_path(seed)
    if not in_path.exists():
        logger.warning(f"[nl_desc] seed={seed}: data-gen output missing, skipping")
        return False, 0, 0

    cmd = [
        "uv", "run", "python", "-m", "nl_describer.cli",
        "--in", str(in_path),
        "--out", str(out_path),
        "--seed", str(seed),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        totals = _parse_token_totals(proc.stdout)
        in_tok, out_tok = totals if totals is not None else (0, 0)
        return True, in_tok, out_tok
    except subprocess.CalledProcessError as exc:
        logger.error(
            f"[nl_desc] seed={seed} FAILED (exit {exc.returncode}):\n{exc.stderr}"
        )
        return False, 0, 0
    except Exception as exc:
        logger.error(f"[nl_desc] seed={seed} FAILED: {exc}")
        return False, 0, 0

# ---------------------------------------------------------------------------
# Sub-batch worker
# ---------------------------------------------------------------------------

def process_sub_batch(
    row_idx: int,
    row: dict[str, str],
    log_path: Path,
    lock: Any,
    base_cfg: Config,
    schema: Schema,
) -> None:
    """
    Worker entry point.  Handles one sub-batch end-to-end:
      Phase 1 – data generation
      Phase 2 – NL description
    """
    # Child processes spawned by mp.Pool don't inherit the parent's logging
    # configuration, so we must set it up here in each worker.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    seed_start = int(row["seed_start"])
    seed_end = int(row["seed_end"])
    label = f"[{seed_start},{seed_end})"
    total = seed_end - seed_start

    log = logging.getLogger(f"batch_run.{seed_start}")
    log.info(f"Worker started for sub-batch {label}")

    cfg = make_config_for_row(row, base_cfg)

    # ------------------------------------------------------------------
    # Phase 1: data generation
    # ------------------------------------------------------------------
    with lock:
        log_rows = _read_log_rows(log_path)
        nb_iter_dg = int(log_rows[row_idx]["nb_iter_data_gen"])
        failed_dg: list[int] = json.loads(log_rows[row_idx]["list_failed_data_gen"])

    seeds_to_gen = range(seed_start + nb_iter_dg, seed_end)
    new_failures_dg: list[int] = []
    local_count = 0

    for seed in seeds_to_gen:
        offset = seed - seed_start          # position within sub-batch
        save_plot = offset < PLOT_FIRST_N

        ok = run_data_gen_seed(seed, cfg, schema, save_plot)
        if not ok:
            new_failures_dg.append(seed)

        nb_iter_dg += 1
        local_count += 1

        if local_count % LOG_UPDATE_EVERY == 0:
            all_failed_dg = failed_dg + new_failures_dg
            update_log_row(
                log_path, lock, row_idx,
                nb_iter_data_gen=nb_iter_dg,
                list_failed_data_gen=json.dumps(all_failed_dg),
            )
            log.info(f"data_gen {label}: {nb_iter_dg}/{total} processed")

    all_failed_dg = failed_dg + new_failures_dg
    update_log_row(
        log_path, lock, row_idx,
        nb_iter_data_gen=nb_iter_dg,
        list_failed_data_gen=json.dumps(all_failed_dg),
    )
    n_ok_dg = total - len(all_failed_dg)
    log.info(
        f"data_gen {label} complete: {n_ok_dg} ok, {len(all_failed_dg)} failed"
    )

    # ------------------------------------------------------------------
    # Phase 2: NL description
    # ------------------------------------------------------------------
    with lock:
        log_rows = _read_log_rows(log_path)
        nb_iter_nl = int(log_rows[row_idx]["nb_iter_nl_desc"])
        failed_nl: list[int] = json.loads(log_rows[row_idx]["list_failed_nl_desc"])
        # Re-read current failed_dg in case phase-1 updated it during resume
        all_failed_dg = json.loads(log_rows[row_idx]["list_failed_data_gen"])

    failed_dg_set = set(all_failed_dg)
    seeds_for_nl = range(seed_start + nb_iter_nl, seed_end)
    new_failures_nl: list[int] = []

    for seed in seeds_for_nl:
        if seed in failed_dg_set:
            # Data gen failed → nothing to describe; count as processed
            nb_iter_nl += 1
            update_log_row(
                log_path, lock, row_idx,
                nb_iter_nl_desc=nb_iter_nl,
            )
            continue

        ok, in_tok, out_tok = run_nl_desc_seed(seed)
        if not ok:
            new_failures_nl.append(seed)
        elif in_tok or out_tok:
            log.info(
                f"[nl_desc] seed={seed} done | "
                f"tokens in={in_tok:,}  out={out_tok:,}"
            )

        nb_iter_nl += 1
        all_failed_nl = failed_nl + new_failures_nl
        # NL calls are slow → flush after every seed
        update_log_row(
            log_path, lock, row_idx,
            nb_iter_nl_desc=nb_iter_nl,
            list_failed_nl_desc=json.dumps(all_failed_nl),
        )

    all_failed_nl = failed_nl + new_failures_nl
    n_ok_nl = total - len(failed_dg_set) - len(all_failed_nl)
    log.info(
        f"nl_desc {label} complete: {n_ok_nl} ok, "
        f"{len(all_failed_nl)} failed, {len(failed_dg_set)} skipped (data-gen failure)"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel worker processes (default: 3)",
    )
    parser.add_argument(
        "--batch-csv", type=Path, default=Path("batch.csv"),
        help="Path to batch config CSV (default: batch.csv)",
    )
    parser.add_argument(
        "--log", type=Path, default=Path("log.csv"),
        help="Path to progress log CSV (default: log.csv)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # ---- Load batch config ----
    batch_rows = read_batch_csv(args.batch_csv)
    logger.info(f"Loaded {len(batch_rows)} sub-batches from {args.batch_csv}")

    # ---- Init or verify log ----
    if not args.log.exists():
        logger.info(f"Creating {args.log} (pre-filling {PREGENERATED} as complete)")
        init_log(args.log, batch_rows)
    else:
        logger.info(f"Resuming from existing {args.log}")

    # ---- Identify pending sub-batches ----
    log_rows = _read_log_rows(args.log)
    if len(log_rows) != len(batch_rows):
        logger.warning(
            f"log.csv has {len(log_rows)} rows but batch.csv has {len(batch_rows)}. "
            "New rows in batch.csv will be appended to log.csv."
        )
        # Append missing rows for any new sub-batches added to batch.csv
        existing_keys = {
            (r["seed_start"].strip(), r["seed_end"].strip()) for r in log_rows
        }
        with open(args.log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            for row in batch_rows:
                key = (row["seed_start"], row["seed_end"])
                if key not in existing_keys:
                    count = int(row["seed_end"]) - int(row["seed_start"])
                    done = count if _is_pregenerated(row) else 0
                    writer.writerow({
                        "seed_start": row["seed_start"],
                        "seed_end": row["seed_end"],
                        "periodical": row["periodical"],
                        "jitter": row["jitter"],
                        "noise": row["noise"],
                        "drift": row["drift"],
                        "event_occurrence": row["event_occurrence"],
                        "nb_iter_data_gen": done,
                        "nb_iter_nl_desc": done,
                        "list_failed_data_gen": "[]",
                        "list_failed_nl_desc": "[]",
                    })
        log_rows = _read_log_rows(args.log)

    pending: list[tuple[int, dict[str, str]]] = []
    for i, (batch_row, log_row) in enumerate(zip(batch_rows, log_rows)):
        seed_start = int(batch_row["seed_start"])
        seed_end = int(batch_row["seed_end"])
        total = seed_end - seed_start
        nb_dg = int(log_row["nb_iter_data_gen"])
        nb_nl = int(log_row["nb_iter_nl_desc"])
        if nb_dg < total or nb_nl < total:
            pending.append((i, batch_row))
            logger.info(
                f"  PENDING  [{seed_start},{seed_end}): "
                f"data_gen={nb_dg}/{total}  nl_desc={nb_nl}/{total}"
            )
        else:
            logger.info(f"  complete [{seed_start},{seed_end})")

    if not pending:
        logger.info("All sub-batches are complete. Nothing to do.")
        return

    logger.info(
        f"Launching {len(pending)} sub-batch(es) across {args.workers} worker(s)"
    )

    # ---- Load base config and schema (shared, read-only) ----
    base_cfg = Config.load(BASE_CONFIG_PATH)
    schema = Schema.load(BASE_SCHEMA_PATH)

    # ---- Run workers ----
    with mp.Manager() as manager:
        lock = manager.Lock()
        # Inject the shared lock into each worker's args
        worker_args_with_lock = [
            (row_idx, batch_row, args.log, lock, base_cfg, schema)
            for row_idx, batch_row in pending
        ]
        with mp.Pool(processes=args.workers) as pool:
            pool.starmap(process_sub_batch, worker_args_with_lock)

    logger.info("All workers finished.")


if __name__ == "__main__":
    main()
