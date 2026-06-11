"""Batch data generation with TensorBoard logging.

Usage:
    uv run python -m tests.batch_generate --n 100
    uv run python -m tests.batch_generate --n 100 --n-start 200
    tensorboard --logdir tests/out/tb
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from tensorboardX import SummaryWriter

from data_gen.config import Config
from data_gen.pipeline import FinalPipelineResult, PipelineResult, run_steps_1_to_19
from data_gen.schema import Schema
from data_gen.serialization import sanitize
from data_gen.types import PrimitiveType

_ALL_TYPES = list(PrimitiveType)
_TYPE_NAMES = [t.value for t in _ALL_TYPES]

# Histogram logging frequency (log running histogram every N seeds)
_HIST_FREQ = 10


def _is_finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (
        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    )


def _collect(result: FinalPipelineResult) -> dict[str, Any]:
    """Extract all loggable statistics from a single pipeline result."""
    base = result.extended.base
    layout = base.layout
    template_prims = base.event_template.primitives
    nb_primitives = len(template_prims)
    template_types = [p.type for p in template_prims]

    # --- Template type counts ---
    type_counts = {t.value: 0 for t in _ALL_TYPES}
    for p in template_prims:
        type_counts[p.type.value] += 1

    # --- Constraints ---
    n_numerical = n_comparable = n_self = 0
    constraints_per_pos: list[int] = []
    constraint_info_values: list[float] = []
    constraint_salience_values: list[float] = []
    for gp in result.extended.constraints.grounded:
        constraints_per_pos.append(len(gp.constraints))
        for c in gp.constraints:
            if c.kind == "numerical":
                n_numerical += 1
            elif c.kind == "comparable":
                n_comparable += 1
            elif c.kind == "self":
                n_self += 1
            constraint_info_values.append(c.info)
            constraint_salience_values.append(c.salience)
    total_constraints = n_numerical + n_comparable + n_self
    avg_constraints_per_pos = (
        total_constraints / nb_primitives if nb_primitives > 0 else 0.0
    )

    # --- Verification ---
    n_total = len(result.verification)
    n_passed = sum(1 for v in result.verification if v.satisfied is True)
    n_failed = sum(1 for v in result.verification if v.satisfied is False)
    n_unevaluable = sum(1 for v in result.verification if v.satisfied is None)
    pass_rate = n_passed / n_total if n_total > 0 else 1.0

    # --- Jitter ---
    fallbacks = result.jitter_report.fallbacks
    jitter_attempts = [fb.attempts for fb in fallbacks] if fallbacks else []

    # --- Feature values per primitive type ---
    # Numerical features accumulate as lists for histogram data
    in_event_num: dict[str, dict[str, list[float]]] = {
        t.value: defaultdict(list) for t in _ALL_TYPES
    }
    in_event_cat: dict[str, dict[str, dict[str, int]]] = {
        t.value: defaultdict(lambda: defaultdict(int)) for t in _ALL_TYPES
    }

    for occ_feats in result.jittered_features.in_event:
        for pos, feats in enumerate(occ_feats):
            if pos >= len(template_types):
                continue
            tname = template_types[pos].value
            for feat_name, feat_val in feats.items():
                if _is_finite(feat_val):
                    in_event_num[tname][feat_name].append(float(feat_val))
                elif isinstance(feat_val, str):
                    in_event_cat[tname][feat_name][feat_val] += 1

    inter_event_num: dict[str, dict[str, list[float]]] = {
        t.value: defaultdict(list) for t in _ALL_TYPES
    }
    inter_event_cat: dict[str, dict[str, dict[str, int]]] = {
        t.value: defaultdict(lambda: defaultdict(int)) for t in _ALL_TYPES
    }

    for slot_idx, slot_feats in enumerate(result.jittered_features.inter_event):
        if slot_idx >= len(layout.inter_event_types):
            continue
        for pos, feats in enumerate(slot_feats):
            if pos >= len(layout.inter_event_types[slot_idx]):
                continue
            tname = layout.inter_event_types[slot_idx][pos].value
            for feat_name, feat_val in feats.items():
                if _is_finite(feat_val):
                    inter_event_num[tname][feat_name].append(float(feat_val))
                elif isinstance(feat_val, str):
                    inter_event_cat[tname][feat_name][feat_val] += 1

    # --- Global signal features ---
    globals_ = result.jittered_features.globals

    return {
        # Scalar metrics (one value per sample)
        "scalars": {
            # Event structure
            "events/nb_events": layout.nb_events,
            "events/event_length": layout.event_length,
            # Template composition
            "template/nb_primitives": nb_primitives,
            **{f"template/count_{t}": type_counts[t] for t in _TYPE_NAMES},
            # Inter-event
            "inter_event/total_primitives": sum(layout.inter_event_counts),
            "inter_event/nb_slots": len(layout.inter_event_counts),
            "inter_event/avg_per_slot": (
                sum(layout.inter_event_counts) / len(layout.inter_event_counts)
                if layout.inter_event_counts
                else 0.0
            ),
            # Signal statistics (pre-normalization scale)
            "signal/raw_mean": result.z_mean,
            "signal/raw_std": result.z_std,
            # Constraint breakdown
            "constraints/total": total_constraints,
            "constraints/numerical": n_numerical,
            "constraints/comparable": n_comparable,
            "constraints/self": n_self,
            "constraints/avg_per_position": avg_constraints_per_pos,
            # Verification
            "verification/total": n_total,
            "verification/passed": n_passed,
            "verification/failed": n_failed,
            "verification/unevaluable": n_unevaluable,
            "verification/pass_rate": pass_rate,
            # Jitter
            "jitter/fallback_count": len(fallbacks),
            "jitter/avg_attempts_on_fallback": (
                float(np.mean(jitter_attempts)) if jitter_attempts else 0.0
            ),
            # Global signal
            "signal_global/mean": globals_.get("mean", 0.0),
            "signal_global/std": globals_.get("std", 0.0),
            "signal_global/peak2peak": globals_.get("peak2peak", 0.0),
            "signal_global/rms": globals_.get("rms", 0.0),
        },
        # Per-primitive-type numerical feature values (for histograms + mean scalars)
        "in_event_num": in_event_num,
        "in_event_cat": in_event_cat,
        "inter_event_num": inter_event_num,
        "inter_event_cat": inter_event_cat,
        # Raw arrays for running histograms
        "raw": {
            "nb_events": layout.nb_events,
            "event_length": layout.event_length,
            "nb_primitives": nb_primitives,
            "inter_event_total": sum(layout.inter_event_counts),
            "constraints_total": total_constraints,
            "constraints_per_pos": list(constraints_per_pos),
            "constraint_info": list(constraint_info_values),
            "constraint_salience": list(constraint_salience_values),
            "pass_rate": pass_rate,
            "fallback_count": len(fallbacks),
            "jitter_attempts": list(jitter_attempts),
        },
    }


def _log_scalars(
    writer: SummaryWriter,
    stats: dict[str, Any],
    step: int,
) -> None:
    for tag, value in stats["scalars"].items():
        writer.add_scalar(tag, value, global_step=step)

    # Per-type feature mean/min/max (from in-event jittered primitives)
    for tname, feat_dict in stats["in_event_num"].items():
        for feat_name, vals in feat_dict.items():
            if not vals:
                continue
            arr = np.array(vals)
            writer.add_scalar(
                f"features/in_event/{tname}/{feat_name}/mean",
                float(np.mean(arr)),
                global_step=step,
            )
            writer.add_scalar(
                f"features/in_event/{tname}/{feat_name}/min",
                float(np.min(arr)),
                global_step=step,
            )
            writer.add_scalar(
                f"features/in_event/{tname}/{feat_name}/max",
                float(np.max(arr)),
                global_step=step,
            )
            writer.add_scalar(
                f"features/in_event/{tname}/{feat_name}/std",
                float(np.std(arr)),
                global_step=step,
            )

    # Categorical features: log per-category counts
    for tname, cat_dict in stats["in_event_cat"].items():
        for feat_name, cat_counts in cat_dict.items():
            for cat_val, cnt in cat_counts.items():
                writer.add_scalar(
                    f"features/in_event/{tname}/{feat_name}/{cat_val}_count",
                    cnt,
                    global_step=step,
                )

    # Inter-event feature mean/min/max
    for tname, feat_dict in stats["inter_event_num"].items():
        for feat_name, vals in feat_dict.items():
            if not vals:
                continue
            arr = np.array(vals)
            writer.add_scalar(
                f"features/inter_event/{tname}/{feat_name}/mean",
                float(np.mean(arr)),
                global_step=step,
            )
            writer.add_scalar(
                f"features/inter_event/{tname}/{feat_name}/std",
                float(np.std(arr)),
                global_step=step,
            )


def _log_histograms(
    writer: SummaryWriter,
    accumulated: dict[str, list[float]],
    step: int,
) -> None:
    for tag, vals in accumulated.items():
        if len(vals) < 2:
            continue
        arr = np.array(vals, dtype=np.float32)
        writer.add_histogram(tag, arr, global_step=step)


_TYPE_INITIAL = {
    PrimitiveType.SPIKE: "s",
    PrimitiveType.DIP: "d",
    PrimitiveType.INCREASE: "i",
    PrimitiveType.DECREASE: "D",
    PrimitiveType.OSCILLATION: "o",
    PrimitiveType.PLATEAU: "p",
}


def _annotate(ax: Axes, base: PipelineResult) -> None:
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    label_y_high = ymax - 0.05 * yspan
    label_y_low = ymax - 0.12 * yspan

    for i, pp in enumerate(base.raw.placed):
        s, e = pp.span
        is_event = pp.event_index is not None
        ax.axvspan(
            s, e,
            alpha=0.18,
            facecolor="tab:orange" if is_event else "lightgray",
            hatch="///" if is_event else None,
            edgecolor="tab:orange" if is_event else "gray",
            linewidth=0.0,
        )
        mid = (s + e) / 2
        y = label_y_high if i % 2 == 0 else label_y_low
        ax.text(mid, y, _TYPE_INITIAL[pp.primitive.type],
                ha="center", va="top", fontsize=7, color="black")

    for i, (s, e) in enumerate(base.raw.event_spans):
        ax.axvline(s, color="red", ls="--", lw=0.8)
        ax.axvline(e, color="red", ls="--", lw=0.8)
        ax.text(
            (s + e) / 2, ymin + 0.04 * yspan, f"event #{i}",
            ha="center", va="bottom", fontsize=8, color="red",
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.7, pad=1.5),
        )


def _legend_handles() -> list:
    return [
        mpatches.Patch(facecolor="tab:orange", alpha=0.35, hatch="///", label="in-event primitive"),
        mpatches.Patch(facecolor="lightgray", alpha=0.5, label="inter-event primitive"),
        Line2D([0], [0], color="red", ls="--", lw=0.8, label="event boundary"),
    ]


def _save_plot(result: FinalPipelineResult, out_path: Path, seed: int, cfg: "Config") -> None:
    base = result.extended.base
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax_top.plot(base.raw.signal, color="black", lw=0.9)
    ax_top.set_title(
        f"Raw concatenated signal · "
        f"signal_length={cfg.signal_length} · "
        f"N_events={base.layout.nb_events} · "
        f"event_length={base.layout.event_length} · "
        f"seed={seed}"
    )
    ax_top.set_ylabel("raw value")
    ax_top.grid(True, alpha=0.3)

    signal_to_plot = result.z_signal_noisy if result.noise_sigma > 0 else result.z_signal
    ax_bot.plot(signal_to_plot, color="black", lw=0.9)
    ax_bot.set_title("Z-normalized signal (noisy)" if result.noise_sigma > 0 else "Z-normalized signal")
    ax_bot.set_ylabel("z-score")
    ax_bot.set_xlabel("sample index")
    ax_bot.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    ax_bot.grid(True, alpha=0.3)

    for ax in (ax_top, ax_bot):
        _annotate(ax, base)

    ax_top.legend(handles=_legend_handles(), loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch data generation with TensorBoard logging"
    )
    parser.add_argument("--n", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--n-start", type=int, default=0, help="Starting seed number (default: 0)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/out/batch"),
        help="Output directory for JSON files (skipped if --no-json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing per-sample JSON files",
    )
    parser.add_argument(
        "--tb-dir",
        type=Path,
        default=Path("tests/out/tb"),
        help="TensorBoard log directory",
    )
    parser.add_argument("--config", type=Path, default=Path("data_gen/config.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("data_gen/feature_schema.yaml"))
    parser.add_argument("--prior-k", type=int, default=200)
    parser.add_argument("--jitter-max-attempts", type=int, default=200)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot PNG for each generated sample",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("tests/out/plots"),
        help="Output directory for plot PNGs (used with --plot)",
    )
    args = parser.parse_args()

    args.tb_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_json:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(args.config)
    schema = Schema.load(args.schema)
    writer = SummaryWriter(str(args.tb_dir))

    # Accumulators for running histograms
    # Structure: tag -> list of float values
    accumulated: dict[str, list[float]] = defaultdict(list)

    t0 = time.monotonic()
    n_failed_total = 0
    seed_end = args.n_start + args.n - 1

    print(f"Generating {args.n} samples, seeds {args.n_start}..{seed_end}")
    print(f"TensorBoard: tensorboard --logdir {args.tb_dir}")

    for seed in range(args.n_start, args.n_start + args.n):
        seed_t0 = time.monotonic()
        rng = np.random.default_rng(seed)

        try:
            result = run_steps_1_to_19(
                cfg,
                schema,
                rng=rng,
                prior_K=args.prior_k,
                jitter_max_attempts=args.jitter_max_attempts,
            )
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {exc}")
            n_failed_total += 1
            writer.add_scalar("batch/error_count", n_failed_total, global_step=seed)
            continue

        stats = _collect(result)

        # --- Save JSON output ---
        if not args.no_json:
            out_path = args.out_dir / f"sample_{seed:06d}.json"
            out_path.write_text(
                json.dumps(sanitize(result.basic_output.model_dump(mode="json")), indent=2)
            )

        # --- Save plot ---
        if args.plot:
            _save_plot(result, args.plot_dir / f"sample_{seed:06d}.png", seed, cfg)

        # --- Log per-sample scalars ---
        _log_scalars(writer, stats, step=seed)

        # --- Accumulate histogram data ---
        raw = stats["raw"]
        accumulated["hist/events/nb_events"].append(raw["nb_events"])
        accumulated["hist/events/event_length"].append(raw["event_length"])
        accumulated["hist/template/nb_primitives"].append(raw["nb_primitives"])
        accumulated["hist/inter_event/total_primitives"].append(raw["inter_event_total"])
        accumulated["hist/constraints/total"].append(raw["constraints_total"])
        accumulated["hist/constraints/info"].extend(raw["constraint_info"])
        accumulated["hist/constraints/salience"].extend(raw["constraint_salience"])
        accumulated["hist/constraints/per_position"].extend(raw["constraints_per_pos"])
        accumulated["hist/verification/pass_rate"].append(raw["pass_rate"])
        accumulated["hist/jitter/fallback_count"].append(raw["fallback_count"])
        if raw["jitter_attempts"]:
            accumulated["hist/jitter/attempts_on_fallback"].extend(raw["jitter_attempts"])

        for tname, feat_dict in stats["in_event_num"].items():
            for feat_name, vals in feat_dict.items():
                accumulated[f"hist/features/in_event/{tname}/{feat_name}"].extend(vals)

        for tname, feat_dict in stats["inter_event_num"].items():
            for feat_name, vals in feat_dict.items():
                accumulated[f"hist/features/inter_event/{tname}/{feat_name}"].extend(vals)

        # --- Log running histograms every _HIST_FREQ seeds ---
        is_last = seed == args.n_start + args.n - 1
        seed_local = seed - args.n_start
        if (seed_local + 1) % _HIST_FREQ == 0 or is_last:
            _log_histograms(writer, accumulated, step=seed)

        # --- Progress counter ---
        elapsed = time.monotonic() - t0
        seed_elapsed = time.monotonic() - seed_t0
        nb_events = stats["scalars"]["events/nb_events"]
        nb_prims = stats["scalars"]["template/nb_primitives"]
        pass_rate = stats["scalars"]["verification/pass_rate"]
        fallbacks = stats["scalars"]["jitter/fallback_count"]
        writer.add_scalar("batch/samples_generated", seed + 1, global_step=seed)
        writer.add_scalar("batch/elapsed_seconds", elapsed, global_step=seed)
        writer.add_scalar("batch/seed_time_seconds", seed_elapsed, global_step=seed)

        print(
            f"[{seed_local + 1:>{len(str(args.n))}}/{args.n}] seed={seed:>6} | "
            f"events={nb_events} prims={nb_prims} | "
            f"pass={pass_rate:.2%} fallbacks={fallbacks} | "
            f"{seed_elapsed:.2f}s  (total {elapsed:.1f}s)"
        )

    writer.close()

    total_elapsed = time.monotonic() - t0
    n_ok = args.n - n_failed_total
    print(
        f"\nDone: {n_ok}/{args.n} succeeded in {total_elapsed:.1f}s "
        f"({total_elapsed / max(args.n, 1):.2f}s/sample)"
    )
    print(f"TensorBoard: tensorboard --logdir {args.tb_dir}")


if __name__ == "__main__":
    main()
