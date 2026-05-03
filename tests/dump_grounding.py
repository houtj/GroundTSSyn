from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from data_gen.config import Config
from data_gen.pipeline import FinalPipelineResult, run_steps_1_to_19
from data_gen.schema import Schema
from data_gen.types import PrimitiveType


TYPE_INITIAL = {
    PrimitiveType.SPIKE: "s",
    PrimitiveType.DIP: "d",
    PrimitiveType.INCREASE: "i",
    PrimitiveType.DECREASE: "D",
    PrimitiveType.OSCILLATION: "o",
    PrimitiveType.PLATEAU: "p",
}


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value
    return obj


def _build_dump(seed: int, cfg: Config, result: FinalPipelineResult) -> dict[str, Any]:
    base = result.extended.base

    basic_output = result.basic_output.model_dump(mode="json")

    verifications = [v.model_dump(mode="json") for v in result.verification]
    summary = {
        "total": len(verifications),
        "passed": sum(1 for v in verifications if v["satisfied"] is True),
        "failed": sum(1 for v in verifications if v["satisfied"] is False),
        "unevaluable": sum(1 for v in verifications if v["satisfied"] is None),
    }

    fallbacks = [
        {
            "event_index": fb.event_index,
            "template_index": fb.template_index,
            "attempts": fb.attempts,
        }
        for fb in result.jitter_report.fallbacks
    ]

    return _sanitize(
        {
            "seed": seed,
            "config_summary": {
                "signal_length": cfg.signal_length,
                "nb_events": base.layout.nb_events,
                "event_length": base.layout.event_length,
                "nb_in_event_primitives": len(base.event_template.primitives),
                "jitter_randomness": cfg.jitter_randomness,
            },
            "basic_output": basic_output,
            "diagnostics": {
                "globals": result.jittered_features.globals,
                "z_stats": {"mean": result.z_mean, "std": result.z_std},
                "jitter_fallbacks": fallbacks,
                "verification": verifications,
                "verification_summary": summary,
            },
        }
    )


def _validate_basic_output(dump: dict[str, Any], cfg: Config) -> list[str]:
    """Programmatic sanity checks on the basic output."""
    warnings: list[str] = []
    bo = dump["basic_output"]
    sig = bo["signal"]
    if len(sig) != cfg.signal_length:
        warnings.append(
            f"signal length {len(sig)} != cfg.signal_length {cfg.signal_length}"
        )
    nb_events = dump["config_summary"]["nb_events"]
    for g in bo["grounding"]:
        if len(g["spans"]) != nb_events:
            warnings.append(
                f"template_index={g['template_index']} has {len(g['spans'])} spans "
                f"but nb_events={nb_events}"
            )
        for span in g["spans"]:
            if not (0 <= span[0] <= span[1] < cfg.signal_length):
                warnings.append(
                    f"out-of-range span {span} for template_index={g['template_index']}"
                )
    for ev_idx, span in enumerate(bo["event_spans"]):
        if not (0 <= span[0] <= span[1] < cfg.signal_length):
            warnings.append(f"out-of-range event_span[{ev_idx}]={span}")
    # Constraint integrity
    for g in bo["grounding"]:
        pos = g["template_index"]
        for c in g["constraints"]:
            if c["kind"] == "numerical":
                if c["target"]["primitive"] != pos:
                    warnings.append(
                        f"numerical constraint at pos={pos} targets "
                        f"primitive={c['target']['primitive']}"
                    )
            else:
                if c["lhs"]["primitive"] != pos:
                    warnings.append(
                        f"comparable constraint at pos={pos} lhs="
                        f"{c['lhs']['primitive']}"
                    )
    return warnings


def _palette(n: int) -> list[tuple[float, float, float, float]]:
    """Distinct colours per template position, cycling through tab colormap."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _plot(result: FinalPipelineResult, cfg: Config, seed: int, out_path: Path) -> None:
    raw = result.raw_jittered
    z_signal = result.z_signal
    template_types = [p.type for p in result.extended.base.event_template.primitives]
    nb_in_event = len(template_types)
    palette = _palette(nb_in_event)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    ax.plot(z_signal, color="black", lw=0.9, zorder=3)
    ax.axhline(0.0, color="gray", lw=0.5, alpha=0.5, zorder=1)
    ax.set_title(
        f"Final z-normed signal (steps 1-19) · "
        f"signal_length={cfg.signal_length} · "
        f"N_events={result.extended.base.layout.nb_events} · "
        f"template_size={nb_in_event} · "
        f"seed={seed}"
    )
    ax.set_xlabel("sample index")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.3, zorder=0)

    pos_in_event = 0
    last_event_index: int | None = None
    for pp in raw.placed:
        s, e = pp.span
        if pp.event_index is None:
            ax.axvspan(s, e, alpha=0.18, facecolor="lightgray", linewidth=0.0, zorder=2)
            continue
        if pp.event_index != last_event_index:
            pos_in_event = 0
            last_event_index = pp.event_index
        color = palette[pos_in_event % nb_in_event]
        ax.axvspan(
            s, e,
            alpha=0.22,
            facecolor=color,
            edgecolor=color,
            linewidth=0.0,
            zorder=2,
        )
        pos_in_event += 1

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    label_y_high = ymax - 0.05 * yspan
    label_y_low = ymax - 0.12 * yspan

    pos_in_event = 0
    last_event_index = None
    for i, pp in enumerate(raw.placed):
        s, e = pp.span
        mid = (s + e) / 2
        if pp.event_index is None:
            y = label_y_low
            ax.text(
                mid, y, TYPE_INITIAL[pp.primitive.type],
                ha="center", va="top", fontsize=7, color="dimgray",
            )
            continue
        if pp.event_index != last_event_index:
            pos_in_event = 0
            last_event_index = pp.event_index
        y = label_y_high if pos_in_event % 2 == 0 else label_y_low
        ax.text(
            mid, y, TYPE_INITIAL[pp.primitive.type],
            ha="center", va="top", fontsize=8, color="black", weight="bold",
        )
        pos_in_event += 1

    pass_by_event: dict[int, tuple[int, int, int]] = {}
    for v in result.verification:
        ev = v.event_index
        p, f, u = pass_by_event.get(ev, (0, 0, 0))
        if v.satisfied is True:
            p += 1
        elif v.satisfied is False:
            f += 1
        else:
            u += 1
        pass_by_event[ev] = (p, f, u)

    for i, (s, e) in enumerate(raw.event_spans):
        ax.axvline(s, color="red", ls="--", lw=0.8, zorder=2)
        ax.axvline(e, color="red", ls="--", lw=0.8, zorder=2)
        p, f, u = pass_by_event.get(i, (0, 0, 0))
        total = p + f + u
        label = f"event #{i}\n{p}/{total} ✓" + (f" ({f}✗)" if f else "")
        ax.text(
            (s + e) / 2,
            ymin + 0.04 * yspan,
            label,
            ha="center", va="bottom",
            fontsize=8, color="red",
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.7, pad=1.5),
        )

    handles = [
        mpatches.Patch(facecolor=palette[i], alpha=0.35,
                       label=f"pos {i} ({TYPE_INITIAL[t]})")
        for i, t in enumerate(template_types)
    ]
    handles += [
        mpatches.Patch(facecolor="lightgray", alpha=0.5, label="inter-event"),
        Line2D([0], [0], color="red", ls="--", lw=0.8, label="event boundary"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=7,
        ncol=min(len(handles), 8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("feature_schema.yaml"))
    parser.add_argument("--prior-k", type=int, default=200)
    parser.add_argument("--jitter-max-attempts", type=int, default=200)
    args = parser.parse_args()

    out_json = args.out_json or Path(f"tests/out/grounding_seed_{args.seed}.json")
    out_png = args.out_png or Path(f"tests/out/grounding_seed_{args.seed}.png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cfg = Config.load(args.config)
    schema = Schema.load(args.schema)
    result = run_steps_1_to_19(
        cfg, schema,
        rng=rng,
        prior_K=args.prior_k,
        jitter_max_attempts=args.jitter_max_attempts,
    )

    dump = _build_dump(args.seed, cfg, result)
    warnings = _validate_basic_output(dump, cfg)
    dump["validation_warnings"] = warnings

    with open(out_json, "w") as f:
        json.dump(dump, f, indent=2, allow_nan=False)
    _plot(result, cfg, args.seed, out_png)

    summary = dump["diagnostics"]["verification_summary"]
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")
    print(
        f"  events={result.extended.base.layout.nb_events}, "
        f"template_size={len(result.extended.base.event_template.primitives)}, "
        f"verification={summary['passed']}/{summary['total']} passed "
        f"({summary['failed']} failed, {summary['unevaluable']} unevaluable), "
        f"jitter_fallbacks={len(dump['diagnostics']['jitter_fallbacks'])}"
    )
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")


if __name__ == "__main__":
    main()
