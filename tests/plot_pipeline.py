from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from data_gen.config import Config
from data_gen.pipeline import PipelineResult, run_steps_1_to_11
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


def _annotate(ax: Axes, result: PipelineResult) -> None:
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    label_y_high = ymax - 0.05 * yspan
    label_y_low = ymax - 0.12 * yspan

    for i, pp in enumerate(result.raw.placed):
        s, e = pp.span
        is_event = pp.event_index is not None
        ax.axvspan(
            s,
            e,
            alpha=0.18,
            facecolor="tab:orange" if is_event else "lightgray",
            hatch="///" if is_event else None,
            edgecolor="tab:orange" if is_event else "gray",
            linewidth=0.0,
        )
        mid = (s + e) / 2
        y = label_y_high if i % 2 == 0 else label_y_low
        ax.text(
            mid,
            y,
            TYPE_INITIAL[pp.primitive.type],
            ha="center",
            va="top",
            fontsize=7,
            color="black",
        )

    for i, (s, e) in enumerate(result.raw.event_spans):
        ax.axvline(s, color="red", ls="--", lw=0.8)
        ax.axvline(e, color="red", ls="--", lw=0.8)
        ax.text(
            (s + e) / 2,
            ymin + 0.04 * yspan,
            f"event #{i}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.7, pad=1.5),
        )


def _legend_handles() -> list:
    return [
        mpatches.Patch(
            facecolor="tab:orange", alpha=0.35, hatch="///", label="in-event primitive"
        ),
        mpatches.Patch(facecolor="lightgray", alpha=0.5, label="inter-event primitive"),
        Line2D([0], [0], color="red", ls="--", lw=0.8, label="event boundary"),
    ]


def _sanity_checks(result: PipelineResult, cfg: Config) -> None:
    assert len(result.raw.signal) == cfg.signal_length, (
        f"signal length {len(result.raw.signal)} != cfg.signal_length {cfg.signal_length}"
    )
    placed = result.raw.placed
    for prev, nxt in zip(placed[:-1], placed[1:]):
        assert prev.span[1] == nxt.span[0], (
            f"boundary mismatch: prev.span={prev.span}, next.span={nxt.span}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("tests/out/pipeline.png"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("feature_schema.yaml"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = Config.load(args.config)
    schema = Schema.load(args.schema)
    result = run_steps_1_to_11(cfg, schema, rng=rng)
    _sanity_checks(result, cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax_top.plot(result.raw.signal, color="black", lw=0.9)
    ax_top.set_title(
        f"Raw concatenated signal (step 10) · "
        f"signal_length={cfg.signal_length} · "
        f"N_events={result.layout.nb_events} · "
        f"event_length={result.layout.event_length} · "
        f"seed={args.seed}"
    )
    ax_top.set_ylabel("raw value")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(result.signal, color="black", lw=0.9)
    ax_bot.set_title("Z-normalized signal (step 11)")
    ax_bot.set_ylabel("z-score")
    ax_bot.set_xlabel("sample index")
    ax_bot.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    ax_bot.grid(True, alpha=0.3)

    # Annotate AFTER plotting so we can use the y-limits.
    for ax in (ax_top, ax_bot):
        _annotate(ax, result)

    ax_top.legend(handles=_legend_handles(), loc="lower right", fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
