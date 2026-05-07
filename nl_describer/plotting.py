"""Matplotlib renderers used by the vision-capable agents.

All renderers produce two outputs:
  * a base64-encoded PNG string (for inlining as a vision message)
  * the underlying PNG bytes (so the orchestrator can also persist them)

A consistent color map is used across plots so the LLM can refer to primitive
types by both name and color.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .config import PlotConfig

_TYPE_COLORS = {
    "spike": "#d62728",
    "dip": "#9467bd",
    "increase": "#2ca02c",
    "decrease": "#ff7f0e",
    "plateau": "#7f7f7f",
    "oscillation": "#1f77b4",
}


def _png_to_b64(buf: io.BytesIO) -> tuple[str, bytes]:
    data = buf.getvalue()
    return base64.b64encode(data).decode("ascii"), data


def _annotate_layout(
    ax,
    layout: list[dict[str, Any]],
    event_spans: list[list[int]],
    *,
    show_visibility: bool = True,
) -> None:
    """Shade each primitive span with its type color and shade event spans."""
    y0, y1 = ax.get_ylim()
    height = y1 - y0
    bar_h = 0.04 * height
    for entry in layout:
        s, e = entry["span"]
        ptype = entry["type"]
        color = _TYPE_COLORS.get(ptype, "#000000")
        if entry.get("kind") == "in_event":
            ax.add_patch(
                Rectangle(
                    (s, y0),
                    e - s,
                    bar_h,
                    color=color,
                    alpha=0.85,
                    linewidth=0,
                )
            )
            mid = (s + e) / 2
            label = ptype[:3]
            if show_visibility and entry.get("visibility") is not None:
                label += f"\nv={entry['visibility']:.2f}"
            ax.text(
                mid,
                y0 + bar_h * 1.1,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                color=color,
            )
        else:
            ax.add_patch(
                Rectangle(
                    (s, y0),
                    e - s,
                    bar_h * 0.5,
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                )
            )
    for i, (es, ee) in enumerate(event_spans):
        ax.axvspan(es, ee, color="yellow", alpha=0.08, zorder=0)
        ax.text(
            (es + ee) / 2,
            y1 - 0.02 * height,
            f"event {i}",
            ha="center",
            va="top",
            fontsize=7,
            color="black",
            alpha=0.6,
        )


def render_global_plot(
    signal_noisy: list[float],
    layout: list[dict[str, Any]],
    event_spans: list[list[int]],
    cfg: PlotConfig,
) -> tuple[str, bytes]:
    fig, ax = plt.subplots(figsize=(cfg.width, cfg.height), dpi=cfg.dpi)
    ax.plot(signal_noisy, linewidth=0.8, color="black", label="signal_noisy")
    ax.set_xlim(0, len(signal_noisy))
    ax.set_xlabel("sample index")
    ax.set_ylabel("z-normed value")
    ax.set_title("Full signal (noisy) — primitive boundaries colored by type")
    _annotate_layout(ax, layout, event_spans, show_visibility=True)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return _png_to_b64(buf)


def render_event_plot(
    signal_noisy: list[float],
    occurrence: dict[str, Any],
    cfg: PlotConfig,
) -> tuple[str, bytes]:
    """Zoomed plot of one event occurrence with per-primitive annotations."""
    span = occurrence["span"]
    s, e = int(span[0]), int(span[1])
    pad = max(2, int(0.05 * (e - s)))
    s_p = max(0, s - pad)
    e_p = min(len(signal_noisy) - 1, e + pad)

    fig, ax = plt.subplots(figsize=(cfg.width, cfg.height), dpi=cfg.dpi)
    xs = list(range(s_p, e_p + 1))
    ax.plot(xs, signal_noisy[s_p : e_p + 1], linewidth=1.2, color="black")
    ax.set_xlim(s_p, e_p)
    ax.set_xlabel("sample index")
    ax.set_ylabel("z-normed value")
    ax.set_title(f"Zoom: event {occurrence.get('event_index', '?')} span [{s}, {e}]")

    for p in occurrence.get("primitives", []):
        ps, pe = p["span"]
        color = _TYPE_COLORS.get(p["type"], "#000000")
        ax.axvspan(ps, pe, color=color, alpha=0.15)
        vis = p.get("visibility")
        cp = p.get("cp_visibility")
        label = f"{p['type'][:4]}#{p['template_index']}"
        if isinstance(vis, (int, float)):
            label += f"\nv={vis:.2f}"
        if isinstance(cp, (int, float)):
            label += f", cp={cp:.2f}"
        ax.text(
            (ps + pe) / 2,
            ax.get_ylim()[1],
            label,
            ha="center",
            va="top",
            fontsize=7,
            color=color,
        )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return _png_to_b64(buf)


def render_slice_plot(
    signal_noisy: list[float],
    s: int,
    e: int,
    cfg: PlotConfig,
) -> tuple[str, bytes]:
    """Ad-hoc zoom for the critic agent (`free_zoom_request`)."""
    s = max(0, int(s))
    e = min(len(signal_noisy) - 1, int(e))
    if e <= s:
        e = min(len(signal_noisy) - 1, s + 1)
    fig, ax = plt.subplots(figsize=(cfg.width, cfg.height), dpi=cfg.dpi)
    ax.plot(range(s, e + 1), signal_noisy[s : e + 1], linewidth=1.2, color="black")
    ax.set_xlim(s, e)
    ax.set_xlabel("sample index")
    ax.set_ylabel("z-normed value")
    ax.set_title(f"Zoom slice [{s}, {e}]")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return _png_to_b64(buf)
