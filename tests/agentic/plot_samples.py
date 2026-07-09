from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

BATCH_DIR = Path("tests/out/batch")
DESCRIBED_DIR = Path("tests/agentic/out")
OUT_DIR = Path("tests/agentic/plots")

N_SAMPLES = 100
SEED = 0


def get_positive_base_text(described: dict) -> str:
    for desc in described.get("descriptions", []):
        if desc.get("kind") == "positive_base":
            return desc.get("text", "")
    return ""


def plot_sample(batch_path: Path, described_path: Path, out_path: Path) -> None:
    data = json.loads(batch_path.read_text())
    described = json.loads(described_path.read_text())

    signal_noisy = data["signal_noisy"]
    event_spans = data.get("event_spans", [])
    title_text = get_positive_base_text(described)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(signal_noisy, color="black", lw=0.9)

    for s, e in event_spans:
        ax.axvspan(s, e, alpha=0.2, facecolor="tab:orange", edgecolor="tab:orange", linewidth=0.0)

    ax.set_xlabel("sample index")
    ax.set_ylabel("signal_noisy")
    ax.grid(True, alpha=0.3)
    ax.set_title(title_text, fontsize=9, wrap=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    described_files = sorted(DESCRIBED_DIR.glob("sample_*_described.json"))
    ids = []
    for f in described_files:
        sample_id = f.name.replace("_described.json", "")
        batch_path = BATCH_DIR / f"{sample_id}.json"
        if batch_path.exists():
            ids.append(sample_id)

    rng = random.Random(SEED)
    chosen = rng.sample(ids, min(N_SAMPLES, len(ids)))

    for sample_id in chosen:
        batch_path = BATCH_DIR / f"{sample_id}.json"
        described_path = DESCRIBED_DIR / f"{sample_id}_described.json"
        out_path = OUT_DIR / f"{sample_id}.png"
        plot_sample(batch_path, described_path, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
