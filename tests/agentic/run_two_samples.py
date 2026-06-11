"""Test harness — run the agentic describer on 2 samples.

Usage:
    uv run python -m tests.agentic.run_two_samples
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from nl_describer.config import Config
from nl_describer.pipeline import describe_sample
from nl_describer.verifier import matches_format

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "tests" / "out" / "batch"
OUT_DIR = REPO_ROOT / "tests" / "agentic" / "out"
PLOTS_DIR = OUT_DIR / "plots"

SAMPLES = [
    "sample_000000.json",
    "sample_000001.json",
]


def _validate(out_path: Path) -> tuple[bool, list[str]]:
    """Validation hooks listed in the plan."""
    issues: list[str] = []
    try:
        with open(out_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"json load failed: {e}"]
    if "descriptions" not in data:
        return False, ["missing 'descriptions' key"]
    descs = data["descriptions"]
    if not descs:
        return False, ["zero descriptions produced"]
    has_variant = False
    for i, d in enumerate(descs):
        kind = d.get("kind")
        text = d.get("text", "")
        if not text:
            issues.append(f"description[{i}] has empty text")
            continue
        if not matches_format(text):
            issues.append(f"description[{i}] text does not match format regex: {text!r}")
        # Round-trip text spans
        for j, gc in enumerate(d.get("grounding", [])):
            ts = gc.get("text_span", [])
            if len(ts) != 2:
                issues.append(f"description[{i}].grounding[{j}].text_span malformed")
                continue
            s, e = ts
            if not (0 <= s <= e <= len(text)):
                issues.append(
                    f"description[{i}].grounding[{j}].text_span [{s},{e}] out of range"
                )
        if kind == "negative":
            if d.get("grounding"):
                issues.append(f"negative description[{i}] has non-empty grounding")
            if d.get("event_spans"):
                issues.append(f"negative description[{i}] has non-empty event_spans")
        if kind == "positive_variant":
            has_variant = True
            if not d.get("applied_observations"):
                issues.append(
                    f"positive_variant description[{i}] has no applied_observations"
                )
    if not has_variant:
        issues.append("no positive_variant descriptions produced (acceptable but worth noting)")
    return len([i for i in issues if "no positive_variant" not in i]) == 0, issues


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    cfg = Config.load(Config.default_path())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_ok = True
    for sample_name in SAMPLES:
        in_path = INPUT_DIR / sample_name
        if not in_path.exists():
            print(f"SKIP {sample_name}: missing input {in_path}")
            continue
        out_path = OUT_DIR / f"{in_path.stem}_described.json"
        sample_plots = PLOTS_DIR / in_path.stem
        print(f"\n=== {sample_name} ===")
        try:
            result = describe_sample(
                in_path,
                cfg,
                plots_dir=sample_plots,
                seed=int(in_path.stem.split("_")[-1]) if in_path.stem.split("_")[-1].isdigit() else 0,
            )
            with open(out_path, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2)
            print(f"  wrote {out_path}")
            ok, issues = _validate(out_path)
            for issue in issues:
                print(f"  WARN {issue}")
            print(f"  validation: {'PASS' if ok else 'FAIL'}")
            print(f"  descriptions: {len(result.descriptions)}")
            for d in result.descriptions:
                print(
                    f"    - {d.kind:>16}  verdict={d.critic_verdict}  text={d.text[:80]!r}"
                )
            overall_ok = overall_ok and ok
        except Exception as e:
            traceback.print_exc()
            print(f"  ERROR running describe_sample: {e}")
            overall_ok = False
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
