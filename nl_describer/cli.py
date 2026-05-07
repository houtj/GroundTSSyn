"""CLI entry point.

Usage:
    uv run python -m nl_describer.cli \
        --in tests/out/batch/sample_000000.json \
        --out tests/agentic/out/sample_000000_described.json \
        --plots-dir tests/agentic/out/plots/sample_000000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import Config
from .pipeline import describe_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in", dest="in_path", required=True, help="Path to BasicOutput JSON"
    )
    parser.add_argument(
        "--out", dest="out_path", required=True, help="Path to write DescribedSample JSON"
    )
    parser.add_argument(
        "--plots-dir",
        dest="plots_dir",
        default=None,
        help="Optional directory to persist global.png and event.png",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to config.yaml (defaults to nl_describer/config.yaml)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    cfg_path = Path(args.config_path) if args.config_path else Config.default_path()
    cfg = Config.load(cfg_path)

    result = describe_sample(
        args.in_path, cfg, plots_dir=args.plots_dir, seed=args.seed
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
