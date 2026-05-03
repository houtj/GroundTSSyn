from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from data_gen.config import Config
from data_gen.pipeline import ExtendedPipelineResult, run_steps_1_to_16
from data_gen.schema import Schema


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


def _level_dependent_keys(schema: Schema, ptype) -> list[str]:
    return [
        name
        for name, spec in schema.features(ptype).items()
        if spec.level_dependent
    ]


def _level_independent_features(schema: Schema, ptype, feats: dict[str, Any]) -> dict:
    keys = {
        name
        for name, spec in schema.features(ptype).items()
        if not spec.level_dependent
    }
    return {k: v for k, v in feats.items() if k in keys}


def _level_dependent_features(schema: Schema, ptype, feats: dict[str, Any]) -> dict:
    keys = set(_level_dependent_keys(schema, ptype))
    return {k: v for k, v in feats.items() if k in keys}


def _build_dump(
    seed: int, cfg: Config, schema: Schema, result: ExtendedPipelineResult
) -> dict[str, Any]:
    base = result.base
    layout = base.layout
    template_types = [p.type for p in base.event_template.primitives]
    first_occ = result.normed.in_event[0] if result.normed.in_event else []

    event_template = []
    for pos, ptype in enumerate(template_types):
        feats = first_occ[pos] if pos < len(first_occ) else {}
        event_template.append(
            {
                "position": pos,
                "type": ptype.value,
                "level_independent_features": _level_independent_features(
                    schema, ptype, feats
                ),
            }
        )

    occurrences = []
    for ev_idx, primitives in enumerate(result.normed.in_event):
        span = base.raw.event_spans[ev_idx] if ev_idx < len(base.raw.event_spans) else None
        per_prim = []
        for pos, feats in enumerate(primitives):
            ptype = template_types[pos]
            per_prim.append(
                {
                    "position": pos,
                    "type": ptype.value,
                    "level_dependent_features": _level_dependent_features(
                        schema, ptype, feats
                    ),
                }
            )
        occurrences.append(
            {
                "event_index": ev_idx,
                "span": list(span) if span is not None else None,
                "primitives": per_prim,
            }
        )

    constraints = []
    for grounded in result.constraints.grounded:
        items = []
        for c in grounded.constraints:
            items.append(c.model_dump(mode="json"))
        constraints.append(
            {
                "template_index": grounded.template_index,
                "type": grounded.type.value,
                "requested_count": grounded.requested_count,
                "actual_count": grounded.actual_count,
                "constraints": items,
            }
        )

    summary = _summarize(constraints)

    return _sanitize(
        {
            "seed": seed,
            "config_summary": {
                "signal_length": cfg.signal_length,
                "nb_events": layout.nb_events,
                "event_length": layout.event_length,
                "nb_in_event_primitives": len(template_types),
            },
            "event_template": event_template,
            "globals": result.normed.globals,
            "occurrences": occurrences,
            "constraints": constraints,
            "summary": summary,
        }
    )


def _summarize(constraints: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(len(g["constraints"]) for g in constraints)
    by_kind: dict[str, int] = {"numerical": 0, "comparable": 0}
    by_count: dict[int, int] = {}
    actual_vs_requested = []
    for g in constraints:
        for c in g["constraints"]:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
        by_count[g["actual_count"]] = by_count.get(g["actual_count"], 0) + 1
        actual_vs_requested.append(
            {
                "template_index": g["template_index"],
                "requested": g["requested_count"],
                "actual": g["actual_count"],
            }
        )
    return {
        "total_constraints": total,
        "by_kind": by_kind,
        "primitives_by_constraint_count": {str(k): v for k, v in sorted(by_count.items())},
        "actual_vs_requested_per_primitive": actual_vs_requested,
    }


def _validate(dump: dict[str, Any]) -> list[str]:
    """Run sanity checks; return list of warnings."""
    warnings: list[str] = []
    for grounded in dump["constraints"]:
        for c in grounded["constraints"]:
            op = c["op"]
            if c["kind"] == "numerical":
                v, t = c["realized_value"], c["threshold"]
                if not _check_op(v, op, t):
                    warnings.append(
                        f"numerical constraint not satisfied by realized value: "
                        f"prim={grounded['template_index']} feat={c['target']['feature']} "
                        f"{v} {op} {t}"
                    )
            else:
                x, y = c["realized_lhs"], c["realized_rhs"]
                if not _check_op(x, op, y):
                    warnings.append(
                        f"comparable constraint not satisfied by realized values: "
                        f"prim={grounded['template_index']} {c['lhs']['feature']} "
                        f"{x} {op} {y} (rhs={c['rhs']['feature']})"
                    )
                if (
                    c["lhs"]["primitive"] == c["rhs"]["primitive"]
                    and c["lhs"]["feature"] == c["rhs"]["feature"]
                ):
                    warnings.append(
                        f"self-loop comparable constraint at prim={grounded['template_index']}"
                    )
            p = c["p_satisfied"]
            if p < 0.01 or p > 0.99:
                warnings.append(
                    f"p_satisfied {p} outside [0.01, 0.99] band at "
                    f"prim={grounded['template_index']}"
                )
    return warnings


def _check_op(a: float, op: str, b: float) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("feature_schema.yaml"))
    parser.add_argument("--prior-k", type=int, default=200)
    args = parser.parse_args()

    out = args.out or Path(f"tests/out/constraints_seed_{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cfg = Config.load(args.config)
    schema = Schema.load(args.schema)
    result = run_steps_1_to_16(cfg, schema, rng=rng, prior_K=args.prior_k)

    dump = _build_dump(args.seed, cfg, schema, result)
    warnings = _validate(dump)
    dump["validation_warnings"] = warnings

    with open(out, "w") as f:
        json.dump(dump, f, indent=2, allow_nan=False)
    print(f"wrote {out}")
    print(
        f"  events={result.base.layout.nb_events}, "
        f"template_size={len(result.base.event_template.primitives)}, "
        f"total_constraints={dump['summary']['total_constraints']}, "
        f"by_kind={dump['summary']['by_kind']}"
    )
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")


if __name__ == "__main__":
    main()
