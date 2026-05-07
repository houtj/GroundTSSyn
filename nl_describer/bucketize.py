"""Signal-relative bucketing of primitive features.

Resolves the open item in `pipline.md` step 20: per-signal anchors derived from
`globals` and per-primitive features make every quantitative claim adapt
across signals (and survive z-norm).

Produces both raw values and natural-language buckets for every feature an LLM
might mention (peak2peak, duration, slope, frequency, sharpness, ...).
"""

from __future__ import annotations

from typing import Any

from .config import BucketizeConfig

# String labels for tertile-style anchors. Ordered low -> high.
_AMP_LABELS = ["small amplitude", "medium amplitude", "high amplitude"]
_DUR_LABELS = ["short", "medium", "long"]
_SLOPE_LABELS = ["gentle slope", "moderate slope", "steep slope"]
_FREQ_LABELS = ["slow oscillation", "moderate oscillation", "fast oscillation"]
_SHARP_LABELS = ["dull", "moderate", "sharp"]


def _classify(value: float, anchors_normed: list[float], labels: list[str]) -> str:
    """Map a normalized value (0..1+) into one of `labels` using the anchors.

    `anchors_normed` are sorted thresholds in the SAME normalized space.
    """
    for thr, label in zip(anchors_normed, labels):
        if value <= thr:
            return label
    return labels[-1]


def feature_summary(
    features: dict[str, Any],
    globals_: dict[str, Any],
    signal_length: int,
    cfg: BucketizeConfig,
) -> dict[str, Any]:
    """Return a dict of {raw values + bucket labels} ready for LLM consumption.

    Always-present keys (when the feature exists on the primitive):
      * peak2peak, peak2peak_bucket
      * duration, duration_bucket  (relative to signal_length)
      * slope, slope_bucket        (when applicable)
      * frequency, frequency_bucket (when applicable)
      * sharpness, sharpness_bucket (when applicable)
      * categorical features pass through unchanged (shape, recovery_type, ...).
    """
    out: dict[str, Any] = {}

    # peak2peak — bucketed against globals.peak2peak (the full-signal range).
    p2p = features.get("peak2peak")
    if isinstance(p2p, (int, float)):
        out["peak2peak"] = float(p2p)
        gp = globals_.get("peak2peak", 0.0)
        ratio = float(p2p) / float(gp) if gp else 0.0
        out["peak2peak_bucket"] = _classify(ratio, cfg.amplitude_anchors, _AMP_LABELS)

    # duration — bucketed against signal_length.
    dur = features.get("duration")
    if isinstance(dur, (int, float)):
        out["duration"] = int(dur)
        ratio = float(dur) / float(signal_length) if signal_length else 0.0
        out["duration_bucket"] = _classify(ratio, cfg.duration_anchors, _DUR_LABELS)

    # slope — anchored at "noticeable change of half the signal range over a
    # tenth of the signal length". Buckets are relative.
    slope = features.get("slope")
    if isinstance(slope, (int, float)):
        out["slope"] = float(slope)
        gp = float(globals_.get("peak2peak", 1.0)) or 1.0
        ref = gp / max(signal_length / 10.0, 1.0)  # "moderate" reference rate
        ratio = abs(float(slope)) / ref if ref else 0.0
        out["slope_bucket"] = _classify(ratio, cfg.amplitude_anchors, _SLOPE_LABELS)

    # frequency (oscillation only)
    freq = features.get("frequency")
    if isinstance(freq, (int, float)):
        out["frequency"] = float(freq)
        out["frequency_bucket"] = _classify(
            float(freq), cfg.frequency_anchors, _FREQ_LABELS
        )

    # sharpness (spike/dip)
    sharp = features.get("sharpness")
    if isinstance(sharp, (int, float)):
        out["sharpness"] = float(sharp)
        gp = float(globals_.get("peak2peak", 1.0)) or 1.0
        ref = gp / 5.0  # "moderate" reference sharpness (~5 samples FWHM at full range)
        ratio = float(sharp) / ref if ref else 0.0
        out["sharpness_bucket"] = _classify(ratio, cfg.amplitude_anchors, _SHARP_LABELS)

    # Pass through useful categoricals & numerics that don't need bucketing.
    for k in (
        "shape",
        "recovery_type",
        "asymmetry",
        "envelope_behavior",
        "envelope_asymmetry",
        "regularity",
        "waveform_shape",
        "center_trend",
        "rise_duration",
        "fall_duration",
        "n_cycles",
        "period",
        "rise_slope",
        "fall_slope",
        "start_value",
        "end_value",
        "trough_value",
        "peak_value",
        "mean_value",
    ):
        if k in features:
            out[k] = features[k]

    return out


def constraint_summary(
    constraint: dict[str, Any],
    template_types: list[str],
    globals_: dict[str, Any],
    signal_length: int,
    cfg: BucketizeConfig,
) -> dict[str, Any]:
    """Render a single constraint dict into an LLM-friendly form.

    Adds bucket annotations on the realized values when the underlying feature
    is bucketable. The LLM still sees the operator + threshold so it can
    preserve semantics; the buckets are an extra hint for natural phrasing.
    """
    kind = constraint.get("kind")
    summary: dict[str, Any] = {"kind": kind}
    if kind == "numerical":
        target = constraint.get("target", {})
        feat = target.get("feature")
        summary["target_primitive_index"] = target.get("primitive")
        summary["feature"] = feat
        summary["op"] = constraint.get("op")
        summary["threshold"] = constraint.get("threshold")
        summary["realized_value"] = constraint.get("realized_value")
        bucket = _bucket_for_value(
            feat, constraint.get("realized_value"), globals_, signal_length, cfg
        )
        if bucket is not None:
            summary["realized_bucket"] = bucket
    elif kind == "comparable":
        lhs = constraint.get("lhs", {})
        rhs = constraint.get("rhs", {})
        summary["lhs"] = {
            "primitive_index": lhs.get("primitive"),
            "feature": lhs.get("feature"),
            "realized_value": constraint.get("realized_lhs"),
        }
        summary["rhs"] = {
            "primitive_index": rhs.get("primitive"),
            "feature": rhs.get("feature"),
            "realized_value": constraint.get("realized_rhs"),
        }
        summary["op"] = constraint.get("op")
        # Provide friendly type names for both sides so the LLM can phrase
        # comparisons like "longer than the second decrease".
        if isinstance(lhs.get("primitive"), int):
            idx = lhs["primitive"]
            if 0 <= idx < len(template_types):
                summary["lhs"]["primitive_type"] = template_types[idx]
        if isinstance(rhs.get("primitive"), int):
            idx = rhs["primitive"]
            if 0 <= idx < len(template_types):
                summary["rhs"]["primitive_type"] = template_types[idx]
        elif rhs.get("primitive") == "global":
            summary["rhs"]["primitive_type"] = "global"
    elif kind == "self":
        target = constraint.get("target", {})
        summary["target_primitive_index"] = target.get("primitive")
        summary["feature"] = target.get("feature")
        summary["value"] = constraint.get("value")
    return summary


def _bucket_for_value(
    feature: str | None,
    value: Any,
    globals_: dict[str, Any],
    signal_length: int,
    cfg: BucketizeConfig,
) -> str | None:
    if not isinstance(value, (int, float)) or feature is None:
        return None
    fake_features = {feature: value}
    summary = feature_summary(fake_features, globals_, signal_length, cfg)
    bucket_key = f"{feature}_bucket"
    return summary.get(bucket_key)
