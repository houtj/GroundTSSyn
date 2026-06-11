"""Unit tests for the BERT-token length gate.

Pure-python tests — does NOT require Azure OpenAI credentials. Verifies:
  * the tokenizer counts include [CLS] / [SEP],
  * `within_limit` rejects a deliberately oversize string,
  * `within_limit` accepts a typical short description,
  * the pipeline's hard-gate helpers (_within_token_budget / _bert_tokens)
    pick up `cfg.length_limit.tokenizer_model` and `max_bert_tokens`.

Usage:
    uv run python -m tests.agentic.test_token_limit
"""

from __future__ import annotations

import sys

from nl_describer.config import Config, LengthLimitConfig
from nl_describer.pipeline import _bert_tokens, _within_token_budget
from nl_describer.token_limit import count_bert_tokens, within_limit


SHORT_DESCRIPTION = (
    "downward slide; gentle dropoff: with a convex profile; level stretch; "
    "brief downward notch; upward ramp; flat segment: lasting the same duration "
    "as the first decrease, lasting the same duration as the increase, no longer "
    "than the second decrease"
)


def _build_oversize(target_tokens: int = 600) -> str:
    """Build a realistic-looking but deliberately oversize description.

    We repeat a typical chunk many times so the result still parses as the
    `primitive: c, c; ...` format and overshoots the 512-token budget.
    """
    chunk = (
        "a sharp upward jolt: with a high amplitude exceeding seven, with a "
        "convex profile, lasting the same duration as the second decrease"
    )
    parts = [chunk] * 50  # 50 chunks ~= well over 600 tokens
    return "; ".join(parts)


def main() -> int:
    failures: list[str] = []

    # 1. Tokenizer counts include specials.
    n = count_bert_tokens("hello world")
    if n != 4:  # [CLS] hello world [SEP]
        failures.append(f"expected 4 tokens for 'hello world', got {n}")
    else:
        print(f"OK: 'hello world' -> {n} tokens (includes [CLS]/[SEP])")

    # 2. Short description well under budget.
    n_short = count_bert_tokens(SHORT_DESCRIPTION)
    if not within_limit(SHORT_DESCRIPTION, 512):
        failures.append(f"short description unexpectedly over 512 tokens: {n_short}")
    else:
        print(f"OK: short description = {n_short} tokens (<= 512)")

    # 3. Oversize description is rejected.
    oversize = _build_oversize()
    n_over = count_bert_tokens(oversize)
    if within_limit(oversize, 512):
        failures.append(
            f"oversize description ({n_over} tokens) was NOT rejected by within_limit"
        )
    else:
        print(f"OK: oversize description = {n_over} tokens (rejected)")

    # 4. Pipeline helpers honor cfg.length_limit.
    cfg = Config()
    cfg.length_limit = LengthLimitConfig(
        max_bert_tokens=512,
        tokenizer_model="bert-base-uncased",
    )
    if _bert_tokens(SHORT_DESCRIPTION, cfg) != n_short:
        failures.append("pipeline._bert_tokens disagreed with count_bert_tokens")
    else:
        print("OK: pipeline._bert_tokens matches count_bert_tokens")

    if not _within_token_budget(SHORT_DESCRIPTION, cfg):
        failures.append("pipeline._within_token_budget false-rejected short description")
    if _within_token_budget(oversize, cfg):
        failures.append("pipeline._within_token_budget false-accepted oversize")
    if not failures:
        print("OK: pipeline gating helpers match raw tokenizer outputs")

    # 5. Tightening max_bert_tokens flips the verdict.
    cfg.length_limit = LengthLimitConfig(max_bert_tokens=10)
    if _within_token_budget(SHORT_DESCRIPTION, cfg):
        failures.append("max_bert_tokens=10 should reject the typical short description")
    else:
        print("OK: max_bert_tokens=10 correctly rejects short description")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll token-limit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
