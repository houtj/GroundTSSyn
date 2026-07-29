"""BERT WordPiece token counting + budget enforcement.

The fine-tuning target is a BERT model with a 512-position embedding window,
so any persisted `Description.text` must encode to at most 512 WordPiece IDs
(the count below already includes the `[CLS]` and `[SEP]` specials added by
BERT's post-processor). Word/character counts are unreliable proxies because
WordPiece happily splits a single English word into several sub-tokens
(`oscillation` -> `osci ##llation`, `peak2peak` -> 4-5 pieces), so we use the
real tokenizer.

Loading is lazy + cached so the cold start cost (HF Hub fetch on first use)
is paid once per process, regardless of how many descriptions we score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from tokenizers import Tokenizer

logger = logging.getLogger("nl_describer.token_limit")

# Local, offline copy of bert-base-uncased's tokenizer.json (this server has
# no network access to the HF Hub). Checked before falling back to
# `Tokenizer.from_pretrained`, which requires internet access.
_LOCAL_TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"

_cache: dict[str, Tokenizer] = {}
_lock = Lock()


def get_tokenizer(model_name: str = "bert-base-uncased") -> Tokenizer:
    """Return a cached `tokenizers.Tokenizer` for the given HF model name.

    Prefers the local `nl_describer/tokenizer.json` file when present (no
    network access required). Otherwise falls back to
    `Tokenizer.from_pretrained`, which downloads `tokenizer.json` from the HF
    Hub on first use. Subsequent calls (in the same process) reuse the
    in-memory instance. Thread-safe.
    """
    with _lock:
        tok = _cache.get(model_name)
        if tok is None:
            if model_name == "bert-base-uncased" and _LOCAL_TOKENIZER_PATH.is_file():
                logger.info(
                    "loading tokenizer %s from local file %s (first use)",
                    model_name,
                    _LOCAL_TOKENIZER_PATH,
                )
                tok = Tokenizer.from_file(str(_LOCAL_TOKENIZER_PATH))
            else:
                logger.info(
                    "loading tokenizer %s from HF Hub (first use)", model_name
                )
                tok = Tokenizer.from_pretrained(model_name)
            _cache[model_name] = tok
        return tok


def count_bert_tokens(text: str, model_name: str = "bert-base-uncased") -> int:
    """Count WordPiece tokens for `text`, including BERT's `[CLS]` / `[SEP]`.

    Uses `Tokenizer.encode` which runs the full pipeline (normalizer ->
    pre-tokenizer -> WordPiece -> post-processor), so the returned count
    matches what BERT will see during training.
    """
    tok = get_tokenizer(model_name)
    enc = tok.encode(text)
    return len(enc.ids)


def within_limit(
    text: str,
    max_tokens: int,
    model_name: str = "bert-base-uncased",
) -> bool:
    """True iff `text` encodes to at most `max_tokens` BERT WordPiece tokens."""
    return count_bert_tokens(text, model_name) <= max_tokens
