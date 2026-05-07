"""Pydantic IO schemas for the agentic system.

Two groups:
  * Final output schemas (Description, GroundingChunk, DescribedSample) that
    persist to disk.
  * Per-agent structured-output schemas consumed by Azure OpenAI's structured
    output mode (`response_format=PydanticModel`).

OpenAI's structured-output mode has a few constraints we honor:
  * All non-default fields must be `required`; pydantic handles this.
  * `tuple[int, int]` is NOT supported; we use `list[int]` (length 2) and
    convert when serializing the final output.
  * `Optional[X]` becomes nullable in JSON schema; we use `X | None` or
    `Optional[X]` consistently with explicit defaults of None.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from data_gen.types import PrimitiveType

# ---------------------------------------------------------------------------
# Final output (persisted)
# ---------------------------------------------------------------------------


class GroundingChunk(BaseModel):
    """One `primitive: c, c` chunk in the assembled description text."""

    text_span: list[int]            # [char_start, char_end] inside Description.text
    ts_spans: list[list[int]]       # one [s, e] per event occurrence


class Description(BaseModel):
    kind: Literal["positive_base", "positive_variant", "negative"]
    text: str
    grounding: list[GroundingChunk] = Field(default_factory=list)
    event_spans: list[list[int]] = Field(default_factory=list)
    applied_observations: list[str] = Field(default_factory=list)
    # Audit / debug fields
    source_neg_index: Optional[int] = None       # for negatives: index into neg_groundings
    mutation: Optional[str] = None               # for negatives: original mutation tag
    critic_verdict: Optional[str] = None         # accept | borderline | reject (post-critic)
    critic_issues: list[str] = Field(default_factory=list)


class DescribedSample(BaseModel):
    source_path: str
    descriptions: list[Description]


# ---------------------------------------------------------------------------
# Agent 1: BaseDescriber IO
# ---------------------------------------------------------------------------


class BaseChunk(BaseModel):
    template_index: int
    primitive_phrase: str
    constraint_phrases: list[str]


class NegativeChunk(BaseModel):
    template_index: int
    primitive_phrase: str
    constraint_phrases: list[str]


class NegativeDescription(BaseModel):
    chunks: list[NegativeChunk]


class BaseAgentOutput(BaseModel):
    positive_chunks: list[BaseChunk]
    negatives: list[NegativeDescription]


# ---------------------------------------------------------------------------
# Agent 2: Variant — observe (Turn 1)
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    op: Literal["merge", "split", "reinterpret"]
    target_template_indices: list[int]
    new_type: PrimitiveType
    split_strategy: Optional[Literal["increase+decrease", "decrease+increase"]] = None
    rationale: str
    confidence: Literal["high", "medium", "low"]


class ObservationsOutput(BaseModel):
    observations: list[Observation]


# ---------------------------------------------------------------------------
# Agent 2: Variant — generate (Turn 2..N)
# ---------------------------------------------------------------------------


class VariantChunk(BaseModel):
    derived_from: list[int]                 # template positions covered by this chunk
    op_applied: Literal["none", "merge", "split", "reinterpret"]
    new_type: PrimitiveType
    primitive_phrase: str
    constraint_phrases: list[str]
    sub_span_fraction: Optional[list[float]] = None  # [start_frac, end_frac] for split


class VariantOutput(BaseModel):
    chunks: list[VariantChunk]


# ---------------------------------------------------------------------------
# Agent 3: Critic
# ---------------------------------------------------------------------------


class CritiqueItem(BaseModel):
    description_index: int
    verdict: Literal["accept", "borderline", "reject"]
    issues: list[str] = Field(default_factory=list)
    free_zoom_request: Optional[list[int]] = None  # [s, e] for ad-hoc zoom


class CritiqueOutput(BaseModel):
    items: list[CritiqueItem]
