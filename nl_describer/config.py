from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AzureConfig(BaseModel):
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    deployment_env: str = "AZURE_OPENAI_DEPLOYMENT"
    api_version_env: str = "AZURE_OPENAI_API_VERSION"


class ObservationSubset(BaseModel):
    min: int = 1
    max: int = 3


class GenerationConfig(BaseModel):
    n_positive_variants: int = 3
    max_negatives: int = 5
    n_observations_max: int = 8
    variant_observation_subset: ObservationSubset = Field(default_factory=ObservationSubset)


class PlotConfig(BaseModel):
    width: float = 12.0
    height: float = 4.0
    dpi: int = 100


class VisionConfig(BaseModel):
    global_plot: PlotConfig = Field(default_factory=PlotConfig)
    event_plot: PlotConfig = Field(default_factory=PlotConfig)
    slice_plot: PlotConfig = Field(default_factory=PlotConfig)


class CandidatesConfig(BaseModel):
    merge_cp_max: float = 0.4
    reinterpret_vis_max: float = 0.4
    split_cp_max: float = 0.4


class AgentLLMConfig(BaseModel):
    temperature: float = 0.5


class LLMConfig(BaseModel):
    base_describer: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(temperature=0.7))
    variant_observe: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(temperature=0.4))
    variant_generate: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(temperature=0.6))
    critic: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(temperature=0.0))
    retries: int = 2
    request_timeout_s: int = 120


class BucketizeConfig(BaseModel):
    amplitude_anchors: list[float] = Field(default_factory=lambda: [0.33, 0.66])
    duration_anchors: list[float] = Field(default_factory=lambda: [0.05, 0.20])
    frequency_anchors: list[float] = Field(default_factory=lambda: [0.10, 0.25])


class LengthLimitConfig(BaseModel):
    """Hard cap on the BERT-tokenized length of every persisted description.

    `max_bert_tokens` is the absolute limit enforced in code (counted with
    `tokenizer_model`'s WordPiece tokenizer, including [CLS]/[SEP]). The
    soft_* fields are advertised to the LLM in system prompts; they are
    intentionally lower than the hard cap to leave headroom for sub-token
    splits. `retry_on_overflow` controls whether to re-call the agent once
    with a "be more concise" instruction before dropping.
    """

    max_bert_tokens: int = 512
    tokenizer_model: str = "bert-base-uncased"
    soft_word_budget: int = 300
    soft_char_budget: int = 1800
    retry_on_overflow: bool = True


class Config(BaseModel):
    azure: AzureConfig = Field(default_factory=AzureConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    bucketize: BucketizeConfig = Field(default_factory=BucketizeConfig)
    length_limit: LengthLimitConfig = Field(default_factory=LengthLimitConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def default_path(cls) -> Path:
        return Path(__file__).parent / "config.yaml"
