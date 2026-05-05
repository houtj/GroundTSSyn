from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from data_gen.types import PrimitiveType

PdfName = Literal["uniform", "linear", "skewed_normal"]


class PostMatchConfig(BaseModel):
    enabled: bool = True


class Config(BaseModel):
    signal_length: int

    nb_event_occurrence: list[int]
    pdf_nb_event_occurrence: PdfName

    nb_primitives_in_event: list[int]
    pdf_nb_primitives_in_event: PdfName

    nb_inter_event_primitives: list[int]
    pdf_nb_inter_event_primitives: PdfName
    min_nb_weight: float
    max_nb_weight: float

    primitive_type_samping_weight: dict[str, float]

    min_primitive_length: int
    min_spike_length: int
    max_spike_length: int

    pdf_event_length: PdfName
    event_average_length_weight: float
    event_extreme_length_weight: float

    pdf_primitives_length: PdfName
    primitive_average_length_weight: float
    primitive_extreme_length_weight: float

    event_primitive_peak_2_peak: list[float]
    pdf_event_primitive_peak_2_peak: PdfName
    inter_event_primitive_peak_2_peak: list[float]
    pdf_inter_event_primitive_peak_2_peak: PdfName

    nb_constraints: list[int]
    pdf_nb_constraints: list[float]

    jitter_randomness: float

    numerical_constraint_weight: float
    comparable_constraint_weight: float

    drift_limit: bool
    drift_threshold: float

    max_junction_angle_deg: float = 175.0

    post_match: PostMatchConfig = PostMatchConfig()

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def peak2peak_range(self, in_event: bool) -> tuple[float, float]:
        rng = (
            self.event_primitive_peak_2_peak
            if in_event
            else self.inter_event_primitive_peak_2_peak
        )
        return float(rng[0]), float(rng[1])

    def primitive_weights(self) -> tuple[list[PrimitiveType], list[float]]:
        types = [PrimitiveType(name) for name in self.primitive_type_samping_weight]
        weights = list(self.primitive_type_samping_weight.values())
        return types, weights
