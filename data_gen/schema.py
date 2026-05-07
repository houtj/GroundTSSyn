from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, RootModel

from data_gen.types import PrimitiveType

Generative = Literal["definitive", "random", "derived"]
ScopeSource = Literal["fixed", "config", "layout"]
PdfName = Literal["uniform", "linear", "skewed_normal", "fixed"]
ComparableDomain = Literal[
    "absolute_value", "magnitude", "duration", "slope", "sharpness", "frequency"
]


class FeatureSpec(BaseModel):
    type: Literal["continuous", "integer", "categorical"]
    generative: Generative
    level_dependent: bool
    constraint: list[Literal["numerical", "comparable", "self"]] = Field(default_factory=list)
    aliases: list[str] | None = None
    description: str

    # Random features only
    scope: list[float | int | str] | None = None
    pdf: PdfName | None = None

    # Definitive features only
    scope_source: ScopeSource | None = None

    # Constraintable features (excluded for global.* and non-constraintable)
    salience: float | None = None

    # Comparable-constraintable features only
    comparable_domain: ComparableDomain | None = None


class Schema(RootModel[dict[str, dict[str, FeatureSpec]]]):
    """Top-level schema: {primitive_name: {feature_name: FeatureSpec}}."""

    @classmethod
    def load(cls, path: str | Path = "data_gen/feature_schema.yaml") -> "Schema":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def features(self, primitive: PrimitiveType | str) -> dict[str, FeatureSpec]:
        key = primitive.value if isinstance(primitive, PrimitiveType) else primitive
        return self.root[key]

    def by_kind(
        self, primitive: PrimitiveType | str, kind: Generative
    ) -> dict[str, FeatureSpec]:
        return {
            name: spec
            for name, spec in self.features(primitive).items()
            if spec.generative == kind
        }

    def definitive_features(
        self, primitive: PrimitiveType | str
    ) -> dict[str, FeatureSpec]:
        return self.by_kind(primitive, "definitive")

    def random_features(
        self, primitive: PrimitiveType | str
    ) -> dict[str, FeatureSpec]:
        return self.by_kind(primitive, "random")

    def derived_features(
        self, primitive: PrimitiveType | str
    ) -> dict[str, FeatureSpec]:
        return self.by_kind(primitive, "derived")

    def salience(
        self, primitive: PrimitiveType | str, feature: str, default: float = 0.5
    ) -> float:
        spec = self.features(primitive).get(feature)
        if spec is None or spec.salience is None:
            return default
        return spec.salience

    def comparable_peers(
        self, primitive: PrimitiveType | str, feature: str
    ) -> list[tuple[str, str]]:
        """All (primitive_str, feature_name) pairs sharing this feature's
        comparable_domain. The feature itself is included; callers must skip
        same-position-same-feature pairs."""
        spec = self.features(primitive).get(feature)
        if spec is None or spec.comparable_domain is None:
            return []
        domain = spec.comparable_domain
        out: list[tuple[str, str]] = []
        for ptype_str, features in self.root.items():
            for fname, fspec in features.items():
                if fspec.comparable_domain == domain:
                    out.append((ptype_str, fname))
        return out
