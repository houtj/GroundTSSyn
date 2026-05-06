from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PrimitiveType(str, Enum):
    SPIKE = "spike"
    DIP = "dip"
    INCREASE = "increase"
    DECREASE = "decrease"
    OSCILLATION = "oscillation"
    PLATEAU = "plateau"


class Primitive(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    type: PrimitiveType
    values: list[float]
    definitive: dict[str, float | int]
    random: dict[str, float | int | str]
    derived: dict[str, Any] | None = None


class Event(BaseModel):
    primitives: list[Primitive]


class PlacedPrimitive(BaseModel):
    primitive: Primitive
    span: tuple[int, int]
    event_index: int | None = None
    template_position: int | None = None


class ConcatResult(BaseModel):
    signal: list[float]
    placed: list[PlacedPrimitive]
    layout: list[Primitive | Event] = Field(default_factory=list)
    event_spans: list[tuple[int, int]] = Field(default_factory=list)
