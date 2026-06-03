from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    text: str
    label: str          # PER, LOC, ORG, PHONE, EMAIL, SSN
    start_char: int
    end_char: int
    score: float
    start_time: float = field(default=0.0)   # populated by mapper
    end_time: float = field(default=0.0)     # populated by mapper
    anonymized_as: str = field(default="")   # populated by EntityRegistry
