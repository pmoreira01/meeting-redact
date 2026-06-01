from __future__ import annotations

from pydantic import BaseModel


class WordOut(BaseModel):
    text: str
    start: float
    end: float
    score: float


class EntityOut(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    score: float
    start_time: float
    end_time: float


class TranscribeResponse(BaseModel):
    text: str
    language: str
    words: list[WordOut]
    entities: list[EntityOut]
