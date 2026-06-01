"""Ensemble detector — routes entity detection by type across multiple models.

Current state: stub that delegates all types to a single default detector.
Architecture is ready for per-type routing per CLAUDE.md:
  PER → DeBERTa, LOC → Flair OntoNotes, ORG → Flair OntoNotes Large
"""
from __future__ import annotations

from meeting_redact.ner.detector import BaseDetector
from meeting_redact.ner.entity import Entity


class EnsembleDetector(BaseDetector):
    def __init__(self, default_detector: BaseDetector) -> None:
        self._default = default_detector

    def detect(self, text: str) -> list[Entity]:
        return self._default.detect(text)
