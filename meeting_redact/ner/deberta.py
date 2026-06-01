"""DeBERTa-large NER detector (CoNLL03 fine-tune).

Model: Gladiator/microsoft-deberta-v3-large_ner_conll2003
Labels: PER, LOC, ORG, MISC  — MISC is dropped by default (not in NER_ENTITY_TYPES).
"""
from __future__ import annotations

from transformers import pipeline

from meeting_redact.config import settings
from meeting_redact.ner.detector import BaseDetector
from meeting_redact.ner.entity import Entity


class DeBERTaDetector(BaseDetector):
    """Token-classification pipeline backed by DeBERTa-v3-large.

    The transformers pipeline is loaded once at construction and reused.
    Results are filtered to the configured entity types and score threshold
    before being returned as Entity objects.
    """

    def __init__(
        self,
        model_name: str = settings.NER_MODEL,
        aggregation_strategy: str = settings.NER_AGGREGATION_STRATEGY,
        score_threshold: float = settings.NER_SCORE_THRESHOLD,
        entity_types: tuple[str, ...] = settings.NER_ENTITY_TYPES,
        device: str = settings.DEVICE,
    ) -> None:
        self._score_threshold = score_threshold
        self._entity_types = frozenset(entity_types)
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

    def detect(self, text: str) -> list[Entity]:
        if not text.strip():
            return []

        raw = self._pipe(text, tokenizer_kwargs={"truncation": True})

        entities: list[Entity] = []
        for r in raw:
            if r["entity_group"] not in self._entity_types:
                continue
            if r["score"] < self._score_threshold:
                continue
            entities.append(
                Entity(
                    text=r["word"],
                    label=r["entity_group"],
                    start_char=r["start"],
                    end_char=r["end"],
                    score=float(r["score"]),
                )
            )
        return entities
