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
        # DeBERTa-v3 max is 512 tokens; use char budget ≈ 512 * 4 with overlap
        # so entities that span a chunk boundary are still caught.
        max_pos = getattr(self._pipe.model.config, "max_position_embeddings", 512)
        self._chunk_chars = max_pos * 4
        self._overlap_chars = 200

    def detect(self, text: str) -> list[Entity]:
        if not text.strip():
            return []

        entities: list[Entity] = []
        seen: set[tuple[int, int]] = set()

        for chunk, offset in self._chunks(text):
            for r in self._pipe(chunk):
                if r["entity_group"] not in self._entity_types:
                    continue
                if r["score"] < self._score_threshold:
                    continue
                abs_start = r["start"] + offset
                abs_end = r["end"] + offset
                if (abs_start, abs_end) in seen:
                    continue
                seen.add((abs_start, abs_end))
                entities.append(
                    Entity(
                        text=r["word"],
                        label=r["entity_group"],
                        start_char=abs_start,
                        end_char=abs_end,
                        score=float(r["score"]),
                    )
                )
        return entities

    def _chunks(self, text: str) -> list[tuple[str, int]]:
        """Split *text* into overlapping chunks, returning (chunk, char_offset) pairs."""
        if len(text) <= self._chunk_chars:
            return [(text, 0)]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self._chunk_chars, len(text))
            # Break at a word boundary to avoid splitting tokens mid-word.
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunks.append((text[start:end], start))
            start = end - self._overlap_chars
            if start < 0:
                start = 0
        return chunks
