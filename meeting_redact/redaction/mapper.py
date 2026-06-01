"""Maps NER entity character spans to audio timestamps via word-level alignment.

The transcript is reconstructed as ``" ".join(w.text for w in words)`` — the
same string that was fed to the NER model — so character offsets from the NER
pipeline align directly with the positions we compute here.  Fuzzy matching is
available as a fallback for minor tokenisation differences between the ASR
output and the NER tokeniser.
"""
from __future__ import annotations

import dataclasses
import difflib

from meeting_redact.asr.transcriber import Word
from meeting_redact.config import settings
from meeting_redact.ner.entity import Entity


def assign_timestamps(entities: list[Entity], words: list[Word]) -> list[Entity]:
    """Return entities with ``start_time`` / ``end_time`` populated.

    Entities whose character span cannot be matched to any word are dropped.
    When *words* is empty the original list is returned unchanged (times stay
    at their default 0.0 — the caller is responsible for handling this).
    """
    if not words or not entities:
        return entities

    word_spans = _build_word_spans(words)
    result: list[Entity] = []
    for entity in entities:
        mapped = _map_entity(entity, word_spans, words)
        if mapped is not None:
            result.append(mapped)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_word_spans(words: list[Word]) -> list[tuple[int, int, Word]]:
    """Compute ``(start_char, end_char, word)`` for each word in joined text."""
    spans: list[tuple[int, int, Word]] = []
    pos = 0
    for word in words:
        end = pos + len(word.text)
        spans.append((pos, end, word))
        pos = end + 1  # single space separator between words
    return spans


def _map_entity(
    entity: Entity,
    word_spans: list[tuple[int, int, Word]],
    words: list[Word],
) -> Entity | None:
    overlapping = [
        w
        for start, end, w in word_spans
        if start < entity.end_char and end > entity.start_char
    ]

    if not overlapping and settings.MAPPER_FUZZY_MATCH:
        overlapping = _fuzzy_match(entity, words)

    if not overlapping:
        return None

    return dataclasses.replace(
        entity,
        start_time=min(w.start for w in overlapping),
        end_time=max(w.end for w in overlapping),
    )


def _fuzzy_match(entity: Entity, words: list[Word]) -> list[Word]:
    """Slide a window over *words* and return the best-matching sequence."""
    entity_lower = entity.text.lower()
    n_tokens = max(1, len(entity_lower.split()))
    word_texts = [w.text.lower() for w in words]

    best_ratio = 0.0
    best_start = 0
    best_end = 0

    # Window size: allow ±2 words around the expected token count.
    for i in range(len(words)):
        for j in range(i + 1, min(i + n_tokens + 3, len(words) + 1)):
            candidate = " ".join(word_texts[i:j])
            ratio = difflib.SequenceMatcher(None, entity_lower, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = j

    if best_ratio >= settings.MAPPER_FUZZY_THRESHOLD:
        return list(words[best_start:best_end])
    return []
