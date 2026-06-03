"""DeBERTa-large NER detector (CoNLL03 fine-tune).

Model: Gladiator/microsoft-deberta-v3-large_ner_conll2003
Labels: PER, LOC, ORG, MISC  — MISC is dropped by default (not in NER_ENTITY_TYPES).

Uses AutoTokenizer + AutoModelForTokenClassification directly to avoid importing
transformers.pipelines, which unconditionally pulls in torchvision regardless of
whether image pipelines are used.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from meeting_redact.config import settings
from meeting_redact.ner.detector import BaseDetector
from meeting_redact.ner.entity import Entity


def _patch_deberta_torchjit() -> None:
    """Replace make_log_bucket_position with a plain Python equivalent.

    The transformers version is decorated with @torch.jit.script, which causes
    the CUDA JIT compiler to emit fabs(int64_t) — ambiguous under CUDA 12.x.
    Swapping it for a regular function avoids the CUDA kernel path entirely
    while keeping identical numerics and GPU tensor support.
    """
    try:
        import transformers.models.deberta_v2.modeling_deberta_v2 as _m

        def _make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
            sign = torch.sign(relative_pos)
            mid = bucket_size // 2
            abs_pos = torch.where(
                (relative_pos < mid) & (relative_pos > -mid),
                torch.tensor(mid - 1).type_as(relative_pos),
                torch.abs(relative_pos),
            )
            log_pos = (
                torch.ceil(
                    torch.log(abs_pos.float() / mid)
                    / math.log((max_position - 1) / mid)
                    * (bucket_size // 2 - 1)
                )
                + mid
            )
            bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
            return bucket_pos.long()

        _m.make_log_bucket_position = _make_log_bucket_position
    except (ImportError, AttributeError):
        pass


class DeBERTaDetector(BaseDetector):
    """Token-classification pipeline backed by DeBERTa-v3-large.

    Inference is performed directly with AutoModelForTokenClassification and
    AutoTokenizer so that transformers.pipelines (and its torchvision import)
    is never loaded.  BIO aggregation (simple strategy) is implemented here.
    """

    def __init__(
        self,
        model_name: str = settings.NER_MODEL,
        aggregation_strategy: str = settings.NER_AGGREGATION_STRATEGY,
        score_threshold: float = settings.NER_SCORE_THRESHOLD,
        entity_types: tuple[str, ...] = settings.NER_ENTITY_TYPES,
        device: str = settings.NER_DEVICE,
    ) -> None:
        self._score_threshold = score_threshold
        self._entity_types = frozenset(entity_types)

        _patch_deberta_torchjit()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name)

        actual_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._device = torch.device(actual_device)
        self._model.to(self._device).eval()

        self._id2label: dict[int, str] = self._model.config.id2label

        # DeBERTa-v3 max is 512 tokens; use char budget ≈ 512 * 4 with overlap
        # so entities that span a chunk boundary are still caught.
        max_pos = getattr(self._model.config, "max_position_embeddings", 512)
        self._max_tokens = max_pos - 2  # reserve for [CLS] / [SEP]
        self._chunk_chars = self._max_tokens * 4
        self._overlap_chars = 200

    def detect(self, text: str) -> list[Entity]:
        if not text.strip():
            return []

        entities: list[Entity] = []
        seen: set[tuple[int, int]] = set()

        for chunk, offset in self._chunks(text):
            for entity in self._run_chunk(chunk, offset):
                key = (entity.start_char, entity.end_char)
                if key not in seen:
                    seen.add(key)
                    entities.append(entity)

        return entities

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _run_chunk(self, text: str, char_offset: int) -> list[Entity]:
        """Run inference on one chunk and return Entity objects."""
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self._max_tokens + 2,
        )
        offset_mapping: list[tuple[int, int]] = encoding.pop("offset_mapping")[0].tolist()

        inputs = {k: v.to(self._device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]  # (seq_len, num_labels)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids: np.ndarray = probs.argmax(axis=-1)

        raw: list[Entity] = []
        current: dict | None = None

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:  # special token ([CLS], [SEP], padding)
                if current is not None:
                    raw.append(self._close(current, text, char_offset))
                    current = None
                continue

            label_full = self._id2label[int(pred_ids[i])]
            score = float(probs[i, int(pred_ids[i])])

            if label_full == "O":
                if current is not None:
                    raw.append(self._close(current, text, char_offset))
                    current = None
                continue

            # Parse optional BIO prefix (B-PER, I-PER, or bare PER).
            if "-" in label_full:
                bio, entity_type = label_full.split("-", 1)
            else:
                bio, entity_type = "B", label_full

            if current is None or bio == "B" or entity_type != current["label"]:
                if current is not None:
                    raw.append(self._close(current, text, char_offset))
                current = {
                    "label": entity_type,
                    "start": tok_start,
                    "end": tok_end,
                    "scores": [score],
                }
            else:
                current["end"] = tok_end
                current["scores"].append(score)

        if current is not None:
            raw.append(self._close(current, text, char_offset))

        return [
            e for e in raw
            if e.label in self._entity_types and e.score >= self._score_threshold
        ]

    @staticmethod
    def _close(current: dict, text: str, char_offset: int) -> Entity:
        span_text = text[current["start"]:current["end"]].strip()
        return Entity(
            text=span_text,
            label=current["label"],
            start_char=current["start"] + char_offset,
            end_char=current["end"] + char_offset,
            score=float(np.mean(current["scores"])),
        )

    def _chunks(self, text: str) -> list[tuple[str, int]]:
        """Split *text* into overlapping chunks, returning (chunk, char_offset) pairs."""
        if len(text) <= self._chunk_chars:
            return [(text, 0)]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self._chunk_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunks.append((text[start:end], start))
            start = end - self._overlap_chars
            if start < 0:
                start = 0
        return chunks
