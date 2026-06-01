"""WhisperX wrapper — produces transcripts with word-level timestamps.

Word timestamps are mandatory: callers downstream rely on them to map
NER character spans to audio time spans for redaction. If alignment
fails or returns no word-level data, this module raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import whisperx

from meeting_redact.config import settings


AudioInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    words: list[Word]
    language: str


class Transcriber:
    """WhisperX-backed transcriber with forced word alignment.

    The ASR and alignment models are loaded once at construction and
    reused across calls — see CLAUDE.md (load once at startup).
    """

    def __init__(
        self,
        model_name: str = settings.ASR_MODEL,
        device: str = settings.DEVICE,
        compute_type: str = settings.COMPUTE_TYPE,
        language: str = settings.ASR_LANGUAGE,
        batch_size: int = settings.ASR_BATCH_SIZE,
    ) -> None:
        self.device = device
        self.language = language
        self.batch_size = batch_size

        self._model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=settings.ASR_ALIGN_MODEL,
        )

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        audio_array = self._load_audio(audio)

        asr_result = self._model.transcribe(
            audio_array,
            batch_size=self.batch_size,
            language=self.language,
        )

        aligned = whisperx.align(
            asr_result["segments"],
            self._align_model,
            self._align_metadata,
            audio_array,
            self.device,
            return_char_alignments=False,
        )

        words = self._collect_words(aligned)
        if not words:
            raise RuntimeError(
                "WhisperX alignment returned no word-level timestamps; "
                "word timestamps are required for downstream redaction."
            )

        text = " ".join(w.text for w in words)
        return TranscriptionResult(text=text, words=words, language=self.language)

    @staticmethod
    def _load_audio(audio: AudioInput) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio
        return whisperx.load_audio(str(audio))

    @staticmethod
    def _collect_words(aligned: dict) -> list[Word]:
        words: list[Word] = []
        for segment in aligned.get("segments", []):
            for w in segment.get("words", []):
                if "start" not in w or "end" not in w:
                    # whisperx occasionally emits a word with no timing
                    # (e.g. punctuation-only); skip — the caller decides.
                    continue
                words.append(
                    Word(
                        text=w["word"].strip(),
                        start=float(w["start"]),
                        end=float(w["end"]),
                        score=float(w.get("score", 0.0)),
                    )
                )
        return words
