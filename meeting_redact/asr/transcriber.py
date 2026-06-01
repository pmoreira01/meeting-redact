"""WhisperX wrapper — produces transcripts with word-level timestamps.

Word timestamps are mandatory: callers downstream rely on them to map
NER character spans to audio time spans for redaction. If alignment
fails or returns no word-level data, this module raises.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import whisperx

from meeting_redact.config import settings


def _ensure_cudnn_path() -> None:
    """Prepend the nvidia-cudnn-cu12 lib dir to LD_LIBRARY_PATH if installed.

    PyTorch 2.6 ships with cuDNN 9, but pyannote's bundled VAD checkpoint
    requires cuDNN 8 (libcudnn_ops_infer.so.8). Installing nvidia-cudnn-cu12
    provides it; this ensures the dynamic linker can find it without requiring
    manual LD_LIBRARY_PATH export before every server start.
    """
    try:
        import nvidia.cudnn
        lib_dir = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in current:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
    except ImportError:
        pass


@contextlib.contextmanager
def _unsafe_load():
    """Temporarily revert torch.load to weights_only=False for pyannote checkpoints.

    PyTorch 2.6 defaulted weights_only=True, breaking pyannote's VAD checkpoint
    which embeds omegaconf types. Patch is scoped to model loading only.
    """
    original = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


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

        _ensure_cudnn_path()
        with _unsafe_load():
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
