"""Central configuration for the meeting-redaction pipeline.

All model paths, device settings, and thresholds live here.
Pipeline modules must read from this file rather than hardcoding values.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = Path(os.getenv("MEETING_REDACT_MODELS_DIR", PROJECT_ROOT / "models"))
CACHE_DIR: Path = Path(os.getenv("MEETING_REDACT_CACHE_DIR", PROJECT_ROOT / ".cache"))
OUTPUT_DIR: Path = Path(os.getenv("MEETING_REDACT_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE: str = "int8"  # whisperx compute_type; int8 fits int8 on RTX 4070 Ti
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# ASR — WhisperX
# ---------------------------------------------------------------------------
ASR_MODEL: str = "distil-large-v3"
ASR_LANGUAGE: str = "en"
ASR_BATCH_SIZE: int = 16
ASR_SAMPLE_RATE: int = 16_000
ASR_ALIGN_MODEL: str | None = None  # None → whisperx default for the language
ASR_RETURN_WORD_TIMESTAMPS: bool = True


# ---------------------------------------------------------------------------
# NER — DeBERTa-large (CoNLL03)
# ---------------------------------------------------------------------------
NER_MODEL: str = "Gladiator/microsoft-deberta-v3-large_ner_conll2003"
NER_AGGREGATION_STRATEGY: str = "simple"
NER_SCORE_THRESHOLD: float = 0.85
NER_ENTITY_TYPES: tuple[str, ...] = ("PER", "LOC", "ORG")
NER_DEVICE: str = DEVICE

# Ensemble routing — kept here so ensemble.py stays config-driven.
# Maps entity label → model identifier. Future models registered alongside.
NER_ENSEMBLE_ROUTING: dict[str, str] = {
    "PER": NER_MODEL,
    "LOC": NER_MODEL,
    "ORG": NER_MODEL,
}


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
REDACTION_METHOD: str = "silence"  # one of: silence | beep | tts
REDACTION_BEEP_FREQ_HZ: int = 1_000
REDACTION_BEEP_GAIN_DB: float = -6.0
REDACTION_PADDING_MS: int = 50  # extra ms applied to either side of an entity span
REDACTION_PRESERVE_DURATION: bool = True


# ---------------------------------------------------------------------------
# TTS replacement (Kokoro)
# ---------------------------------------------------------------------------
TTS_ENABLED: bool = True   # set False to skip loading the TTS model at startup
TTS_VOICE: str = "af_heart"  # Kokoro voice ID — af_heart, af_bella, am_adam, am_michael …


# ---------------------------------------------------------------------------
# Entity → audio mapping
# ---------------------------------------------------------------------------
MAPPER_FUZZY_MATCH: bool = True
MAPPER_FUZZY_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("MEETING_REDACT_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("MEETING_REDACT_PORT", "8000"))
API_MAX_UPLOAD_MB: int = 200
API_ALLOWED_AUDIO_TYPES: tuple[str, ...] = ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3")
