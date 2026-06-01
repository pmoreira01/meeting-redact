"""FastAPI application — /transcribe and /redact endpoints.

Models are loaded once at startup via the ASGI lifespan and stored in
``app.state``.  Every request reads from that shared state; no model is
instantiated per-request.

Redaction response encoding
----------------------------
POST /redact returns the processed audio as a WAV body (audio/wav).
The list of redacted entities is attached as the ``X-Entities`` response
header — a Base64-encoded UTF-8 JSON array — to keep the audio stream
self-contained without base64-bloating the body.

Decode on the client::

    import base64, json
    entities = json.loads(base64.b64decode(response.headers["X-Entities"]))
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from meeting_redact.asr import Transcriber
from meeting_redact.asr.transcriber import Word
from meeting_redact.config import settings
from meeting_redact.ner import DeBERTaDetector
from meeting_redact.ner.entity import Entity
from meeting_redact.redaction import assign_timestamps
from meeting_redact.redaction import redact as _redact_audio
from meeting_redact.api.schemas import EntityOut, TranscribeResponse, WordOut


# ---------------------------------------------------------------------------
# Lifespan — load models once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcriber = Transcriber()
    app.state.detector = DeBERTaDetector()
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Meeting Redaction API",
    description="Detect and redact PII in meeting audio.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV or MP3)"),
):
    """Transcribe *audio* and return the transcript, word timestamps, and detected entities.

    No audio is modified — this is a read-only analysis endpoint.
    """
    _validate_content_type(audio.content_type)
    audio_array = await _read_audio(audio)

    transcriber: Transcriber = request.app.state.transcriber
    detector: DeBERTaDetector = request.app.state.detector

    result = transcriber.transcribe(audio_array)
    entities = detector.detect(result.text)
    entities = assign_timestamps(entities, result.words)

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        words=[_word_out(w) for w in result.words],
        entities=[_entity_out(e) for e in entities],
    )


@app.post("/redact")
async def redact(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV or MP3)"),
    method: str = Form(
        default=settings.REDACTION_METHOD,
        description="Redaction method: 'silence' or 'beep'.",
    ),
    entity_types: str = Form(
        default=",".join(settings.NER_ENTITY_TYPES),
        description="Comma-separated entity labels to redact, e.g. 'PER,ORG'.",
    ),
):
    """Redact PII from *audio* and return the processed WAV.

    The list of redacted entities is in the ``X-Entities`` response header
    (Base64-encoded UTF-8 JSON).
    """
    _validate_content_type(audio.content_type)
    _validate_method(method)

    requested = frozenset(t.strip().upper() for t in entity_types.split(",") if t.strip())

    audio_array = await _read_audio(audio)

    transcriber: Transcriber = request.app.state.transcriber
    detector: DeBERTaDetector = request.app.state.detector

    result = transcriber.transcribe(audio_array)
    entities = detector.detect(result.text)
    entities = [e for e in entities if e.label in requested]
    entities = assign_timestamps(entities, result.words)

    redacted = _redact_audio(audio_array, settings.ASR_SAMPLE_RATE, entities, method=method)
    wav_bytes = _to_wav(redacted, settings.ASR_SAMPLE_RATE)

    entities_header = base64.b64encode(
        json.dumps([_entity_out(e).model_dump() for e in entities]).encode()
    ).decode("ascii")

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "X-Entities": entities_header,
            "Content-Disposition": 'attachment; filename="redacted.wav"',
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_content_type(content_type: str | None) -> None:
    allowed = set(settings.API_ALLOWED_AUDIO_TYPES) | {"application/octet-stream"}
    if content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type {content_type!r}. "
                f"Accepted: {sorted(settings.API_ALLOWED_AUDIO_TYPES)}"
            ),
        )


def _validate_method(method: str) -> None:
    if method not in ("silence", "beep"):
        raise HTTPException(
            status_code=422,
            detail=f"method must be 'silence' or 'beep', got {method!r}.",
        )


async def _read_audio(upload: UploadFile) -> np.ndarray:
    """Read *upload* into a float32 numpy array at ASR_SAMPLE_RATE."""
    import whisperx

    data = await upload.read()
    max_bytes = settings.API_MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {settings.API_MAX_UPLOAD_MB} MB.",
        )

    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return whisperx.load_audio(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono *audio* as 16-bit PCM WAV bytes."""
    pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _word_out(w: Word) -> WordOut:
    return WordOut(text=w.text, start=w.start, end=w.end, score=w.score)


def _entity_out(e: Entity) -> EntityOut:
    return EntityOut(
        text=e.text,
        label=e.label,
        start_char=e.start_char,
        end_char=e.end_char,
        score=e.score,
        start_time=e.start_time,
        end_time=e.end_time,
    )
