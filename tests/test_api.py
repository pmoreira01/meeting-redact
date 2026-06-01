"""Tests for the FastAPI layer.

All heavy dependencies (Transcriber, DeBERTaDetector, whisperx) are mocked so
tests run without GPU, model downloads, or audio I/O.
"""
from __future__ import annotations

import base64
import io
import json
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from meeting_redact.api.main import app
from meeting_redact.asr.transcriber import TranscriptionResult, Word
from meeting_redact.ner.entity import Entity


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# "John Smith called Google"
#  0    5     11    18
_WORDS = [
    Word(text="John",   start=0.0, end=0.3, score=1.0),
    Word(text="Smith",  start=0.4, end=0.7, score=1.0),
    Word(text="called", start=0.8, end=1.1, score=1.0),
    Word(text="Google", start=1.2, end=1.5, score=1.0),
]

_TRANSCRIPT = TranscriptionResult(
    text="John Smith called Google",
    words=_WORDS,
    language="en",
)

# Entity char offsets align with _TRANSCRIPT.text:
#   "John Smith" → 0..10   "Google" → 18..24
_ENTITIES_RAW = [
    Entity(text="John Smith", label="PER", start_char=0,  end_char=10, score=0.99),
    Entity(text="Google",     label="ORG", start_char=18, end_char=24, score=0.95),
]

# 1-second clip of silence at 16 kHz — returned by the mocked _read_audio
_MOCK_AUDIO = np.zeros(16_000, dtype=np.float32)


def _make_mocks():
    t = MagicMock()
    t.transcribe.return_value = _TRANSCRIPT
    d = MagicMock()
    d.detect.return_value = _ENTITIES_RAW
    return t, d


@pytest.fixture
def client():
    mock_t, mock_d = _make_mocks()
    with (
        patch("meeting_redact.api.main.Transcriber", return_value=mock_t),
        patch("meeting_redact.api.main.DeBERTaDetector", return_value=mock_d),
        patch("meeting_redact.api.main._read_audio", new=AsyncMock(return_value=_MOCK_AUDIO)),
    ):
        with TestClient(app) as c:
            yield c


def _wav_bytes(n_samples: int = 160, sample_rate: int = 16_000) -> bytes:
    """Minimal valid WAV — stdlib only, no pydub."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /transcribe — happy path
# ---------------------------------------------------------------------------

def test_transcribe_status(client):
    resp = client.post("/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert resp.status_code == 200


def test_transcribe_text(client):
    resp = client.post("/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    body = resp.json()
    assert body["text"] == "John Smith called Google"
    assert body["language"] == "en"


def test_transcribe_words(client):
    resp = client.post("/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    words = resp.json()["words"]
    assert len(words) == 4
    assert words[0] == {"text": "John", "start": 0.0, "end": 0.3, "score": 1.0}


def test_transcribe_entity_count(client):
    resp = client.post("/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert len(resp.json()["entities"]) == 2


def test_transcribe_entity_times_mapped(client):
    """assign_timestamps must populate start_time / end_time from word alignment."""
    resp = client.post("/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    entities = resp.json()["entities"]
    per = next(e for e in entities if e["label"] == "PER")
    assert per["start_time"] == pytest.approx(0.0)
    assert per["end_time"]   == pytest.approx(0.7)
    org = next(e for e in entities if e["label"] == "ORG")
    assert org["start_time"] == pytest.approx(1.2)
    assert org["end_time"]   == pytest.approx(1.5)


def test_transcribe_entity_char_offsets(client):
    resp = client.post("/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    per = next(e for e in resp.json()["entities"] if e["label"] == "PER")
    assert per["start_char"] == 0
    assert per["end_char"]   == 10


# ---------------------------------------------------------------------------
# POST /transcribe — validation
# ---------------------------------------------------------------------------

def test_transcribe_rejects_non_audio(client):
    resp = client.post("/transcribe", files={"audio": ("f.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_transcribe_accepts_octet_stream(client):
    # Browsers / curl often send application/octet-stream for binary uploads.
    resp = client.post(
        "/transcribe",
        files={"audio": ("a.wav", _wav_bytes(), "application/octet-stream")},
    )
    assert resp.status_code == 200


def test_transcribe_rejects_video(client):
    resp = client.post("/transcribe", files={"audio": ("v.mp4", b"x", "video/mp4")})
    assert resp.status_code == 415


# ---------------------------------------------------------------------------
# POST /redact — happy path
# ---------------------------------------------------------------------------

def test_redact_status(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    assert resp.status_code == 200


def test_redact_content_type(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    assert resp.headers["content-type"].startswith("audio/wav")


def test_redact_content_disposition(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    assert "redacted.wav" in resp.headers.get("content-disposition", "")


def test_redact_body_is_valid_wav(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    buf = io.BytesIO(resp.content)
    with wave.open(buf) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_redact_x_entities_header_present(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    assert "x-entities" in resp.headers


def test_redact_x_entities_decodes(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    entities = json.loads(base64.b64decode(resp.headers["x-entities"]))
    assert len(entities) == 2
    assert {e["label"] for e in entities} == {"PER", "ORG"}


def test_redact_entity_times_in_header(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence"},
    )
    entities = json.loads(base64.b64decode(resp.headers["x-entities"]))
    per = next(e for e in entities if e["label"] == "PER")
    assert per["start_time"] == pytest.approx(0.0)
    assert per["end_time"]   == pytest.approx(0.7)


def test_redact_beep_method(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "beep"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /redact — entity_types filter
# ---------------------------------------------------------------------------

def test_redact_filter_per_only(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence", "entity_types": "PER"},
    )
    entities = json.loads(base64.b64decode(resp.headers["x-entities"]))
    assert all(e["label"] == "PER" for e in entities)
    assert len(entities) == 1


def test_redact_filter_org_only(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence", "entity_types": "ORG"},
    )
    entities = json.loads(base64.b64decode(resp.headers["x-entities"]))
    assert all(e["label"] == "ORG" for e in entities)


def test_redact_filter_unknown_type_yields_empty(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "silence", "entity_types": "EMAIL"},
    )
    entities = json.loads(base64.b64decode(resp.headers["x-entities"]))
    assert entities == []


# ---------------------------------------------------------------------------
# POST /redact — validation
# ---------------------------------------------------------------------------

def test_redact_invalid_method(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"method": "tts"},
    )
    assert resp.status_code == 422


def test_redact_rejects_non_audio(client):
    resp = client.post(
        "/redact",
        files={"audio": ("f.txt", b"hello", "text/plain")},
        data={"method": "silence"},
    )
    assert resp.status_code == 415


def test_redact_accepts_mp3_content_type(client):
    resp = client.post(
        "/redact",
        files={"audio": ("a.mp3", _wav_bytes(), "audio/mpeg")},
        data={"method": "silence"},
    )
    assert resp.status_code == 200
