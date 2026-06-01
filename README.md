# meeting-redact

Real-time pipeline for detecting and redacting Personally Identifiable Information (PII) in meeting recordings. Audio goes in, redacted audio + transcript come out.

```
Audio → ASR (transcription + word timestamps) → NER (entity detection) → Redaction (audio editing) → Redacted audio + transcript
```

## Stack

- **Python** 3.12
- **ASR** — WhisperX with `distil-large-v3` (word-level timestamps via forced alignment)
- **NER** — DeBERTa-large fine-tuned on CoNLL03 (`Gladiator/microsoft-deberta-v3-large_ner_conll2003`)
- **Redaction** — NumPy (silence/beep in-place); stdlib `wave` for WAV export; TTS-ready architecture
- **API** — FastAPI + python-multipart
- **GPU** — CUDA 12.x, `compute_type=int8`

## Project structure

```
meeting_redact/
  asr/
    transcriber.py   WhisperX wrapper — transcript + word-level timestamps
  ner/
    entity.py        Entity dataclass (text, label, char offsets, audio times, score)
    detector.py      BaseDetector abstract interface
    deberta.py       DeBERTa-large implementation (transformers pipeline)
    ensemble.py      Ensemble router stub — delegates to DeBERTa, ready for per-type routing
  redaction/
    mapper.py        Maps NER char spans → audio timestamps via word alignment + fuzzy fallback
    audio.py         Silence / beep replacement, preserves duration
    tts.py           TTSReplacer stub (future Kokoro integration)
  api/
    main.py          FastAPI app — /health, /transcribe, /redact
    schemas.py       Pydantic response models
  config/
    settings.py      Single source of truth for model paths, device, thresholds
tests/
  test_detector.py   NER unit tests (mocked pipeline)
  test_redaction.py  Mapper and audio redaction unit tests
  test_api.py        API endpoint tests (mocked models, no GPU required)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU users: install a CUDA-matching `torch` build from <https://pytorch.org> **before** `pip install -r requirements.txt` so pip does not overwrite it with the CPU wheel.

## Configuration

All model paths, thresholds, and device settings live in `meeting_redact/config/settings.py`. Pipeline modules must read from there — no hardcoded values. Override paths via env vars:

- `MEETING_REDACT_MODELS_DIR`
- `MEETING_REDACT_CACHE_DIR`
- `MEETING_REDACT_OUTPUT_DIR`
- `MEETING_REDACT_HOST`, `MEETING_REDACT_PORT`

## API

Start the server:

```bash
uvicorn meeting_redact.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

---

### `GET /health`

```
200 OK   {"status": "ok"}
```

---

### `POST /transcribe`

Transcribe audio and detect entities. No audio is modified.

**Request** — `multipart/form-data`

| Field  | Type | Description               |
|--------|------|---------------------------|
| `audio` | file | WAV or MP3 audio file     |

**Response** — `application/json`

```json
{
  "text": "John Smith called Google yesterday",
  "language": "en",
  "words": [
    {"text": "John",  "start": 0.0, "end": 0.3, "score": 0.99},
    ...
  ],
  "entities": [
    {"text": "John Smith", "label": "PER", "start_char": 0, "end_char": 10,
     "score": 0.99, "start_time": 0.0, "end_time": 0.7},
    ...
  ]
}
```

**Example**

```bash
curl -X POST http://localhost:8000/transcribe \
     -F "audio=@meeting.wav"
```

---

### `POST /redact`

Redact detected entities from audio. Returns the processed WAV file.

**Request** — `multipart/form-data`

| Field          | Type   | Default        | Description                             |
|----------------|--------|----------------|-----------------------------------------|
| `audio`        | file   | —              | WAV or MP3 audio file                   |
| `method`       | string | `silence`      | `silence` or `beep`                     |
| `entity_types` | string | `PER,LOC,ORG`  | Comma-separated labels to redact        |

**Response** — `audio/wav` (streaming)

The response body is the redacted audio. Detected entities are returned in the `X-Entities` response header as a Base64-encoded UTF-8 JSON array.

```bash
# Decode on the command line
curl -s -X POST http://localhost:8000/redact \
     -F "audio=@meeting.wav" \
     -F "method=silence" \
     -F "entity_types=PER,ORG" \
     -o redacted.wav \
     -D - | grep x-entities | cut -d' ' -f2 | base64 -d | python -m json.tool
```

```python
# Decode in Python
import base64, json, requests

resp = requests.post(
    "http://localhost:8000/redact",
    files={"audio": open("meeting.wav", "rb")},
    data={"method": "silence", "entity_types": "PER,ORG"},
)
resp.raise_for_status()
open("redacted.wav", "wb").write(resp.content)
entities = json.loads(base64.b64decode(resp.headers["X-Entities"]))
```

**Status codes**

| Code | Reason                                      |
|------|---------------------------------------------|
| 200  | Success                                     |
| 413  | File exceeds `API_MAX_UPLOAD_MB` (200 MB)   |
| 415  | Unsupported media type                      |
| 422  | Invalid `method` value                      |

## Tests

```bash
pytest
```

All tests mock GPU-bound dependencies (Transcriber, DeBERTaDetector, whisperx) so the suite runs without a GPU or model downloads:

| File | Coverage |
|------|----------|
| `test_detector.py` | Entity filtering, score threshold, fuzzy labels, ensemble delegation |
| `test_redaction.py` | Word-span mapping, fuzzy fallback, silence/beep/padding, duration preservation |
| `test_api.py` | `/health`, `/transcribe`, `/redact` — response schema, entity type filtering, validation |

## Design notes

- Word timestamps are **mandatory** output from ASR — the transcriber raises if alignment yields none.
- All audio in the pipeline is 16 kHz mono float32 numpy.
- Redaction preserves duration: spans are silenced/beeped in place, not removed.
- NER is ensemble-ready: `ensemble.py` routes entity types to different models via `NER_ENSEMBLE_ROUTING` in settings.
- Redaction is TTS-ready: `tts.py` stub accepts entity text + duration, returns audio.
