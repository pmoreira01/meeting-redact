"""Tests for the ASR transcriber module.

whisperx (ASR + alignment models) is mocked throughout so tests run without
a GPU, model downloads, or real audio files.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_redact.asr.transcriber import Transcriber, TranscriptionResult, Word


# ---------------------------------------------------------------------------
# Canonical mock whisperx output shapes
# ---------------------------------------------------------------------------

_TWO_WORDS = {
    "segments": [
        {
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95},
                {"word": "world", "start": 0.5, "end": 0.9, "score": 0.92},
            ]
        }
    ]
}

_MULTI_SEGMENT = {
    "segments": [
        {"words": [{"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95}]},
        {"words": [{"word": "world", "start": 0.5, "end": 0.9, "score": 0.92}]},
    ]
}

_WITH_UNTIMED = {
    "segments": [
        {
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95},
                {"word": "..."},                # no start/end — must be skipped
                {"word": "world", "start": 0.5, "end": 0.9, "score": 0.92},
            ]
        }
    ]
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tx():
    """Transcriber with whisperx fully mocked out."""
    with patch("meeting_redact.asr.transcriber.whisperx") as mock_wx:
        mock_wx.load_model.return_value = MagicMock()
        mock_wx.load_align_model.return_value = (MagicMock(), {})
        t = Transcriber(model_name="mock-model", device="cpu")
        t._wx = mock_wx          # convenience handle for per-test setup
        yield t


def _audio(n: int = 16_000) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# __init__ — model loading
# ---------------------------------------------------------------------------

def test_init_calls_load_model(tx):
    tx._wx.load_model.assert_called_once()


def test_init_calls_load_align_model(tx):
    tx._wx.load_align_model.assert_called_once()


def test_init_stores_language(tx):
    assert tx.language == "en"


# ---------------------------------------------------------------------------
# _load_audio
# ---------------------------------------------------------------------------

def test_load_audio_float32_returned_as_is():
    arr = np.ones(100, dtype=np.float32)
    out = Transcriber._load_audio(arr)
    assert out is arr


def test_load_audio_float64_promoted():
    arr = np.zeros(100, dtype=np.float64)
    out = Transcriber._load_audio(arr)
    assert out.dtype == np.float32


def test_load_audio_int16_promoted():
    arr = np.zeros(100, dtype=np.int16)
    out = Transcriber._load_audio(arr)
    assert out.dtype == np.float32


def test_load_audio_str_calls_whisperx():
    with patch("meeting_redact.asr.transcriber.whisperx") as mock_wx:
        mock_wx.load_audio.return_value = np.zeros(100, dtype=np.float32)
        Transcriber._load_audio("recording.wav")
        mock_wx.load_audio.assert_called_once_with("recording.wav")


def test_load_audio_path_coerced_to_str():
    with patch("meeting_redact.asr.transcriber.whisperx") as mock_wx:
        mock_wx.load_audio.return_value = np.zeros(100, dtype=np.float32)
        Transcriber._load_audio(Path("recording.wav"))
        mock_wx.load_audio.assert_called_once_with("recording.wav")


# ---------------------------------------------------------------------------
# _collect_words
# ---------------------------------------------------------------------------

def test_collect_words_count():
    words = Transcriber._collect_words(_TWO_WORDS)
    assert len(words) == 2


def test_collect_words_fields():
    words = Transcriber._collect_words(_TWO_WORDS)
    assert words[0].text  == "Hello"
    assert words[0].start == pytest.approx(0.0)
    assert words[0].end   == pytest.approx(0.4)
    assert words[0].score == pytest.approx(0.95)


def test_collect_words_skips_untimed():
    words = Transcriber._collect_words(_WITH_UNTIMED)
    assert len(words) == 2
    assert all(w.text in ("Hello", "world") for w in words)


def test_collect_words_multi_segment_flattened():
    words = Transcriber._collect_words(_MULTI_SEGMENT)
    assert len(words) == 2
    assert words[0].text == "Hello"
    assert words[1].text == "world"


def test_collect_words_empty_segments():
    assert Transcriber._collect_words({"segments": []}) == []


def test_collect_words_missing_key():
    assert Transcriber._collect_words({}) == []


def test_collect_words_score_defaults_zero():
    aligned = {"segments": [{"words": [{"word": "hi", "start": 0.0, "end": 0.2}]}]}
    words = Transcriber._collect_words(aligned)
    assert words[0].score == pytest.approx(0.0)


def test_collect_words_text_stripped():
    aligned = {"segments": [{"words": [{"word": " hi ", "start": 0.0, "end": 0.2, "score": 1.0}]}]}
    words = Transcriber._collect_words(aligned)
    assert words[0].text == "hi"


def test_collect_words_returns_word_instances():
    words = Transcriber._collect_words(_TWO_WORDS)
    assert all(isinstance(w, Word) for w in words)


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

def test_transcribe_returns_transcription_result(tx):
    tx._wx.align.return_value = _TWO_WORDS
    tx._model.transcribe.return_value = {"segments": []}
    result = tx.transcribe(_audio())
    assert isinstance(result, TranscriptionResult)


def test_transcribe_text_joined_by_space(tx):
    tx._wx.align.return_value = _TWO_WORDS
    tx._model.transcribe.return_value = {"segments": []}
    result = tx.transcribe(_audio())
    assert result.text == "Hello world"


def test_transcribe_language_propagated(tx):
    tx._wx.align.return_value = _TWO_WORDS
    tx._model.transcribe.return_value = {"segments": []}
    result = tx.transcribe(_audio())
    assert result.language == tx.language


def test_transcribe_words_populated(tx):
    tx._wx.align.return_value = _TWO_WORDS
    tx._model.transcribe.return_value = {"segments": []}
    result = tx.transcribe(_audio())
    assert len(result.words) == 2


def test_transcribe_raises_on_empty_alignment(tx):
    tx._wx.align.return_value = {"segments": []}
    tx._model.transcribe.return_value = {"segments": []}
    with pytest.raises(RuntimeError, match="word-level timestamps"):
        tx.transcribe(_audio())


def test_transcribe_passes_audio_array_to_model(tx):
    tx._wx.align.return_value = _TWO_WORDS
    tx._model.transcribe.return_value = {"segments": []}
    audio = _audio()
    tx.transcribe(audio)
    passed = tx._model.transcribe.call_args.args[0]
    assert np.array_equal(passed, audio)


def test_transcribe_calls_align(tx):
    tx._wx.align.return_value = _TWO_WORDS
    tx._model.transcribe.return_value = {"segments": []}
    tx.transcribe(_audio())
    tx._wx.align.assert_called_once()
