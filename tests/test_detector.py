"""Tests for the NER detector module.

All tests mock the transformers pipeline so the DeBERTa model is never
downloaded or loaded during the test run.
"""
import pytest
from unittest.mock import MagicMock, patch

from meeting_redact.ner.entity import Entity
from meeting_redact.ner.deberta import DeBERTaDetector
from meeting_redact.ner.ensemble import EnsembleDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PIPELINE_OUTPUT = [
    {"entity_group": "PER", "word": "John Smith", "start": 0,  "end": 10, "score": 0.99},
    {"entity_group": "LOC", "word": "London",     "start": 14, "end": 20, "score": 0.95},
    {"entity_group": "ORG", "word": "Anthropic",  "start": 24, "end": 33, "score": 0.92},
    # MISC — not in NER_ENTITY_TYPES, must be dropped
    {"entity_group": "MISC", "word": "NATO",      "start": 35, "end": 39, "score": 0.91},
    # PER below default threshold (0.85), must be dropped
    {"entity_group": "PER", "word": "Jane",       "start": 44, "end": 48, "score": 0.60},
]

_TEXT = "John Smith in London at Anthropic NATO is Jane"


@pytest.fixture
def detector():
    with patch("meeting_redact.ner.deberta.pipeline") as mock_pipe_cls:
        mock_pipe_instance = MagicMock(return_value=_PIPELINE_OUTPUT)
        mock_pipe_cls.return_value = mock_pipe_instance
        d = DeBERTaDetector(model_name="mock-model", device="cpu")
        yield d


# ---------------------------------------------------------------------------
# DeBERTaDetector — filtering
# ---------------------------------------------------------------------------

def test_detect_count(detector):
    results = detector.detect(_TEXT)
    assert len(results) == 3  # PER, LOC, ORG pass; MISC and low-score PER filtered


def test_detect_filters_misc(detector):
    results = detector.detect(_TEXT)
    assert all(e.label != "MISC" for e in results)


def test_detect_filters_below_threshold(detector):
    results = detector.detect(_TEXT)
    assert all(e.score >= 0.85 for e in results)


# ---------------------------------------------------------------------------
# DeBERTaDetector — entity fields
# ---------------------------------------------------------------------------

def test_detect_per_fields(detector):
    results = detector.detect(_TEXT)
    per = next(e for e in results if e.label == "PER")
    assert per.text == "John Smith"
    assert per.start_char == 0
    assert per.end_char == 10
    assert per.score == pytest.approx(0.99)


def test_detect_loc_fields(detector):
    results = detector.detect(_TEXT)
    loc = next(e for e in results if e.label == "LOC")
    assert loc.text == "London"


def test_detect_org_fields(detector):
    results = detector.detect(_TEXT)
    org = next(e for e in results if e.label == "ORG")
    assert org.text == "Anthropic"


def test_entity_default_times_zero(detector):
    results = detector.detect(_TEXT)
    for e in results:
        assert e.start_time == 0.0
        assert e.end_time == 0.0


# ---------------------------------------------------------------------------
# DeBERTaDetector — empty / whitespace input
# ---------------------------------------------------------------------------

def test_detect_empty_string(detector):
    assert detector.detect("") == []


def test_detect_whitespace_only(detector):
    assert detector.detect("   \t\n") == []


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------

def test_entity_fields():
    e = Entity(text="Alice", label="PER", start_char=0, end_char=5, score=0.97)
    assert e.text == "Alice"
    assert e.label == "PER"
    assert e.start_char == 0
    assert e.end_char == 5
    assert e.score == pytest.approx(0.97)
    assert e.start_time == 0.0
    assert e.end_time == 0.0


def test_entity_time_override():
    e = Entity(text="Alice", label="PER", start_char=0, end_char=5, score=0.97,
               start_time=1.2, end_time=1.8)
    assert e.start_time == pytest.approx(1.2)
    assert e.end_time == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# EnsembleDetector
# ---------------------------------------------------------------------------

def test_ensemble_delegates_to_default():
    expected = [Entity(text="Alice", label="PER", start_char=0, end_char=5, score=0.99)]
    mock_detector = MagicMock(spec=DeBERTaDetector)
    mock_detector.detect.return_value = expected

    ensemble = EnsembleDetector(default_detector=mock_detector)
    results = ensemble.detect("Alice works here")

    mock_detector.detect.assert_called_once_with("Alice works here")
    assert results == expected


def test_ensemble_propagates_empty():
    mock_detector = MagicMock(spec=DeBERTaDetector)
    mock_detector.detect.return_value = []

    ensemble = EnsembleDetector(default_detector=mock_detector)
    assert ensemble.detect("no entities here") == []
