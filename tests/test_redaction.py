"""Tests for the redaction stage: mapper and audio modules."""
import numpy as np
import pytest

from meeting_redact.asr.transcriber import Word
from meeting_redact.ner.entity import Entity
from meeting_redact.redaction.mapper import assign_timestamps, _build_word_spans
from meeting_redact.redaction.audio import redact, _generate_beep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, score=1.0)


def make_entity(text: str, label: str, start_char: int, end_char: int) -> Entity:
    return Entity(text=text, label=label, start_char=start_char, end_char=end_char, score=0.99)


def entity_with_times(start_t: float, end_t: float) -> Entity:
    e = make_entity("Alice", "PER", 0, 5)
    return Entity(
        text=e.text, label=e.label,
        start_char=e.start_char, end_char=e.end_char,
        score=e.score, start_time=start_t, end_time=end_t,
    )


# ---------------------------------------------------------------------------
# mapper — _build_word_spans
# ---------------------------------------------------------------------------
# Reconstructed text for WORDS: "John Smith works at Google in London"
#                                 0    5     11    17 20    27 30
# Positions (0-indexed, exclusive end):
#   John   → (0,  4)
#   Smith  → (5,  10)
#   works  → (11, 16)
#   at     → (17, 19)
#   Google → (20, 26)
#   in     → (27, 29)
#   London → (30, 36)

WORDS = [
    make_word("John",   0.0, 0.3),
    make_word("Smith",  0.4, 0.7),
    make_word("works",  0.8, 1.1),
    make_word("at",     1.2, 1.3),
    make_word("Google", 1.4, 1.8),
    make_word("in",     1.9, 2.0),
    make_word("London", 2.1, 2.5),
]


def test_word_spans_offsets():
    spans = _build_word_spans(WORDS)
    starts = [s for s, _, _ in spans]
    ends   = [e for _, e, _ in spans]
    assert starts == [0, 5, 11, 17, 20, 27, 30]
    assert ends   == [4, 10, 16, 19, 26, 29, 36]


def test_word_spans_word_references():
    spans = _build_word_spans(WORDS)
    for i, (_, _, w) in enumerate(spans):
        assert w is WORDS[i]


# ---------------------------------------------------------------------------
# mapper — assign_timestamps (exact overlap)
# ---------------------------------------------------------------------------

def test_map_single_word():
    entity = make_entity("London", "LOC", 30, 36)
    result = assign_timestamps([entity], WORDS)
    assert len(result) == 1
    assert result[0].start_time == pytest.approx(2.1)
    assert result[0].end_time   == pytest.approx(2.5)


def test_map_multi_word_entity():
    # "John Smith" → chars 0–10
    entity = make_entity("John Smith", "PER", 0, 10)
    result = assign_timestamps([entity], WORDS)
    assert len(result) == 1
    assert result[0].start_time == pytest.approx(0.0)
    assert result[0].end_time   == pytest.approx(0.7)


def test_map_middle_entity():
    # "Google" → chars 20–26
    entity = make_entity("Google", "ORG", 20, 26)
    result = assign_timestamps([entity], WORDS)
    assert result[0].start_time == pytest.approx(1.4)
    assert result[0].end_time   == pytest.approx(1.8)


def test_map_preserves_other_fields():
    entity = make_entity("Google", "ORG", 20, 26)
    result = assign_timestamps([entity], WORDS)
    r = result[0]
    assert r.text  == "Google"
    assert r.label == "ORG"
    assert r.score == pytest.approx(0.99)
    assert r.start_char == 20
    assert r.end_char   == 26


def test_map_multiple_entities():
    entities = [
        make_entity("John Smith", "PER", 0,  10),
        make_entity("Google",     "ORG", 20, 26),
        make_entity("London",     "LOC", 30, 36),
    ]
    result = assign_timestamps(entities, WORDS)
    assert len(result) == 3
    labels = [e.label for e in result]
    assert labels == ["PER", "ORG", "LOC"]


def test_map_out_of_range_entity_dropped():
    entity = make_entity("Nonexistent", "PER", 100, 111)
    result = assign_timestamps([entity], WORDS)
    assert result == []


def test_map_empty_entities():
    assert assign_timestamps([], WORDS) == []


def test_map_empty_words_returns_original():
    entity = make_entity("John", "PER", 0, 4)
    result = assign_timestamps([entity], [])
    # No words available → return as-is (times stay 0.0)
    assert result == [entity]


# ---------------------------------------------------------------------------
# mapper — fuzzy fallback
# ---------------------------------------------------------------------------

def test_fuzzy_match_finds_close_word():
    # Slightly misspelled entity text — should still find "London"
    entity = make_entity("Londn", "LOC", 100, 105)  # out-of-range char span forces fuzzy
    result = assign_timestamps([entity], WORDS)
    assert len(result) == 1
    assert result[0].start_time == pytest.approx(2.1)
    assert result[0].end_time   == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# audio — silence
# ---------------------------------------------------------------------------

SR = 16_000
N = SR * 2  # 2-second clip


@pytest.fixture
def ones():
    return np.ones(N, dtype=np.float32)


def test_silence_zeros_span(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="silence", padding_ms=0)
    s, end = int(0.5 * SR), int(1.0 * SR)
    assert np.all(out[s:end] == 0.0)


def test_silence_preserves_before(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="silence", padding_ms=0)
    assert np.all(out[:int(0.5 * SR)] == 1.0)


def test_silence_preserves_after(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="silence", padding_ms=0)
    assert np.all(out[int(1.0 * SR):] == 1.0)


def test_silence_preserves_length(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="silence", padding_ms=0)
    assert len(out) == N


def test_silence_does_not_mutate_input(ones):
    original = ones.copy()
    e = entity_with_times(0.5, 1.0)
    redact(ones, SR, [e], method="silence", padding_ms=0)
    assert np.array_equal(ones, original)


# ---------------------------------------------------------------------------
# audio — beep
# ---------------------------------------------------------------------------

def test_beep_replaces_span(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="beep", padding_ms=0)
    s, end = int(0.5 * SR), int(1.0 * SR)
    # Sine wave — not all zeros, not all ones
    assert not np.all(out[s:end] == 1.0)
    assert not np.all(out[s:end] == 0.0)


def test_beep_preserves_length(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="beep", padding_ms=0)
    assert len(out) == N


def test_beep_preserves_outside(ones):
    e = entity_with_times(0.5, 1.0)
    out = redact(ones, SR, [e], method="beep", padding_ms=0)
    assert np.all(out[:int(0.5 * SR)] == 1.0)
    assert np.all(out[int(1.0 * SR):] == 1.0)


def test_generate_beep_shape_and_dtype():
    beep = _generate_beep(1000, SR)
    assert beep.shape == (1000,)
    assert beep.dtype == np.float32


def test_generate_beep_is_oscillating():
    beep = _generate_beep(SR, SR)  # 1 second
    # A 1 kHz sine crosses zero; values should span both positive and negative
    assert beep.max() > 0
    assert beep.min() < 0


# ---------------------------------------------------------------------------
# audio — padding
# ---------------------------------------------------------------------------

def test_padding_extends_silence(ones):
    e = entity_with_times(1.0, 1.5)
    out = redact(ones, SR, [e], method="silence", padding_ms=50)
    pad = int(0.05 * SR)
    # Padded samples should be zeroed
    assert out[int(1.0 * SR) - pad] == pytest.approx(0.0)
    assert out[int(1.5 * SR) + pad - 1] == pytest.approx(0.0)


def test_padding_clamps_at_start(ones):
    e = entity_with_times(0.0, 0.1)
    out = redact(ones, SR, [e], method="silence", padding_ms=100)
    assert len(out) == N  # no IndexError or truncation


def test_padding_clamps_at_end(ones):
    e = entity_with_times(1.9, 2.0)
    out = redact(ones, SR, [e], method="silence", padding_ms=100)
    assert len(out) == N


# ---------------------------------------------------------------------------
# audio — edge cases
# ---------------------------------------------------------------------------

def test_no_entities_unchanged(ones):
    out = redact(ones, SR, [], method="silence", padding_ms=0)
    assert np.array_equal(out, ones)


def test_invalid_method_raises(ones):
    e = entity_with_times(0.5, 1.0)
    with pytest.raises(ValueError, match="Unknown redaction method"):
        redact(ones, SR, [e], method="tts")


def test_float64_input_converted(ones):
    audio64 = ones.astype(np.float64)
    e = entity_with_times(0.5, 1.0)
    out = redact(audio64, SR, [e], method="silence", padding_ms=0)
    assert out.dtype == np.float32


def test_multiple_entities_all_redacted(ones):
    entities = [entity_with_times(0.2, 0.4), entity_with_times(0.8, 1.0)]
    out = redact(ones, SR, entities, method="silence", padding_ms=0)
    for e in entities:
        s, end = int(e.start_time * SR), int(e.end_time * SR)
        assert np.all(out[s:end] == 0.0)
