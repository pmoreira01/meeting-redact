"""Audio redaction — replaces entity time spans with silence, a beep, or TTS audio.

All operations are performed on numpy float32 arrays at the pipeline sample
rate (16 kHz).  The output is always the same length as the input — duration
is preserved by in-place replacement, never by splicing.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from meeting_redact.config import settings
from meeting_redact.ner.entity import Entity

if TYPE_CHECKING:
    from meeting_redact.redaction.tts import TTSReplacer


def redact(
    audio: np.ndarray,
    sample_rate: int,
    entities: list[Entity],
    method: str = settings.REDACTION_METHOD,
    padding_ms: int = settings.REDACTION_PADDING_MS,
    tts_replacer: TTSReplacer | None = None,
) -> np.ndarray:
    """Return a copy of *audio* with each entity span replaced per *method*.

    Args:
        audio: 1-D float32 array, mono, at *sample_rate* Hz.
        sample_rate: Audio sample rate in Hz.
        entities: Entities with ``start_time`` / ``end_time`` set (seconds).
        method: ``"silence"``, ``"beep"``, or ``"tts"``.
        padding_ms: Extra milliseconds added to each side of every span.
        tts_replacer: Required when *method* is ``"tts"``.

    Returns:
        New float32 array of the same length as *audio*.
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if method == "tts":
        if tts_replacer is None:
            raise ValueError("tts_replacer must be provided when method='tts'.")
        # Build voice reference once from non-entity regions.
        pad = int(padding_ms * sample_rate / 1_000)
        voice_ref = _extract_voice_reference(
            audio,
            sample_rate,
            entities,
            padding_samples=pad,
            min_sec=settings.TTS_VOICE_REF_MIN_SEC,
            max_sec=settings.TTS_VOICE_REF_MAX_SEC,
        )

    result = audio.copy()
    pad = int(padding_ms * sample_rate / 1_000)

    for entity in entities:
        start = max(0, int(entity.start_time * sample_rate) - pad)
        end = min(len(result), int(entity.end_time * sample_rate) + pad)

        if start >= end:
            continue

        span_len = end - start

        if method == "silence":
            result[start:end] = 0.0
        elif method == "beep":
            result[start:end] = _generate_beep(span_len, sample_rate)
        elif method == "tts":
            spoken = entity.anonymized_as or entity.label.lower()
            replacement = tts_replacer.synthesize(  # type: ignore[union-attr]
                spoken_text=spoken,
                duration_sec=span_len / sample_rate,
                reference_audio=voice_ref,  # type: ignore[possibly-undefined]
                sample_rate=sample_rate,
            )
            result[start:end] = replacement
        else:
            raise ValueError(
                f"Unknown redaction method: {method!r}. "
                "Valid options: 'silence', 'beep', 'tts'."
            )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_beep(n_samples: int, sample_rate: int) -> np.ndarray:
    gain = 10 ** (settings.REDACTION_BEEP_GAIN_DB / 20.0)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    return (gain * np.sin(2.0 * math.pi * settings.REDACTION_BEEP_FREQ_HZ * t)).astype(
        np.float32
    )


def _extract_voice_reference(
    audio: np.ndarray,
    sample_rate: int,
    entities: list[Entity],
    padding_samples: int,
    min_sec: float,
    max_sec: float,
) -> np.ndarray:
    """Collect non-entity audio as a voice reference for TTS cloning.

    Segments outside every entity span (plus padding) are concatenated up to
    *max_sec*.  If the total non-entity audio is shorter than *min_sec*, the
    full audio is used as a fallback — enough for reasonable voice cloning.
    """
    max_samples = round(max_sec * sample_rate)

    if not entities:
        return audio[:max_samples]

    # Build padded, clamped exclusion intervals.
    excluded: list[tuple[int, int]] = []
    for e in entities:
        s = max(0, round(e.start_time * sample_rate) - padding_samples)
        en = min(len(audio), round(e.end_time * sample_rate) + padding_samples)
        excluded.append((s, en))

    # Merge overlapping intervals.
    excluded.sort()
    merged: list[list[int]] = []
    for s, en in excluded:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([s, en])

    # Collect the gaps between excluded intervals.
    segments: list[np.ndarray] = []
    total = 0
    prev = 0
    for s, en in merged:
        if prev < s:
            seg = audio[prev:s]
            segments.append(seg)
            total += len(seg)
        prev = en
    if prev < len(audio):
        seg = audio[prev:]
        segments.append(seg)
        total += len(seg)

    if not segments or total < round(min_sec * sample_rate):
        return audio[:max_samples]

    reference = np.concatenate(segments)
    return reference[:max_samples]
