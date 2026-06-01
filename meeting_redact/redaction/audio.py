"""Audio redaction — replaces entity time spans with silence or a beep tone.

All operations are performed on numpy float32 arrays at the pipeline sample
rate (16 kHz).  The output is always the same length as the input — duration
is preserved by in-place replacement, never by splicing.
"""
from __future__ import annotations

import math

import numpy as np

from meeting_redact.config import settings
from meeting_redact.ner.entity import Entity


def redact(
    audio: np.ndarray,
    sample_rate: int,
    entities: list[Entity],
    method: str = settings.REDACTION_METHOD,
    padding_ms: int = settings.REDACTION_PADDING_MS,
) -> np.ndarray:
    """Return a copy of *audio* with each entity span replaced per *method*.

    Args:
        audio: 1-D float32 array, mono, at *sample_rate* Hz.
        sample_rate: Audio sample rate in Hz.
        entities: Entities with ``start_time`` / ``end_time`` set (seconds).
        method: ``"silence"`` or ``"beep"``.
        padding_ms: Extra milliseconds added to each side of every span.

    Returns:
        New float32 array of the same length as *audio*.
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

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
        else:
            raise ValueError(
                f"Unknown redaction method: {method!r}. Valid options: 'silence', 'beep'."
            )

    return result


def _generate_beep(n_samples: int, sample_rate: int) -> np.ndarray:
    gain = 10 ** (settings.REDACTION_BEEP_GAIN_DB / 20.0)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    return (gain * np.sin(2.0 * math.pi * settings.REDACTION_BEEP_FREQ_HZ * t)).astype(
        np.float32
    )
