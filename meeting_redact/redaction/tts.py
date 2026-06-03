"""Standard TTS replacement — Chatterbox TTS (no voice cloning).

Synthesizes a spoken anonymization label (e.g. "person one") using
Chatterbox's default voice, then fits the output to the target duration
via time-stretching or zero-padding so the audio timeline is preserved.

The spoken text is supplied by the caller (resolved via EntityRegistry) so
that repeated occurrences of the same entity always produce the same label.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample as scipy_resample

from meeting_redact.config import settings

# Chatterbox TTS outputs at 22050 Hz.
_MODEL_SAMPLE_RATE = 22_050


class TTSReplacer:
    """Standard TTS replacer backed by Chatterbox TTS (default voice).

    The model is loaded once at construction time.  Call :meth:`synthesize`
    per entity to obtain a duration-matched replacement waveform.
    """

    def __init__(self) -> None:
        from chatterbox.tts import ChatterboxTTS  # lazy — heavyweight dep

        self._model = ChatterboxTTS.from_pretrained(device=settings.DEVICE)

    def synthesize(
        self,
        spoken_text: str,
        duration_sec: float,
        sample_rate: int = settings.ASR_SAMPLE_RATE,
    ) -> np.ndarray:
        """Return float32 mono audio of *spoken_text* in the default voice.

        The returned array has exactly ``round(duration_sec * sample_rate)``
        samples — duration is always preserved by stretching or padding.

        Args:
            spoken_text: Text to synthesize, e.g. "person one".  Resolved by
                the caller via :class:`EntityRegistry` for session consistency.
            duration_sec: Target span length in seconds.
            sample_rate: Pipeline sample rate (default 16 kHz).
        """
        target_samples = round(duration_sec * sample_rate)

        wav_tensor = self._model.generate(spoken_text)  # type: ignore[attr-defined]

        wav = wav_tensor.squeeze().cpu().numpy().astype(np.float32)

        if _MODEL_SAMPLE_RATE != sample_rate:
            wav = _resample(wav, _MODEL_SAMPLE_RATE, sample_rate)

        return _fit_to_samples(wav, target_samples)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    n_out = round(len(audio) * to_rate / from_rate)
    return scipy_resample(audio, n_out).astype(np.float32)


def _fit_to_samples(audio: np.ndarray, target: int) -> np.ndarray:
    """Time-stretch *audio* to *target* samples, falling back to pad/truncate."""
    if len(audio) == target:
        return audio

    ratio = len(audio) / target
    if 0.4 <= ratio <= 2.5:
        try:
            import librosa

            stretched = librosa.effects.time_stretch(audio, rate=ratio)
            return _pad_or_truncate(stretched, target)
        except Exception:
            pass

    return _pad_or_truncate(audio, target)


def _pad_or_truncate(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])
