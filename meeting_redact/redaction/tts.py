"""Standard TTS replacement — Chatterbox TTS (no voice cloning).

Synthesizes a spoken anonymization label (e.g. "person one") using
Chatterbox's default voice, then fits the output to the target duration
via time-stretching or zero-padding so the audio timeline is preserved.

The spoken text is supplied by the caller (resolved via EntityRegistry) so
that repeated occurrences of the same entity always produce the same label.
"""
from __future__ import annotations

import numpy as np

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
            spoken_text: Text to synthesize, e.g. "Rachel Simmons".  Resolved
                by the caller via :class:`EntityRegistry`.
            duration_sec: Target span length in seconds.
            sample_rate: Pipeline sample rate (default 16 kHz).
        """
        target_samples = round(duration_sec * sample_rate)

        wav_tensor = self._model.generate(spoken_text)  # type: ignore[attr-defined]
        wav = wav_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Normalise amplitude — Chatterbox can produce values outside [-1, 1].
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * 0.85

        # High-quality resampling via librosa (avoids FFT ringing from scipy).
        if _MODEL_SAMPLE_RATE != sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=_MODEL_SAMPLE_RATE, target_sr=sample_rate)

        return _fit_to_samples(wav, target_samples)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_to_samples(audio: np.ndarray, target: int) -> np.ndarray:
    """Fit *audio* to exactly *target* samples without aggressive distortion.

    Strategy:
    - TTS shorter than span  → zero-pad (silence after the spoken name).
    - TTS up to 2× longer    → gentle time-stretch to fill the span.
    - TTS more than 2× longer → hard-truncate with a 50 ms fade-out so the
      cut sounds like a natural end rather than a glitch.
    """
    if len(audio) == target:
        return audio

    if len(audio) < target:
        # Pad with silence — name was shorter than the original entity span.
        return np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])

    ratio = len(audio) / target  # > 1: TTS is longer than span

    if ratio <= 2.0:
        try:
            import librosa
            stretched = librosa.effects.time_stretch(audio, rate=ratio)
            # time_stretch result may be ±1 sample off target.
            return _pad_or_truncate(stretched, target)
        except Exception:
            pass

    # TTS much longer than span: take the first target samples and fade out
    # over the last 50 ms to avoid a hard-cut glitch.
    truncated = audio[:target].copy()
    fade_len = min(int(0.05 * target), target)
    if fade_len > 0:
        truncated[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return truncated


def _pad_or_truncate(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])
