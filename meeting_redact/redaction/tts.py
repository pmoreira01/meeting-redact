"""Standard TTS replacement — Kokoro ONNX.

Synthesizes spoken anonymization labels (e.g. "Rachel Simmons") using
Kokoro's neural voices via ONNX Runtime.  No voice cloning — consistent,
high-quality output on CPU without GPU or iterative diffusion sampling.

Model weights are downloaded automatically from HuggingFace on first use
and cached in the HuggingFace cache directory.
"""
from __future__ import annotations

import numpy as np

from meeting_redact.config import settings

# Kokoro ONNX outputs at 24 kHz.
_MODEL_SAMPLE_RATE = 24_000


class TTSReplacer:
    """TTS replacer backed by Kokoro ONNX (default neural voice).

    Loaded once at startup.  Call :meth:`synthesize` per entity.
    """

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro  # lazy — downloads weights on first call

        self._kokoro = Kokoro.from_pretrained()
        self._voice = settings.TTS_VOICE

    def synthesize(
        self,
        spoken_text: str,
        duration_sec: float,
        sample_rate: int = settings.ASR_SAMPLE_RATE,
    ) -> np.ndarray:
        """Return float32 mono audio of *spoken_text* fitted to *duration_sec*.

        Args:
            spoken_text: Text to synthesize, e.g. "Rachel Simmons".
            duration_sec: Target span length in seconds.
            sample_rate: Pipeline sample rate (default 16 kHz).
        """
        target_samples = round(duration_sec * sample_rate)

        samples, sr = self._kokoro.create(
            spoken_text,
            voice=self._voice,
            speed=1.0,
            lang="en-us",
        )
        wav = np.array(samples, dtype=np.float32)

        # Normalise — prevent clipping when converting to int16 later.
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * 0.85

        # Resample from model rate to pipeline rate using high-quality resampler.
        if sr != sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)

        return _fit_to_samples(wav, target_samples)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_to_samples(audio: np.ndarray, target: int) -> np.ndarray:
    """Fit *audio* to exactly *target* samples without aggressive distortion.

    - Shorter than span  → zero-pad (silence after speech).
    - Up to 2× longer   → gentle time-stretch via librosa.
    - More than 2× longer → truncate with 50 ms fade-out to avoid hard cut.
    """
    if len(audio) == target:
        return audio

    if len(audio) < target:
        return np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])

    ratio = len(audio) / target  # > 1: TTS longer than span

    if ratio <= 2.0:
        try:
            import librosa
            stretched = librosa.effects.time_stretch(audio, rate=ratio)
            return _pad_or_truncate(stretched, target)
        except Exception:
            pass

    # Too long to stretch cleanly — truncate with a short fade-out.
    truncated = audio[:target].copy()
    fade_len = min(int(0.05 * target), target)
    if fade_len > 0:
        truncated[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return truncated


def _pad_or_truncate(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])
