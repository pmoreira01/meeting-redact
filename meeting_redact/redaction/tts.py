"""Voice-cloning TTS replacement — Chatterbox TTS integration.

Synthesizes a spoken anonymization label (e.g. "person one") in a voice cloned
from a reference audio clip, then fits the output to the target duration via
time-stretching or zero-padding so the audio timeline is always preserved.

The spoken text is supplied by the caller (resolved via EntityRegistry) so
that repeated occurrences of the same entity always produce the same label.
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample as scipy_resample

from meeting_redact.config import settings

# Chatterbox TTS outputs at 22050 Hz.
_MODEL_SAMPLE_RATE = 22_050


class TTSReplacer:
    """Voice-cloning TTS replacer backed by Chatterbox TTS.

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
        reference_audio: np.ndarray,
        sample_rate: int = settings.ASR_SAMPLE_RATE,
    ) -> np.ndarray:
        """Return float32 mono audio of *spoken_text* in the cloned voice.

        The returned array has exactly ``round(duration_sec * sample_rate)``
        samples — duration is always preserved by stretching or padding.

        Args:
            spoken_text: Text to synthesize, e.g. "person one".  Resolved by
                the caller via :class:`EntityRegistry` for session consistency.
            duration_sec: Target span length in seconds.
            reference_audio: Non-entity audio used as the voice reference.
            sample_rate: Pipeline sample rate (default 16 kHz).
        """
        target_samples = round(duration_sec * sample_rate)

        # Chatterbox requires a WAV file path as the voice prompt.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = Path(f.name)
        try:
            _write_wav(reference_audio, sample_rate, ref_path)
            wav_tensor = self._model.generate(
                spoken_text,
                audio_prompt_path=str(ref_path),
            )  # type: ignore[attr-defined]
        finally:
            ref_path.unlink(missing_ok=True)

        # Chatterbox returns a torch tensor; convert to numpy float32.
        wav = wav_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Resample from model output rate to pipeline rate if they differ.
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
    # Time-stretch when the speed-up/slow-down is within a natural range.
    if 0.4 <= ratio <= 2.5:
        try:
            import librosa

            stretched = librosa.effects.time_stretch(audio, rate=ratio)
            return _pad_or_truncate(stretched, target)
        except Exception:
            pass  # fall through on any librosa failure

    return _pad_or_truncate(audio, target)


def _pad_or_truncate(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=np.float32)])


def _write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
