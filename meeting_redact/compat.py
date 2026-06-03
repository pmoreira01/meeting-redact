"""Runtime compatibility patches for library version mismatches.

Import this module before anything that transitively imports torchaudio
(e.g. whisperx / pyannote-audio).

torchaudio 2.5+ removed the old dispatcher-backend system.  Several
top-level names that pyannote-audio 3.4 depends on no longer exist:

  - AudioMetaData   (used as a return-type annotation evaluated at import)
  - list_audio_backends()
  - get_audio_backend()
  - set_audio_backend()

Stubs are added only when absent so the patch is a no-op on older torchaudio.
"""
from __future__ import annotations

import torchaudio


def _patch_torchaudio() -> None:
    # ------------------------------------------------------------------
    # AudioMetaData — used only as a type hint; a dataclass stub is fine.
    # ------------------------------------------------------------------
    if not hasattr(torchaudio, "AudioMetaData"):
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Backend-discovery / selection functions removed in torchaudio 2.5+.
    # soundfile is always available in this environment; returning it as
    # the sole backend satisfies pyannote's backend-selection logic.
    # ------------------------------------------------------------------
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]


_patch_torchaudio()
