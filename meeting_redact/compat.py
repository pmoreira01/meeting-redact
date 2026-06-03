"""Runtime compatibility patches for library version mismatches.

Import this module before anything that transitively imports torchaudio
(e.g. whisperx / pyannote-audio).

torchaudio 2.5+ removed AudioMetaData from the top-level namespace.
pyannote-audio 3.4 references it as a return-type annotation which Python
evaluates at class-definition time, crashing on import.  A dataclass stub
with the same field names is sufficient — pyannote only uses it as a type
hint, not in isinstance checks.
"""
from __future__ import annotations

import torchaudio


def _patch_torchaudio() -> None:
    if hasattr(torchaudio, "AudioMetaData"):
        return

    from dataclasses import dataclass

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]


_patch_torchaudio()
