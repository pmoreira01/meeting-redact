"""TTS replacement stub — future Kokoro integration.

When implemented: synthesize a spoken label (e.g. "[PERSON]") at the entity's
duration and return a float32 mono array at 16 kHz for seamless splicing.
The interface is intentionally minimal so Kokoro (or any other TTS engine) can
be dropped in without touching the call sites.
"""
from __future__ import annotations

import numpy as np


class TTSReplacer:
    def synthesize(
        self,
        entity_text: str,
        duration_sec: float,
        sample_rate: int = 16_000,
    ) -> np.ndarray:
        """Return audio that speaks *entity_text* (or a redaction label) at *duration_sec*.

        Not yet implemented — requires Kokoro integration.
        Use ``method='silence'`` or ``method='beep'`` in the meantime.
        """
        raise NotImplementedError(
            "TTS replacement is not yet implemented. "
            "Use method='silence' or method='beep'."
        )
