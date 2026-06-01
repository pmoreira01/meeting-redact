from __future__ import annotations

from abc import ABC, abstractmethod

from meeting_redact.ner.entity import Entity


class BaseDetector(ABC):
    """Abstract base for all NER detectors.

    Implementations must be loadable once and callable many times.
    The detect method must be stateless with respect to the detector's
    internals — parallel calls with different texts must be safe.
    """

    @abstractmethod
    def detect(self, text: str) -> list[Entity]:
        """Return all detected entities in *text* above the implementation's threshold."""
        ...
