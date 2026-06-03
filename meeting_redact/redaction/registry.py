"""Session-level entity identity registry.

Maps each unique entity surface form to a consistent numbered anonymization
label across all requests in the same server session.  "John Smith" seen for
the first time becomes "person one"; every subsequent occurrence — in any
request — returns the same label.  Call :meth:`clear` to start a fresh
session (e.g. a new meeting).
"""
from __future__ import annotations

import threading


_LABEL_BASE: dict[str, str] = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "PHONE": "phone number",
    "EMAIL": "email",
    "SSN": "social security number",
}

_ORDINALS = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]


def _ordinal(n: int) -> str:
    return _ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else str(n)


class EntityRegistry:
    """Thread-safe mapping of entity surface forms to anonymized labels.

    Keys are ``(text.lower(), label.upper())`` so that minor capitalisation
    differences in the transcript do not create duplicate identities.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mapping: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def get_or_assign(self, text: str, label: str) -> str:
        """Return the stable anonymized label for *text*, assigning one if new."""
        key = (text.strip().lower(), label.upper())
        with self._lock:
            if key not in self._mapping:
                n = self._counters.get(label, 0) + 1
                self._counters[label] = n
                base = _LABEL_BASE.get(label.upper(), label.lower())
                self._mapping[key] = f"{base} {_ordinal(n)}"
            return self._mapping[key]

    def snapshot(self) -> dict[str, str]:
        """Return ``{original_text (LABEL): anonymized_label}`` for all known entities."""
        with self._lock:
            return {
                f"{text} ({label})": anon
                for (text, label), anon in self._mapping.items()
            }

    def clear(self) -> None:
        """Reset all mappings — call at the start of a new meeting session."""
        with self._lock:
            self._mapping.clear()
            self._counters.clear()
