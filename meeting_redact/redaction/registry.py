"""Session-level entity identity registry.

Maps each unique entity surface form to a consistent, realistic-sounding
pseudonym across all requests in the same server session.  "John Smith"
becomes a fake but plausible name like "David Hartley"; every subsequent
occurrence in any request returns the same pseudonym.  Call :meth:`clear`
to start a fresh session (e.g. a new meeting).

Pseudonyms are generated per entity type using the Faker library:
  PER  → full name      (e.g. "Rachel Simmons")
  LOC  → city           (e.g. "Burlington")
  ORG  → company name   (e.g. "Nexon Solutions")
  PHONE / EMAIL / SSN → type-appropriate fake values

Uniqueness within each label type is guaranteed — two distinct real entities
will never receive the same pseudonym.
"""
from __future__ import annotations

import threading
from typing import Callable

from faker import Faker

_fake = Faker()

# Generators by entity label — each callable returns a new fake string.
_GENERATORS: dict[str, Callable[[], str]] = {
    "PER":   _fake.name,
    "LOC":   _fake.city,
    "ORG":   _fake.company,
    "PHONE": _fake.phone_number,
    "EMAIL": _fake.email,
    "SSN":   _fake.ssn,
}


class EntityRegistry:
    """Thread-safe mapping of entity surface forms to pseudonymous labels.

    Keys are ``(text.lower(), label.upper())`` so minor capitalisation
    differences in the transcript do not create duplicate identities.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mapping: dict[tuple[str, str], str] = {}
        self._used: dict[str, set[str]] = {}  # label → set of assigned pseudonyms

    def get_or_assign(self, text: str, label: str) -> str:
        """Return the stable pseudonym for *text*, assigning one if new."""
        key = (text.strip().lower(), label.upper())
        with self._lock:
            if key not in self._mapping:
                self._mapping[key] = self._unique_pseudonym(label.upper())
            return self._mapping[key]

    def snapshot(self) -> dict[str, str]:
        """Return ``{original_text (LABEL): pseudonym}`` for all known entities."""
        with self._lock:
            return {
                f"{text} ({label})": pseudo
                for (text, label), pseudo in self._mapping.items()
            }

    def clear(self) -> None:
        """Reset all mappings — call at the start of a new meeting session."""
        with self._lock:
            self._mapping.clear()
            self._used.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _unique_pseudonym(self, label: str) -> str:
        """Generate a pseudonym not yet used for this label type."""
        used = self._used.setdefault(label, set())
        generate = _GENERATORS.get(label, lambda: _fake.word())

        for _ in range(200):
            candidate = generate()
            if candidate not in used:
                used.add(candidate)
                return candidate

        # Extremely unlikely fallback (pool exhausted)
        candidate = f"{generate()} {len(used)}"
        used.add(candidate)
        return candidate
