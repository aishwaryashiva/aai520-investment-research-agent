"""Persistent research notes so later runs can learn from earlier ones."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class MemoryError(RuntimeError):
    """Raised when the memory file exists but cannot be read safely."""


class MemoryStore:
    """A small JSON notebook of prior theses, gaps, and next-run instructions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.notes = self._load()

    @classmethod
    def default(cls) -> "MemoryStore":
        return cls(Path("data") / "memory" / "research_memory.json")

    def recall(self, symbol: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return the most recent notes for one symbol, oldest first."""
        matched = [
            note
            for note in self.notes
            if str(note.get("symbol", "")).upper() == symbol.upper()
        ]
        return matched[-limit:]

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve notes whose text overlaps the query tokens."""
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not tokens:
            return []
        ranked: list[tuple[int, dict[str, Any]]] = []
        for note in self.notes:
            haystack = json.dumps(note, default=str).lower()
            score = sum(1 for token in tokens if token in haystack)
            if score:
                ranked.append((score, note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in ranked[:limit]]

    def remember(self, note: dict[str, Any]) -> dict[str, Any]:
        """Append one note and write the file."""
        if not note.get("symbol"):
            raise MemoryError("A memory note needs a symbol.")
        stored = dict(note)
        stored["symbol"] = str(stored["symbol"]).upper()
        self.notes.append(stored)
        self._save()
        return stored

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MemoryError(f"Could not read memory file {self.path}.") from exc
        if not isinstance(data, list):
            raise MemoryError("Memory file must contain a JSON list.")
        return [item for item in data if isinstance(item, dict)]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.notes, indent=2, ensure_ascii=False)
        self.path.write_text(text + "\n", encoding="utf-8")
