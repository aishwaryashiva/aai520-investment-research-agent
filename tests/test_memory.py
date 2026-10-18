"""Tests for persistent research memory."""

from aai520_research.memory import MemoryStore


def test_notes_round_trip_and_search(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.remember(
        {
            "symbol": "msft",
            "thesis": "Cloud margins remain the swing factor.",
            "lessons": ["Pull the income statement next time."],
            "gaps": [],
            "next_run": {"pull_income_statement": True},
        }
    )
    reloaded = MemoryStore(path)
    notes = reloaded.recall("MSFT")
    assert len(notes) == 1
    assert notes[0]["symbol"] == "MSFT"
    found = reloaded.search("cloud margins")
    assert found[0]["symbol"] == "MSFT"
    assert reloaded.recall("AAPL") == []
