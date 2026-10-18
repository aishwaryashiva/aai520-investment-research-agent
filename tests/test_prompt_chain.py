"""Tests for the ingest, preprocess, classify, extract, and summarize chain."""

from aai520_research.heuristic import HeuristicLLM
from aai520_research.prompt_chain import NewsPromptChain


def _articles():
    return [
        {
            "title": "Example Corp beats earnings expectations as cloud growth accelerates",
            "publisher": "Wire",
            "link": "https://example.test/a",
            "summary": "EPS beat the published estimate.",
            "published": "2026-01-15",
        },
        {
            "title": "Example Corp beats earnings expectations as cloud growth accelerates",
            "publisher": "Wire",
            "link": "https://example.test/dup",
            "summary": "Duplicate headline.",
            "published": "2026-01-15",
        },
        {
            "title": "Regulators open a probe into Example cloud licensing",
            "publisher": "Wire",
            "link": "https://example.test/b",
            "summary": "",
            "published": "2026-02-01",
        },
        {
            "title": "Fed holds rates steady and lifts software valuations",
            "publisher": "Wire",
            "link": "https://example.test/c",
            "summary": "",
            "published": "2026-02-02",
        },
        {"title": "Tiny", "publisher": "", "link": "", "summary": "", "published": None},
    ]


def test_chain_keeps_each_stage_and_drops_bad_rows():
    result = NewsPromptChain(HeuristicLLM()).run(_articles())
    assert len(result.ingested) == 5
    assert result.dropped == 2
    assert len(result.preprocessed) == 3
    categories = {item["title"]: item["category"] for item in result.classified}
    assert categories[_articles()[0]["title"]] == "earnings"
    assert "probe" in result.classified[1]["title"].lower()
    assert result.classified[1]["category"] == "legal"
    assert result.classified[1]["sentiment"] == "negative"
    assert result.classified[2]["category"] == "macro"
    assert len(result.facts) == 3
    assert result.brief["item_count"] == 3
    assert "3 headlines" in result.brief["summary"]


def test_empty_news_still_returns_a_brief():
    result = NewsPromptChain(HeuristicLLM()).run([])
    assert result.facts == []
    assert result.brief["item_count"] == 0
    assert "No headlines" in result.brief["summary"]
