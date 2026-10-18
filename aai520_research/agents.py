"""The three specialist agents the orchestrator delegates to.

The news agent owns the headline chain. The earnings agent owns statements
and ratios. The market agent owns the chart, including moving averages and RSI.
They do not call each other. The orchestrator hands each one a packet and
collects the notes.
"""

from __future__ import annotations

import json
from typing import Any

from aai520_research.models import ChainResult
from aai520_research.prompt_chain import NewsPromptChain
from aai520_research.prompts import (
    ANALYZE_EARNINGS_SYSTEM,
    ANALYZE_MARKET_SYSTEM,
    ANALYZE_NEWS_SYSTEM,
)
from aai520_research.specialists import earnings_view, market_view, news_view


class NewsSentimentAgent:
    """Aggregates headlines and the tone of the sample."""

    name = "News Sentiment Specialist"
    persona = (
        "Reads headlines only. Runs ingest, preprocess, classify, extract, "
        "and summarize, then writes the sentiment note."
    )

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.chain = NewsPromptChain(llm)

    def review_headlines(self, articles: list[dict[str, Any]]) -> ChainResult:
        """First handoff: turn raw headlines into classified facts and a brief."""
        return self.chain.run(articles)

    def analyze(self, assignment: dict[str, Any]) -> dict[str, Any]:
        """Second handoff: write the sentiment note from the routed headlines."""
        return _ask(self, ANALYZE_NEWS_SYSTEM, assignment, news_view)


class EarningsFundamentalAgent:
    """Reads financial statements, ratios, and the analyst snapshot."""

    name = "Earnings Fundamental Specialist"
    persona = (
        "Reads valuation, growth, margins, the income statement, and earnings "
        "headlines. Leaves the price chart to the market specialist."
    )

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def analyze(self, assignment: dict[str, Any]) -> dict[str, Any]:
        return _ask(self, ANALYZE_EARNINGS_SYSTEM, assignment, earnings_view)


class MarketTechnicalAgent:
    """Reads trend, moving averages, and RSI."""

    name = "Market Technical Specialist"
    persona = (
        "Reads return, volatility, drawdown, the 20-day and 50-day simple "
        "moving averages, and RSI(14). Leaves valuation to the earnings specialist."
    )

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def analyze(self, assignment: dict[str, Any]) -> dict[str, Any]:
        return _ask(self, ANALYZE_MARKET_SYSTEM, assignment, market_view)


def _ask(agent: Any, system: str, assignment: dict[str, Any], fallback: Any) -> dict[str, Any]:
    raw = agent.llm.complete_json(system, json.dumps(assignment, default=str))
    findings = _strings(raw.get("findings") if isinstance(raw, dict) else None)
    risks = _strings(raw.get("risks") if isinstance(raw, dict) else None)
    if not findings or not risks:
        report = fallback(assignment)
    else:
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence"))))
        except (TypeError, ValueError):
            confidence = 0.5
        report = {"findings": findings[:6], "risks": risks[:4], "confidence": confidence}
    report["agent"] = agent.name
    report["persona"] = agent.persona
    return report


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
