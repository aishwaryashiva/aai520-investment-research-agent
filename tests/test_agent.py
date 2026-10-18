"""End-to-end tests with local tools so the loop does not call Yahoo."""

from aai520_research.agent import InvestmentResearchAgent, ResearchError
from aai520_research.heuristic import HeuristicLLM
from aai520_research.memory import MemoryStore
from aai520_research.tools import ToolRegistry


def _prices(symbol: str, period: str = "1y") -> dict:
    return {
        "symbol": symbol,
        "period": period,
        "currency": "USD",
        "last_price": 100.0,
        "period_return": 0.12,
        "annualized_volatility": 0.22,
        "max_drawdown": -0.08,
        "average_volume": 1_000_000,
        "high": 110.0,
        "low": 80.0,
        "observations": 250,
        "monthly_closes": [{"date": "2026-01-31", "close": 100.0}],
    }


def _fundamentals(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "short_name": "Example Corp",
        "sector": "Technology",
        "industry": "Software",
        "market_cap": 200_000_000_000,
        "trailing_pe": 20.0,
        "forward_pe": 18.0,
        "profit_margin": 0.25,
        "revenue_growth": 0.1,
        "debt_to_equity": 30.0,
        "return_on_equity": 0.3,
        "free_cash_flow": 5_000_000_000,
        "total_revenue": 40_000_000_000,
        "trailing_eps": 5.0,
        "currency": "USD",
    }


def _news(symbol: str, limit: int = 8) -> dict:
    return {
        "symbol": symbol,
        "source": "test",
        "items": [
            {
                "title": "Example Corp beats earnings expectations as cloud growth accelerates",
                "publisher": "Test Wire",
                "link": "https://example.test/a",
                "summary": "EPS beat the estimate.",
                "published": "2026-01-15",
            },
            {
                "title": "Analysts raise the Example price target after the quarter",
                "publisher": "Test Wire",
                "link": "https://example.test/b",
                "summary": "",
                "published": "2026-01-20",
            },
        ][:limit],
    }


def _analyst(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "target_mean": 120.0,
        "target_high": 140.0,
        "target_low": 90.0,
        "recommendation": "buy",
        "recommendation_mean": 1.8,
        "opinions": 12,
        "currency": "USD",
    }


def _income(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "available": True,
        "periods": ["2024-06-30", "2025-06-30"],
        "revenue": [100.0, 110.0],
        "net_income": [20.0, 24.0],
        "net_margin": [0.2, 0.218],
        "revenue_change": 0.1,
        "net_margin_first": 0.2,
        "net_margin_last": 0.218,
    }


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("get_price_history", _prices, "prices")
    registry.register("get_fundamentals", _fundamentals, "fundamentals")
    registry.register("get_company_news", _news, "news")
    registry.register("get_analyst_snapshot", _analyst, "analyst")
    registry.register("get_income_statement", _income, "income")
    return registry


def _agent(tmp_path, registry=None, llm=None) -> InvestmentResearchAgent:
    return InvestmentResearchAgent(
        llm=llm or HeuristicLLM(),
        memory=MemoryStore(tmp_path / "memory.json"),
        registry=registry or _registry(),
        max_revisions=2,
    )


def test_run_revises_until_the_rubric_passes_and_writes_memory(tmp_path):
    agent = _agent(tmp_path)
    result = agent.run("msft")
    assert result.symbol == "MSFT"
    assert result.llm_name == "heuristic"
    assert [step["tool"] for step in result.plan] == [
        "get_price_history",
        "get_fundamentals",
        "get_company_news",
        "get_analyst_snapshot",
    ]
    assert result.gap_steps == []
    assert result.iterations[0]["passed"] is False
    assert result.iterations[-1]["passed"] is True
    assert result.iterations[-1]["overall"] >= 4
    assert "get_income_statement" not in [step["tool"] for step in result.plan]
    assert result.memory_note["next_run"]["pull_income_statement"] is True
    assert result.chain["brief"]["item_count"] == 2
    assert any(route["route"] == "earnings" for route in result.routes)
    names = {report["agent"] for report in result.specialists.values()}
    assert names == {
        "News Sentiment Specialist",
        "Earnings Fundamental Specialist",
        "Market Technical Specialist",
    }
    handed_to = {item["to"] for item in result.delegations}
    assert names <= handed_to

    second = agent.run("MSFT")
    assert "get_income_statement" in [step["tool"] for step in second.plan]
    assert second.memory_context[0]["stance"] == result.memory_note["stance"]
    assert second.evidence["had_income_statement"] is True
    assert second.evidence["had_prior_thesis"] is True
    brief = second.final_markdown.lower()
    assert "prior thesis" in brief or "prior stance" in brief
    assert len(agent.memory.recall("MSFT")) == 2


def test_empty_news_triggers_a_second_tool_call(tmp_path):
    calls = {"news": 0}

    def news(symbol: str, limit: int = 8) -> dict:
        calls["news"] += 1
        if calls["news"] == 1:
            return {"symbol": symbol, "source": "test", "items": []}
        return _news(symbol, limit)

    registry = _registry()
    registry.register("get_company_news", news, "news")
    result = _agent(tmp_path, registry).run("MSFT")
    assert calls["news"] == 2
    assert result.gap_steps[0]["tool"] == "get_company_news"
    assert result.chain["brief"]["item_count"] == 2


def test_missing_prices_raise_a_clear_error(tmp_path):
    def prices(symbol: str, period: str = "1y") -> dict:
        raise RuntimeError(f"no prices for {symbol} {period}")

    registry = _registry()
    registry.register("get_price_history", prices, "prices")
    try:
        _agent(tmp_path, registry).run("MSFT")
    except ResearchError as exc:
        assert "MSFT" in str(exc)
    else:
        raise AssertionError("Expected ResearchError")
