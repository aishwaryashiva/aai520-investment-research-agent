"""Research orchestrator.

Plans the work, calls tools, and delegates packets to the news, earnings,
and market agents. It then drafts one brief from their notes, revises it,
and stores what the next run should do differently.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from aai520_research.agents import (
    EarningsFundamentalAgent,
    MarketTechnicalAgent,
    NewsSentimentAgent,
)
from aai520_research.evaluator import (
    apply_rubric,
    coerce_report,
    coerce_scores,
    reflect_on_run,
    refine_report,
)
from aai520_research.formatting import fmt_big, fmt_number, fmt_pct, fmt_price
from aai520_research.llm import LLMError, build_llm
from aai520_research.memory import MemoryStore
from aai520_research.models import AgentRun, IterationRecord, ToolResult
from aai520_research.planner import ResearchPlanner, coverage_gap_steps
from aai520_research.prompts import DRAFT_SYSTEM, EVALUATE_SYSTEM, REFINE_SYSTEM
from aai520_research.reporting import render_report
from aai520_research.router import ContentRouter
from aai520_research.tools import build_market_registry


class ResearchError(RuntimeError):
    """Raised when a run cannot gather the price evidence it needs."""


class InvestmentResearchAgent:
    """Orchestrator for one ticker. The specialists do the reading."""

    def __init__(
        self,
        llm: Any | None = None,
        memory: MemoryStore | None = None,
        registry: Any | None = None,
        max_revisions: int = 2,
        score_threshold: float = 4.0,
    ) -> None:
        self.llm = llm or build_llm()
        self.memory = memory or MemoryStore.default()
        self.registry = registry or build_market_registry()
        self.registry.register(
            "retrieve_memory",
            self._retrieve_memory,
            "Recall prior research notes and next-run instructions for a symbol.",
        )
        self.planner = ResearchPlanner(self.llm)
        self.news_agent = NewsSentimentAgent(self.llm)
        self.earnings_agent = EarningsFundamentalAgent(self.llm)
        self.market_agent = MarketTechnicalAgent(self.llm)
        self.router = ContentRouter(self.llm)
        self.max_revisions = max_revisions
        self.score_threshold = score_threshold

    def run(self, symbol: str) -> AgentRun:
        """Research one ticker and persist what the next run should learn."""
        symbol = symbol.upper().strip()
        if not symbol:
            raise ResearchError("A stock symbol is required.")

        retrieval = self.registry.call("retrieve_memory", {"symbol": symbol, "limit": 3})
        notes = list(retrieval.data.get("notes") or []) if retrieval.ok else []
        plan = self.planner.create_plan(symbol, notes, self.registry.names())
        results = [retrieval]
        for step in plan:
            results.append(self.registry.call(step["tool"], step["arguments"]))

        gap_steps = coverage_gap_steps(
            symbol,
            [item.as_dict() for item in results],
            start_id=len(plan) + 1,
        )
        for step in gap_steps:
            results.append(self.registry.call(step["tool"], step["arguments"]))
        self._require_prices(symbol, results)

        articles = _news_items(results)
        chain = self.news_agent.review_headlines(articles)
        delegations = [
            {
                "from": "Research Orchestrator",
                "to": self.news_agent.name,
                "task": "Run the headline chain and prepare a sentiment brief.",
                "detail": f"{chain.brief.get('item_count', 0)} headlines kept",
            }
        ]
        contents = _content_objects(symbol, results, chain)
        routes = [self.router.route(content) for content in contents]
        routed_headlines = _headlines_by_route(routes)
        market_packet = _market_payload(
            symbol, results, routed_headlines.get("market") or []
        )
        earnings_packet = _earnings_payload(
            symbol, results, routed_headlines.get("earnings") or []
        )
        news_packet = {
            "symbol": symbol,
            "brief": chain.brief,
            "headlines": routed_headlines.get("news") or [],
        }
        specialists = {
            "market": self.market_agent.analyze(market_packet),
            "earnings": self.earnings_agent.analyze(earnings_packet),
            "news": self.news_agent.analyze(news_packet),
        }
        delegations.extend(
            [
                {
                    "from": "Research Orchestrator",
                    "to": self.market_agent.name,
                    "task": "Read trend, 20-day and 50-day moving averages, and RSI.",
                    "detail": (
                        f"RSI(14) {market_packet.get('rsi_14')}, "
                        f"SMA20 {market_packet.get('sma_20')}"
                    ),
                },
                {
                    "from": "Research Orchestrator",
                    "to": self.earnings_agent.name,
                    "task": "Read ratios, the income statement, and earnings headlines.",
                    "detail": f"{len(earnings_packet.get('headlines') or [])} routed headlines",
                },
                {
                    "from": "Research Orchestrator",
                    "to": self.news_agent.name,
                    "task": "Write the sentiment note from the routed headlines.",
                    "detail": f"{len(news_packet.get('headlines') or [])} routed headlines",
                },
            ]
        )
        evidence = _evidence(symbol, notes, results, chain.brief, specialists)
        report = self._draft(evidence)
        iterations: list[IterationRecord] = []
        best_report = copy.deepcopy(report)
        best_eval: dict[str, Any] | None = None

        for index in range(self.max_revisions + 1):
            evaluation = apply_rubric(report, evidence, self.score_threshold)
            evaluation = self._append_model_critique(report, evidence, evaluation)
            iterations.append(
                IterationRecord(
                    revision=int(report.get("revision") or 0),
                    report_markdown=render_report(report),
                    scores=dict(evaluation["scores"]),
                    overall=float(evaluation["overall"]),
                    passed=bool(evaluation["passed"]),
                    feedback=list(evaluation["feedback"]),
                )
            )
            if best_eval is None or evaluation["overall"] >= best_eval["overall"]:
                best_eval = evaluation
                best_report = copy.deepcopy(report)
            if evaluation["passed"] or index == self.max_revisions:
                break
            report = self._refine(report, evaluation, evidence)

        assert best_eval is not None
        reflection = reflect_on_run(
            {
                "symbol": symbol,
                "stance": best_report.get("stance"),
                "thesis": best_report.get("thesis"),
                "news_count": evidence.get("news_count") or 0,
                "had_income_statement": bool(evidence.get("had_income_statement")),
                "feedback": best_eval.get("feedback") or [],
                "overall": best_eval.get("overall"),
            }
        )
        note = self.memory.remember(
            {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "company_name": best_report.get("company_name"),
                "stance": best_report.get("stance"),
                "thesis": best_report.get("thesis"),
                "overall_score": best_eval.get("overall"),
                "lessons": reflection["lessons"],
                "gaps": reflection["gaps"],
                "next_run": reflection["next_run"],
                "llm": self.llm.name,
            }
        )
        final = render_report(best_report)
        final += (
            "\n\n---\n"
            f"Evaluator score: {best_eval['overall']:.2f} "
            f"after {len(iterations)} draft(s). Model: {self.llm.name}."
        )
        return AgentRun(
            symbol=symbol,
            llm_name=self.llm.name,
            memory_context=notes,
            plan=plan,
            gap_steps=gap_steps,
            tool_results=[item.as_dict() for item in results],
            chain=chain.as_dict(),
            routes=[item.as_dict() for item in routes],
            delegations=delegations,
            specialists=specialists,
            iterations=[item.as_dict() for item in iterations],
            final_markdown=final,
            reflection=reflection,
            memory_note=note,
            evidence=evidence,
        )

    def _retrieve_memory(self, symbol: str, limit: int = 3) -> dict[str, Any]:
        notes = self.memory.recall(symbol, limit=limit)
        return {"symbol": symbol.upper(), "notes": notes, "count": len(notes)}

    def _draft(self, evidence: dict[str, Any]) -> dict[str, Any]:
        raw = self.llm.complete_json(DRAFT_SYSTEM, _dump(evidence))
        return coerce_report(raw, str(evidence.get("symbol") or ""))

    def _refine(
        self,
        report: dict[str, Any],
        evaluation: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        raw = self.llm.complete_json(
            REFINE_SYSTEM,
            _dump(
                {
                    "report": report,
                    "feedback": evaluation.get("feedback") or [],
                    "evidence": evidence,
                }
            ),
        )
        revised = coerce_report(raw, str(evidence.get("symbol") or report.get("symbol") or ""))
        revised_score = apply_rubric(revised, evidence, self.score_threshold)["overall"]
        if revised_score < evaluation["overall"]:
            revised = refine_report(report, evidence, list(evaluation.get("feedback") or []))
            notes = list(revised.get("revision_notes") or [])
            notes.append(
                "The model revision did not improve the rubric, so the "
                "deterministic optimizer was applied."
            )
            revised["revision_notes"] = notes
        return revised

    def _append_model_critique(
        self,
        report: dict[str, Any],
        evidence: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        if not str(self.llm.name).startswith("openai"):
            return evaluation
        try:
            raw = self.llm.complete_json(
                EVALUATE_SYSTEM,
                _dump({"report": report, "evidence": evidence}),
            )
        except LLMError:
            return evaluation
        extra = coerce_scores(raw, self.score_threshold)
        feedback = list(evaluation["feedback"])
        for item in extra["feedback"]:
            if item not in feedback:
                feedback.append(item)
        updated = dict(evaluation)
        updated["feedback"] = feedback
        return updated

    def _require_prices(self, symbol: str, results: list[ToolResult]) -> None:
        price = _latest(results, "get_price_history")
        if price is None or price.data.get("last_price") is None:
            errors = [
                item.error
                for item in results
                if item.tool == "get_price_history" and item.error
            ]
            detail = errors[-1] if errors else "no price tool result"
            raise ResearchError(f"Could not load prices for {symbol}: {detail}")


def _evidence(
    symbol: str,
    notes: list[dict[str, Any]],
    results: list[ToolResult],
    brief: dict[str, Any],
    specialists: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prices = (_latest(results, "get_price_history") or _empty()).data
    fundamentals = (_latest(results, "get_fundamentals") or _empty()).data
    analyst = (_latest(results, "get_analyst_snapshot") or _empty()).data
    income_result = _latest(results, "get_income_statement")
    income = income_result.data if income_result is not None else None
    currency = str(prices.get("currency") or fundamentals.get("currency") or "USD")
    last_price = prices.get("last_price")
    target = analyst.get("target_mean")
    upside = None
    if isinstance(last_price, (int, float)) and isinstance(target, (int, float)) and last_price:
        upside = float(target) / float(last_price) - 1.0
    counts = brief.get("sentiment_counts") or {}
    net = int(counts.get("positive") or 0) - int(counts.get("negative") or 0)
    prior = notes[-1] if notes else None
    had_income = bool(isinstance(income, dict) and income.get("available"))
    evidence: dict[str, Any] = {
        "symbol": symbol,
        "company_name": fundamentals.get("short_name") or symbol,
        "currency": currency,
        "period": prices.get("period") or "1y",
        "last_price": last_price,
        "period_return": prices.get("period_return"),
        "annualized_volatility": prices.get("annualized_volatility"),
        "max_drawdown": prices.get("max_drawdown"),
        "average_volume": prices.get("average_volume"),
        "high": prices.get("high"),
        "low": prices.get("low"),
        "sma_20": prices.get("sma_20"),
        "sma_50": prices.get("sma_50"),
        "rsi_14": prices.get("rsi_14"),
        "observations": prices.get("observations"),
        "trailing_pe": fundamentals.get("trailing_pe"),
        "forward_pe": fundamentals.get("forward_pe"),
        "profit_margin": fundamentals.get("profit_margin"),
        "revenue_growth": fundamentals.get("revenue_growth"),
        "debt_to_equity": fundamentals.get("debt_to_equity"),
        "market_cap": fundamentals.get("market_cap"),
        "sector": fundamentals.get("sector"),
        "industry": fundamentals.get("industry"),
        "free_cash_flow": fundamentals.get("free_cash_flow"),
        "total_revenue": fundamentals.get("total_revenue"),
        "target_mean": target,
        "target_upside": upside,
        "recommendation": analyst.get("recommendation"),
        "analyst_opinions": analyst.get("opinions"),
        "news_count": int(brief.get("item_count") or 0),
        "news_net_sentiment": net,
        "sentiment_counts": counts,
        "news_summary": brief.get("summary"),
        "income_statement": income if had_income else None,
        "had_income_statement": had_income,
        "had_prior_thesis": bool(prior and prior.get("thesis")),
        "prior_note": prior,
        "specialists": specialists,
    }
    evidence["figure_bank"] = _figure_bank(evidence)
    return evidence


def _figure_bank(evidence: dict[str, Any]) -> list[str]:
    currency = str(evidence.get("currency") or "USD")
    period = str(evidence.get("period") or "1y")
    figures = []
    if isinstance(evidence.get("last_price"), (int, float)):
        figures.append(f"Last price {fmt_price(float(evidence['last_price']), currency)}")
    if isinstance(evidence.get("period_return"), (int, float)):
        figures.append(f"{period} total return {fmt_pct(float(evidence['period_return']))}")
    if isinstance(evidence.get("annualized_volatility"), (int, float)):
        volatility = fmt_pct(float(evidence["annualized_volatility"]))
        figures.append(f"Annualized volatility {volatility}")
    if isinstance(evidence.get("max_drawdown"), (int, float)):
        figures.append(f"Maximum drawdown {fmt_pct(float(evidence['max_drawdown']))}")
    if isinstance(evidence.get("sma_20"), (int, float)):
        figures.append(f"20-day SMA {fmt_price(float(evidence['sma_20']), currency)}")
    if isinstance(evidence.get("sma_50"), (int, float)):
        figures.append(f"50-day SMA {fmt_price(float(evidence['sma_50']), currency)}")
    if isinstance(evidence.get("rsi_14"), (int, float)):
        figures.append(f"RSI(14) {fmt_number(float(evidence['rsi_14']), 1)}")
    if isinstance(evidence.get("trailing_pe"), (int, float)):
        figures.append(f"Trailing P/E {fmt_number(float(evidence['trailing_pe']), 1)}x")
    if isinstance(evidence.get("profit_margin"), (int, float)):
        figures.append(f"Profit margin {fmt_pct(float(evidence['profit_margin']))}")
    if isinstance(evidence.get("revenue_growth"), (int, float)):
        figures.append(f"Revenue growth {fmt_pct(float(evidence['revenue_growth']))}")
    if isinstance(evidence.get("market_cap"), (int, float)):
        figures.append(f"Market cap {fmt_big(float(evidence['market_cap']))}")
    if isinstance(evidence.get("target_mean"), (int, float)):
        upside = evidence.get("target_upside")
        upside_text = fmt_pct(float(upside)) if isinstance(upside, (int, float)) else "n/a"
        figures.append(
            "Analyst mean target "
            f"{fmt_price(float(evidence['target_mean']), currency)} ({upside_text} vs last price)"
        )
    figures.append(
        f"News sample {int(evidence.get('news_count') or 0)} headlines, "
        f"net sentiment {int(evidence.get('news_net_sentiment') or 0)}"
    )
    return figures


def _market_payload(
    symbol: str,
    results: list[ToolResult],
    headlines: list[str],
) -> dict[str, Any]:
    prices = (_latest(results, "get_price_history") or _empty()).data
    return {
        "symbol": symbol,
        "currency": prices.get("currency") or "USD",
        "period": prices.get("period") or "1y",
        "last_price": prices.get("last_price"),
        "period_return": prices.get("period_return"),
        "annualized_volatility": prices.get("annualized_volatility"),
        "max_drawdown": prices.get("max_drawdown"),
        "average_volume": prices.get("average_volume"),
        "high": prices.get("high"),
        "low": prices.get("low"),
        "sma_20": prices.get("sma_20"),
        "sma_50": prices.get("sma_50"),
        "rsi_14": prices.get("rsi_14"),
        "headlines": headlines,
    }


def _earnings_payload(
    symbol: str,
    results: list[ToolResult],
    headlines: list[str],
) -> dict[str, Any]:
    fundamentals = (_latest(results, "get_fundamentals") or _empty()).data
    analyst = (_latest(results, "get_analyst_snapshot") or _empty()).data
    income_result = _latest(results, "get_income_statement")
    income = income_result.data if income_result is not None else None
    return {
        "symbol": symbol,
        "currency": fundamentals.get("currency") or "USD",
        "fundamentals": fundamentals,
        "analyst": analyst,
        "income_statement": income,
        "headlines": headlines,
    }


def _content_objects(symbol: str, results: list[ToolResult], chain: Any) -> list[dict[str, Any]]:
    contents = []
    prices = _latest(results, "get_price_history")
    if prices is not None:
        data = prices.data
        contents.append(
            {
                "id": "prices",
                "kind": "market_data",
                "text": (
                    f"{symbol} last price {data.get('last_price')} and "
                    f"{data.get('period')} return {data.get('period_return')}."
                ),
                "category": None,
                "sentiment": None,
            }
        )
    fundamentals = _latest(results, "get_fundamentals")
    if fundamentals is not None:
        data = fundamentals.data
        contents.append(
            {
                "id": "fundamentals",
                "kind": "fundamentals",
                "text": (
                    f"{data.get('short_name') or symbol} margin {data.get('profit_margin')} "
                    f"and revenue growth {data.get('revenue_growth')}."
                ),
                "category": None,
                "sentiment": None,
            }
        )
    analyst = _latest(results, "get_analyst_snapshot")
    if analyst is not None:
        contents.append(
            {
                "id": "analyst",
                "kind": "analyst",
                "text": (
                    f"Analyst target {analyst.data.get('target_mean')} and "
                    f"recommendation {analyst.data.get('recommendation')}."
                ),
                "category": None,
                "sentiment": None,
            }
        )
    income = _latest(results, "get_income_statement")
    if income is not None and income.data.get("available"):
        contents.append(
            {
                "id": "income-statement",
                "kind": "income_statement",
                "text": f"Income statement periods {income.data.get('periods')}.",
                "category": None,
                "sentiment": None,
            }
        )
    for item in chain.classified:
        publisher = f" ({item['publisher']})" if item.get("publisher") else ""
        contents.append(
            {
                "id": item["id"],
                "kind": "news",
                "text": f"{item['title']}{publisher}",
                "category": item.get("category"),
                "sentiment": item.get("sentiment"),
            }
        )
    return contents


def _headlines_by_route(routes: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {"earnings": [], "news": [], "market": []}
    for route in routes:
        if route.kind != "news":
            continue
        grouped.setdefault(route.route, []).append(route.text)
    return grouped


def _news_items(results: list[ToolResult]) -> list[dict[str, Any]]:
    calls = [item for item in results if item.tool == "get_company_news" and item.ok]
    for item in reversed(calls):
        articles = item.data.get("items") or []
        if articles:
            return list(articles)
    return []


def _latest(results: list[ToolResult], tool: str) -> ToolResult | None:
    found = [item for item in results if item.tool == tool and item.ok]
    return found[-1] if found else None


def _empty() -> ToolResult:
    return ToolResult(tool="", arguments={}, ok=False, data={}, error=None)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)
