"""Research planner.

The planner proposes tool calls. ``ensure_required_steps`` then guarantees
the core tools and applies next-run instructions saved by an earlier
reflection, which is how the agent learns across runs.
"""

from __future__ import annotations

import json
from typing import Any

from aai520_research.prompts import PLAN_SYSTEM

CORE_TOOLS = (
    "get_price_history",
    "get_fundamentals",
    "get_company_news",
    "get_analyst_snapshot",
)
_GOALS = {
    "get_price_history": "Measure the recent price trend, volatility, and drawdown.",
    "get_fundamentals": "Collect valuation, growth, profitability, and profile fields.",
    "get_company_news": "Ingest recent headlines for the prompt-chaining workflow.",
    "get_analyst_snapshot": "Collect the consensus target and recommendation.",
    "get_income_statement": "Compare revenue and net margin across reported periods.",
}
_REASONS = {
    "get_price_history": "A stance needs the recent return, volatility, and drawdown.",
    "get_fundamentals": "The earnings specialist needs current reported fundamentals.",
    "get_company_news": "Headlines feed classification, extraction, and the news brief.",
    "get_analyst_snapshot": "The published target gives the brief an external benchmark.",
    "get_income_statement": "A prior reflection asked to verify the multi-year margin trend.",
}


class ResearchPlanner:
    """Create an ordered tool plan for one symbol."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def create_plan(
        self,
        symbol: str,
        memory_notes: list[dict[str, Any]],
        known_tools: set[str],
    ) -> list[dict[str, Any]]:
        payload = {
            "symbol": symbol.upper(),
            "memory": [_public_note(note) for note in memory_notes],
            "tools": sorted(known_tools),
        }
        raw = self.llm.complete_json(PLAN_SYSTEM, json.dumps(payload))
        proposed = []
        for item in raw.get("steps") or []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "")
            if tool not in known_tools or tool == "retrieve_memory":
                continue
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            proposed.append(
                {
                    "id": 0,
                    "goal": str(item.get("goal") or _GOALS.get(tool) or tool),
                    "tool": tool,
                    "arguments": dict(arguments),
                    "reason": str(
                        item.get("reason")
                        or _REASONS.get(tool)
                        or "Requested by the planner."
                    ),
                }
            )
        steps = ensure_required_steps(proposed, symbol, memory_notes, known_tools)
        for index, step in enumerate(steps, start=1):
            step["id"] = index
        return steps


def fallback_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Local planner used when no hosted model is configured."""
    symbol = str(payload.get("symbol") or "").upper()
    notes = payload.get("memory") or []
    latest = notes[-1] if notes else {}
    next_run = latest.get("next_run") if isinstance(latest, dict) else {}
    if not isinstance(next_run, dict):
        next_run = {}
    news_count = int(next_run.get("news_count") or 8)
    steps = [
        {
            "goal": _GOALS["get_price_history"],
            "tool": "get_price_history",
            "arguments": {"symbol": symbol, "period": "1y"},
            "reason": _REASONS["get_price_history"],
        },
        {
            "goal": _GOALS["get_fundamentals"],
            "tool": "get_fundamentals",
            "arguments": {"symbol": symbol},
            "reason": _REASONS["get_fundamentals"],
        },
        {
            "goal": _GOALS["get_company_news"],
            "tool": "get_company_news",
            "arguments": {"symbol": symbol, "limit": news_count},
            "reason": _REASONS["get_company_news"],
        },
        {
            "goal": _GOALS["get_analyst_snapshot"],
            "tool": "get_analyst_snapshot",
            "arguments": {"symbol": symbol},
            "reason": _REASONS["get_analyst_snapshot"],
        },
    ]
    if next_run.get("pull_income_statement"):
        steps.append(
            {
                "goal": _GOALS["get_income_statement"],
                "tool": "get_income_statement",
                "arguments": {"symbol": symbol},
                "reason": _REASONS["get_income_statement"],
            }
        )
    return {"steps": steps}


def ensure_required_steps(
    steps: list[dict[str, Any]],
    symbol: str,
    memory_notes: list[dict[str, Any]],
    known_tools: set[str],
) -> list[dict[str, Any]]:
    """Insert missing core tools and apply memory from the prior reflection."""
    symbol = symbol.upper()
    next_run = _latest_next_run(memory_notes)
    news_count = _news_count(next_run)
    by_tool = {step["tool"]: step for step in steps}
    ordered: list[dict[str, Any]] = []
    for tool in CORE_TOOLS:
        if tool not in known_tools:
            continue
        current = by_tool.get(tool)
        arguments = {"symbol": symbol}
        if tool == "get_price_history":
            period = "1y"
            if current and isinstance(current["arguments"].get("period"), str):
                period = current["arguments"]["period"] or "1y"
            arguments["period"] = period
        if tool == "get_company_news":
            limit = news_count if "news_count" in next_run else _existing_limit(current)
            arguments["limit"] = limit
        reason = current["reason"] if current else _REASONS[tool]
        if tool == "get_company_news" and "news_count" in next_run:
            reason = f"A prior reflection set the news sample size to {news_count}."
        goal = current["goal"] if current else _GOALS[tool]
        ordered.append(
            {
                "id": 0,
                "goal": goal,
                "tool": tool,
                "arguments": arguments,
                "reason": reason,
            }
        )
    wants_income = bool(next_run.get("pull_income_statement")) or (
        "get_income_statement" in by_tool
    )
    if wants_income and "get_income_statement" in known_tools:
        current = by_tool.get("get_income_statement")
        reason = _REASONS["get_income_statement"] if next_run.get("pull_income_statement") else (
            current["reason"] if current else _REASONS["get_income_statement"]
        )
        ordered.append(
            {
                "id": 0,
                "goal": _GOALS["get_income_statement"],
                "tool": "get_income_statement",
                "arguments": {"symbol": symbol},
                "reason": reason,
            }
        )
    seen = {step["tool"] for step in ordered}
    for step in steps:
        if step["tool"] in seen:
            continue
        arguments = dict(step["arguments"])
        arguments["symbol"] = symbol
        ordered.append({**step, "arguments": arguments})
        seen.add(step["tool"])
    if ordered:
        ordered[0]["reason"] = _with_focus(ordered[0]["reason"], next_run.get("focus") or [])
    return ordered


def coverage_gap_steps(
    symbol: str,
    results: list[dict[str, Any]],
    start_id: int,
) -> list[dict[str, Any]]:
    """Add a follow-up tool call when a required observation came back empty."""
    symbol = symbol.upper()
    extra: list[dict[str, Any]] = []
    next_id = start_id
    news_calls = [item for item in results if item.get("tool") == "get_company_news"]
    news_ok = any(item.get("ok") and (item.get("data") or {}).get("items") for item in news_calls)
    if news_calls and not news_ok and len(news_calls) < 2:
        extra.append(
            {
                "id": next_id,
                "goal": "Retry headlines after an empty news sample.",
                "tool": "get_company_news",
                "arguments": {"symbol": symbol, "limit": 20},
                "reason": (
                    "Self-reflection: the news tool returned no articles, so the "
                    "agent requests a larger sample before it writes the brief."
                ),
            }
        )
        next_id += 1
    price_calls = [item for item in results if item.get("tool") == "get_price_history"]
    price_ok = any(
        item.get("ok") and (item.get("data") or {}).get("last_price") is not None
        for item in price_calls
    )
    if not price_ok and len(price_calls) < 2:
        extra.append(
            {
                "id": next_id,
                "goal": "Retry price history on a shorter window.",
                "tool": "get_price_history",
                "arguments": {"symbol": symbol, "period": "6mo"},
                "reason": (
                    "Self-reflection: price history was missing, so the agent "
                    "retries a shorter window before it abandons the run."
                ),
            }
        )
    return extra


def _public_note(note: dict[str, Any]) -> dict[str, Any]:
    thesis = str(note.get("thesis") or "")
    return {
        "timestamp": note.get("timestamp"),
        "stance": note.get("stance"),
        "thesis": thesis[:400],
        "lessons": note.get("lessons") or [],
        "gaps": note.get("gaps") or [],
        "next_run": note.get("next_run") or {},
        "overall_score": note.get("overall_score"),
    }


def _latest_next_run(notes: list[dict[str, Any]]) -> dict[str, Any]:
    if not notes:
        return {}
    latest = notes[-1].get("next_run") or {}
    return latest if isinstance(latest, dict) else {}


def _news_count(next_run: dict[str, Any]) -> int:
    try:
        count = int(next_run.get("news_count") or 8)
    except (TypeError, ValueError):
        count = 8
    return min(max(count, 1), 20)


def _with_focus(reason: str, focus: Any) -> str:
    """Attach the prior run's focus once, without doubling the final period."""
    base = str(reason or "").strip()
    if "Memory focus:" in base:
        return base if base.endswith(".") else base + "."
    cleaned = [str(item).strip().rstrip(".") for item in focus or [] if str(item).strip()]
    if not base.endswith("."):
        base += "."
    if not cleaned:
        return base
    return base + " Memory focus: " + "; ".join(cleaned) + "."


def _existing_limit(step: dict[str, Any] | None) -> int:
    if not step:
        return 8
    try:
        return min(max(int(step["arguments"].get("limit") or 8), 1), 20)
    except (TypeError, ValueError):
        return 8
