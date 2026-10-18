"""Tests for plan repair and same-run tool follow-ups."""

import json

from aai520_research.heuristic import HeuristicLLM
from aai520_research.llm import task_id
from aai520_research.planner import ResearchPlanner, coverage_gap_steps


class _EmptyPlan(HeuristicLLM):
    def complete_json(self, system: str, user: str) -> dict:
        if task_id(system) == "plan_research":
            json.loads(user)
            return {"steps": []}
        return super().complete_json(system, user)


def test_empty_model_plan_still_covers_core_tools():
    planner = ResearchPlanner(_EmptyPlan())
    tools = {
        "get_price_history",
        "get_fundamentals",
        "get_company_news",
        "get_analyst_snapshot",
        "get_income_statement",
        "retrieve_memory",
    }
    steps = planner.create_plan("msft", [], tools)
    assert [step["tool"] for step in steps] == [
        "get_price_history",
        "get_fundamentals",
        "get_company_news",
        "get_analyst_snapshot",
    ]


def test_memory_adds_the_income_statement_and_focus():
    planner = ResearchPlanner(HeuristicLLM())
    notes = [
        {
            "stance": "Balanced",
            "thesis": "Prior thesis.",
            "next_run": {
                "pull_income_statement": True,
                "news_count": 12,
                "focus": ["Challenge the prior Balanced stance."],
            },
        }
    ]
    tools = {
        "get_price_history",
        "get_fundamentals",
        "get_company_news",
        "get_analyst_snapshot",
        "get_income_statement",
    }
    steps = planner.create_plan("MSFT", notes, tools)
    by_tool = {step["tool"]: step for step in steps}
    assert "get_income_statement" in by_tool
    assert by_tool["get_company_news"]["arguments"]["limit"] == 12
    assert "Memory focus" in steps[0]["reason"]


def test_empty_news_requests_a_larger_sample():
    results = [
        {
            "tool": "get_company_news",
            "ok": True,
            "data": {"items": []},
            "arguments": {"symbol": "MSFT", "limit": 8},
            "error": None,
        },
        {
            "tool": "get_price_history",
            "ok": True,
            "data": {"last_price": 100},
            "arguments": {},
            "error": None,
        },
    ]
    extra = coverage_gap_steps("MSFT", results, start_id=5)
    assert [step["tool"] for step in extra] == ["get_company_news"]
    assert extra[0]["arguments"]["limit"] == 20
    assert extra[0]["id"] == 5
