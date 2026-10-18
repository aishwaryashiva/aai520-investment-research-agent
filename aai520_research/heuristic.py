"""Deterministic provider that implements every agent task locally.

The hosted model and this provider share prompts and JSON contracts. This
provider keeps the notebook runnable for review when no API key is set.
"""

from __future__ import annotations

import json
from typing import Any

from aai520_research.evaluator import apply_rubric, refine_report, reflect_on_run
from aai520_research.llm import LLMError, task_id
from aai520_research.planner import fallback_plan
from aai520_research.prompt_chain import (
    classify_with_rules,
    extract_with_rules,
    summarize_facts,
)
from aai520_research.router import route_with_rules
from aai520_research.specialists import (
    compose_draft,
    earnings_view,
    market_view,
    news_view,
)


class HeuristicLLM:
    """Local task handlers selected by the prompt's task tag."""

    name = "heuristic"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        task = task_id(system)
        try:
            payload = json.loads(user) if user else {}
        except json.JSONDecodeError as exc:
            raise LLMError("Heuristic provider expected a JSON user payload.") from exc
        if not isinstance(payload, dict):
            raise LLMError("Heuristic provider expected a JSON object.")
        handler = _HANDLERS.get(task)
        if handler is None:
            raise LLMError(f"No heuristic handler for {task}.")
        result = handler(payload)
        if not isinstance(result, dict):
            raise LLMError(f"Handler {task} did not return an object.")
        return result


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    return apply_rubric(payload.get("report") or {}, payload.get("evidence") or {})


def _refine(payload: dict[str, Any]) -> dict[str, Any]:
    return refine_report(
        payload.get("report") or {},
        payload.get("evidence") or {},
        list(payload.get("feedback") or []),
    )


_HANDLERS = {
    "plan_research": fallback_plan,
    "classify_news": classify_with_rules,
    "extract_facts": extract_with_rules,
    "summarize_news": summarize_facts,
    "route_content": route_with_rules,
    "analyze_earnings": earnings_view,
    "analyze_market": market_view,
    "analyze_news": news_view,
    "draft_report": compose_draft,
    "evaluate_report": _evaluate,
    "refine_report": _refine,
    "reflect": reflect_on_run,
}
