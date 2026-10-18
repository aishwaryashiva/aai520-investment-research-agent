"""Routing workflow.

Structured observations have an owning specialist. Headlines are routed from
the category assigned by the prompt chain: earnings headlines go to the
earnings specialist, macro headlines go to the market specialist, and the
remaining headlines go to the news specialist.
"""

from __future__ import annotations

import json
from typing import Any

from aai520_research.models import RoutingDecision
from aai520_research.prompts import ROUTE_SYSTEM

ROUTES = ("earnings", "news", "market")
STRUCTURED_ROUTES = {
    "market_data": "market",
    "fundamentals": "earnings",
    "income_statement": "earnings",
    "analyst": "earnings",
}
_NEWS_ROUTES = {
    "earnings": "earnings",
    "macro": "market",
    "product": "news",
    "legal": "news",
    "analyst": "news",
    "other": "news",
}


class ContentRouter:
    """Send each content object to exactly one specialist."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def route(self, content: dict[str, Any]) -> RoutingDecision:
        raw = self.llm.complete_json(ROUTE_SYSTEM, json.dumps(_payload(content)))
        model_route = raw.get("route") if isinstance(raw, dict) else None
        model_reason = str(raw.get("reason") or "").strip() if isinstance(raw, dict) else ""
        route, reason = resolve_route(content, model_route, model_reason)
        return RoutingDecision(
            content_id=str(content.get("id") or ""),
            kind=str(content.get("kind") or ""),
            route=route,
            reason=reason,
            text=str(content.get("text") or "")[:240],
        )


def route_with_rules(payload: dict[str, Any]) -> dict[str, str]:
    """Route from kind and category without a hosted model."""
    route, reason = resolve_route(payload, None, "")
    return {"route": route, "reason": reason}


def resolve_route(
    content: dict[str, Any],
    model_route: Any,
    model_reason: str,
) -> tuple[str, str]:
    """Prefer the structured map, then a valid model choice, then the category map."""
    kind = str(content.get("kind") or "")
    if kind in STRUCTURED_ROUTES:
        route = STRUCTURED_ROUTES[kind]
        reason = f"Structured {kind} observations belong to the {route} specialist."
        if model_reason and model_route == route:
            reason = f"{reason} {model_reason}"
        return route, reason
    category = str(content.get("category") or "other")
    fallback = _NEWS_ROUTES.get(category, "news")
    if model_route in ROUTES:
        route = str(model_route)
        reason = model_reason or f"Routed to {route} from the content classifier."
        return route, reason
    return fallback, f"The {category} headline is assigned to the {fallback} specialist."


def _payload(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": content.get("id"),
        "kind": content.get("kind"),
        "text": str(content.get("text") or "")[:800],
        "category": content.get("category"),
        "sentiment": content.get("sentiment"),
    }
