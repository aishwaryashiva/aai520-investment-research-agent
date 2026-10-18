"""Text renderers used by the command line and the project notebook."""

from __future__ import annotations

from typing import Any


def render_report(report: dict[str, Any]) -> str:
    """Render one brief, including the revision the optimizer has reached."""
    prior = report.get("prior_thesis_check") or "No prior thesis was on file for this pass."
    notes = report.get("revision_notes") or []
    note_block = _bullets(notes) if notes else "- Initial draft."
    return "\n".join(
        [
            f"# {report.get('symbol')} - Investment Research Brief",
            f"**Company:** {report.get('company_name')}",
            f"**Stance:** {report.get('stance')}",
            f"**Revision:** {report.get('revision')}",
            "",
            "## Thesis",
            str(report.get("thesis") or ""),
            "",
            "## Market",
            str(report.get("market_section") or ""),
            "",
            "## Fundamentals",
            str(report.get("fundamentals_section") or ""),
            "",
            "## News",
            str(report.get("news_section") or ""),
            "",
            "## Bull case",
            _bullets(report.get("bull_case")),
            "",
            "## Bear case",
            _bullets(report.get("bear_case")),
            "",
            "## Key figures",
            _bullets(report.get("key_figures")),
            "",
            "## What would change this view",
            _bullets(report.get("catalysts")),
            "",
            "## Prior thesis check",
            str(prior),
            "",
            "## Revision notes",
            note_block,
            "",
            "*Educational research brief for AAI-520. Not investment advice.*",
        ]
    )


def format_steps(steps: list[dict[str, Any]]) -> str:
    """Render a plan, or the follow-up steps added by self-reflection."""
    if not steps:
        return "No steps."
    lines = []
    for step in steps:
        lines.append(f"{step.get('id')}. {step.get('tool')}")
        lines.append(f"   Goal: {step.get('goal')}")
        lines.append(f"   Why: {step.get('reason')}")
        lines.append(f"   Arguments: {step.get('arguments')}")
    return "\n".join(lines)


def format_routes(routes: list[dict[str, Any]]) -> str:
    """Render the routing table."""
    if not routes:
        return "No content was routed."
    lines = []
    for route in routes:
        lines.append(
            f"{route.get('content_id')} [{route.get('kind')}] -> {route.get('route')}: "
            f"{route.get('reason')}"
        )
    return "\n".join(lines)


def _bullets(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return "- None recorded."
    return "\n".join(f"- {item}" for item in items)
