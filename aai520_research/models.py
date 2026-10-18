"""Datatypes shared by the research agent and its workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clean(value: Any) -> Any:
    """Convert dataclasses and tuples into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


@dataclass
class ToolResult:
    """One tool call and the payload it returned."""

    tool: str
    arguments: dict[str, Any]
    ok: bool
    data: dict[str, Any]
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PlanStep:
    """A single planned tool call."""

    id: int
    goal: str
    tool: str
    arguments: dict[str, Any]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class ChainResult:
    """Intermediate artifacts from the news prompt chain."""

    ingested: list[dict[str, Any]]
    preprocessed: list[dict[str, Any]]
    classified: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    brief: dict[str, Any]
    dropped: int

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class RoutingDecision:
    """Where one piece of content was sent, and why."""

    content_id: str
    kind: str
    route: str
    reason: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class SpecialistReport:
    """Findings from one specialist."""

    route: str
    findings: list[str]
    risks: list[str]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class IterationRecord:
    """One generate-evaluate cycle, including the draft that was scored."""

    revision: int
    report_markdown: str
    scores: dict[str, float]
    overall: float
    passed: bool
    feedback: list[str]

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class AgentRun:
    """Everything a notebook needs to show how one research run worked."""

    symbol: str
    llm_name: str
    memory_context: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    gap_steps: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    chain: dict[str, Any]
    routes: list[dict[str, Any]]
    delegations: list[dict[str, Any]]
    specialists: dict[str, dict[str, Any]]
    iterations: list[dict[str, Any]]
    final_markdown: str
    reflection: dict[str, Any]
    memory_note: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
