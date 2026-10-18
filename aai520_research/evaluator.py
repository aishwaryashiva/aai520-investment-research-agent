"""Evaluator-optimizer rubric, revision, and cross-run reflection."""

from __future__ import annotations

from typing import Any

from aai520_research.formatting import fmt_big, fmt_number, fmt_pct, fmt_price

DIMENSIONS = (
    "evidence",
    "coverage",
    "balance",
    "actionability",
    "consistency",
)
STANCES = ("Constructive", "Balanced", "Cautious")


def choose_stance(evidence: dict[str, Any]) -> str:
    """Turn return, quality, news, and target upside into a stance label."""
    score = 0
    period_return = evidence.get("period_return")
    if isinstance(period_return, (int, float)):
        if period_return > 0.05:
            score += 1
        elif period_return < -0.05:
            score -= 1
    growth = evidence.get("revenue_growth")
    if isinstance(growth, (int, float)):
        if growth > 0.05:
            score += 1
        elif growth < 0:
            score -= 1
    margin = evidence.get("profit_margin")
    if isinstance(margin, (int, float)):
        if margin > 0.15:
            score += 1
        elif margin < 0:
            score -= 1
    net_sentiment = evidence.get("news_net_sentiment") or 0
    if isinstance(net_sentiment, (int, float)):
        if net_sentiment > 0:
            score += 1
        elif net_sentiment < 0:
            score -= 1
    upside = evidence.get("target_upside")
    if isinstance(upside, (int, float)):
        if upside > 0.05:
            score += 1
        elif upside < -0.02:
            score -= 1
    drawdown = evidence.get("max_drawdown")
    if isinstance(drawdown, (int, float)) and drawdown <= -0.20:
        score -= 1
    if score >= 2:
        return "Constructive"
    if score <= -2:
        return "Cautious"
    return "Balanced"


def apply_rubric(
    report: dict[str, Any],
    evidence: dict[str, Any],
    threshold: float = 4.0,
) -> dict[str, Any]:
    """Score a brief. The optimizer runs until this clears ``threshold``."""
    feedback: list[str] = []
    scores: dict[str, int] = {}

    figures = _strings(report.get("key_figures"))
    if len(figures) >= 4:
        scores["evidence"] = 5
    elif len(figures) >= 2:
        scores["evidence"] = 3
        feedback.append(
            "Evidence is thin. Cite at least four key figures from the tool "
            "outputs, including the period return and a valuation or target figure."
        )
    else:
        scores["evidence"] = 1
        feedback.append("Add a key-figures list grounded in the tool outputs.")

    sections = (
        report.get("thesis"),
        report.get("market_section"),
        report.get("fundamentals_section"),
        report.get("news_section"),
    )
    long_sections = sum(1 for section in sections if len(str(section or "").strip()) >= 80)
    if long_sections == 4:
        scores["coverage"] = 5
    elif long_sections >= 3:
        scores["coverage"] = 3
        feedback.append(
            "One of the thesis, market, fundamentals, or news sections is missing "
            "or too thin."
        )
    else:
        scores["coverage"] = 2
        feedback.append(
            "Write a substantive thesis, market, fundamentals, and news section."
        )

    bulls = _strings(report.get("bull_case"))
    bears = _strings(report.get("bear_case"))
    if len(bulls) >= 2 and len(bears) >= 2:
        scores["balance"] = 5
    elif bulls and bears:
        scores["balance"] = 3
        feedback.append(
            "The case is thin on one side. Give at least two bull points and two "
            "bear points."
        )
    else:
        scores["balance"] = 1
        feedback.append("Add both a bull case and a bear case.")

    catalysts = _strings(report.get("catalysts"))
    quantified = [item for item in catalysts if any(char.isdigit() for char in item)]
    stance = str(report.get("stance") or "")
    if stance in STANCES and len(quantified) >= 2:
        scores["actionability"] = 5
    elif stance and catalysts:
        scores["actionability"] = 3
        feedback.append(
            "Catalysts are too vague. Name at least two catalysts that include a "
            "number from the evidence."
        )
    else:
        scores["actionability"] = 2
        feedback.append("State a stance and at least two catalysts that would change it.")

    scores["consistency"] = 5
    net = evidence.get("news_net_sentiment")
    period_return = evidence.get("period_return")
    if (
        stance == "Constructive"
        and isinstance(net, (int, float))
        and net < 0
        and isinstance(period_return, (int, float))
        and period_return < 0
    ):
        scores["consistency"] = 2
        feedback.append(
            "The constructive stance conflicts with negative news and a negative "
            "period return. Soften the stance or explain the conflict with figures."
        )
    elif (
        stance == "Cautious"
        and isinstance(net, (int, float))
        and net > 0
        and isinstance(period_return, (int, float))
        and period_return > 0.1
    ):
        scores["consistency"] = 2
        feedback.append(
            "The cautious stance conflicts with positive news and a strong period "
            "return. Soften the stance or explain the conflict with figures."
        )
    prior_check = str(report.get("prior_thesis_check") or "").strip()
    if evidence.get("had_prior_thesis") and not prior_check:
        scores["consistency"] = min(scores["consistency"], 3)
        feedback.append(
            "Memory contains a prior thesis. Add a prior-thesis check that says "
            "whether this stance confirms it or changes it."
        )

    overall = round(sum(scores[name] for name in DIMENSIONS) / len(DIMENSIONS), 2)
    if overall < threshold and not feedback:
        feedback.append(f"Raise the overall rubric score to at least {threshold:.1f}.")
    return {
        "scores": scores,
        "overall": overall,
        "passed": overall >= threshold,
        "feedback": feedback,
    }


def refine_report(
    report: dict[str, Any],
    evidence: dict[str, Any],
    feedback: list[str],
) -> dict[str, Any]:
    """Apply rubric feedback without adding figures that are not in evidence."""
    revised = dict(report)
    notes = list(_strings(report.get("revision_notes")))
    revised["bull_case"] = _strings(report.get("bull_case"))
    revised["bear_case"] = _strings(report.get("bear_case"))
    revised["key_figures"] = _strings(report.get("key_figures"))
    revised["catalysts"] = _strings(report.get("catalysts"))

    if len(revised["key_figures"]) < 4:
        revised["key_figures"] = list(evidence.get("figure_bank") or [])[:6]
        notes.append("Expanded the key-figures list from the tool outputs.")

    if len(revised["bull_case"]) < 2:
        revised["bull_case"] = _collect_points(evidence, "findings")[:3] or revised["bull_case"]
        notes.append("Added bull points from the specialist findings.")
    if len(revised["bear_case"]) < 2:
        risks = _collect_points(evidence, "risks")
        if len(risks) < 2:
            risks = risks + _fallback_risks(evidence)
        revised["bear_case"] = risks[:3]
        notes.append("Expanded the bear case so the brief states two distinct risks.")

    quantified = [item for item in revised["catalysts"] if any(char.isdigit() for char in item)]
    if len(quantified) < 2:
        revised["catalysts"] = _catalysts(evidence)
        notes.append("Replaced vague catalysts with thresholds taken from the evidence.")

    prior_check = str(revised.get("prior_thesis_check") or "").strip()
    if evidence.get("had_prior_thesis") and not prior_check:
        revised["prior_thesis_check"] = _prior_check(evidence, str(revised.get("stance") or ""))
        notes.append("Added a check against the prior thesis stored in memory.")

    stance = str(revised.get("stance") or "")
    period_return = evidence.get("period_return")
    net = evidence.get("news_net_sentiment") or 0
    conflict = (
        stance == "Constructive"
        and isinstance(net, (int, float))
        and net < 0
        and isinstance(period_return, (int, float))
        and period_return < 0
    ) or (
        stance == "Cautious"
        and isinstance(net, (int, float))
        and net > 0
        and isinstance(period_return, (int, float))
        and period_return > 0.1
    )
    if conflict:
        revised["stance"] = "Balanced"
        revised["thesis"] = (
            "The first draft's stance fought the combined price and news signal, "
            "so this revision uses a balanced stance. "
            + str(revised.get("thesis") or "")
        )
        notes.append("Softened the stance to Balanced because it conflicted with the evidence.")

    if feedback:
        notes.append("Addressed evaluator feedback: " + feedback[0])
    revised["revision"] = int(revised.get("revision") or 0) + 1
    revised["revision_notes"] = notes
    return revised


def reflect_on_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Write the structured memory that the next planner must read."""
    symbol = str(payload.get("symbol") or "").upper()
    stance = str(payload.get("stance") or "Balanced")
    news_count = int(payload.get("news_count") or 0)
    had_income = bool(payload.get("had_income_statement"))
    feedback = payload.get("feedback") or []
    overall = payload.get("overall")
    lessons: list[str] = []
    gaps: list[str] = []
    focus: list[str] = []

    if not had_income:
        gaps.append("The income statement was not reviewed, so the margin trend is unverified.")
        lessons.append(
            "On the next run, pull the income statement and compare margins with this thesis."
        )
        focus.append("Verify the revenue and net-margin trend on the income statement.")
    else:
        lessons.append("Income-statement history was reviewed and should stay in the plan.")

    if news_count < 5:
        gaps.append("Fewer than five headlines were available, so the news view is fragile.")
        lessons.append("Increase the news limit when the first sample is sparse.")
        news_next = 15
    else:
        lessons.append(
            "The headline sample was large enough to score sentiment, but it is not a filing."
        )
        news_next = max(news_count, 8)

    if feedback:
        lessons.append("Carry this evaluator note forward: " + str(feedback[0]))
    if isinstance(overall, (int, float)) and overall < 4:
        gaps.append("The brief left the evaluator loop below the 4.0 bar.")
    focus.append(f"Challenge the prior {stance} stance against fresh prices and headlines.")

    score_text = f"{overall:.2f}" if isinstance(overall, (int, float)) else "n/a"
    narrative = (
        f"{symbol} closed this run with a {stance} stance and an evaluator score of "
        f"{score_text}. "
        + " ".join(lessons)
    )
    return {
        "lessons": lessons,
        "gaps": gaps,
        "next_run": {
            "pull_income_statement": not had_income,
            "news_count": news_next,
            "focus": focus,
        },
        "narrative": narrative,
    }


def coerce_report(data: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Keep a model-written report inside the fields the renderer understands."""
    stance = str(data.get("stance") or "Balanced")
    if stance not in STANCES:
        stance = "Balanced"
    return {
        "symbol": symbol.upper(),
        "company_name": str(data.get("company_name") or symbol.upper()),
        "stance": stance,
        "thesis": str(data.get("thesis") or "").strip(),
        "market_section": str(data.get("market_section") or "").strip(),
        "fundamentals_section": str(data.get("fundamentals_section") or "").strip(),
        "news_section": str(data.get("news_section") or "").strip(),
        "bull_case": _strings(data.get("bull_case")),
        "bear_case": _strings(data.get("bear_case")),
        "key_figures": _strings(data.get("key_figures")),
        "catalysts": _strings(data.get("catalysts")),
        "prior_thesis_check": str(data.get("prior_thesis_check") or "").strip(),
        "revision": int(data.get("revision") or 0),
        "revision_notes": _strings(data.get("revision_notes")),
    }


def coerce_scores(data: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Clamp model scores and recompute the overall from those five dimensions."""
    raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    scores = {}
    for name in DIMENSIONS:
        value = raw_scores.get(name, 1)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 1
        scores[name] = min(5, max(1, int(round(number))))
    overall = round(sum(scores.values()) / len(DIMENSIONS), 2)
    feedback = _strings(data.get("feedback"))
    return {
        "scores": scores,
        "overall": overall,
        "passed": overall >= threshold,
        "feedback": feedback,
    }


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _collect_points(evidence: dict[str, Any], field: str) -> list[str]:
    points: list[str] = []
    specialists = evidence.get("specialists") or {}
    if isinstance(specialists, dict):
        for name in ("market", "earnings", "news"):
            report = specialists.get(name) or {}
            if isinstance(report, dict):
                points.extend(_strings(report.get(field)))
    return points


def _fallback_risks(evidence: dict[str, Any]) -> list[str]:
    risks = []
    volatility = evidence.get("annualized_volatility")
    if isinstance(volatility, (int, float)):
        risks.append(
            "Annualized volatility of "
            f"{fmt_pct(float(volatility))} means the position can move sharply "
            "before the next review."
        )
    risks.append(
        "Headline sentiment can reverse, and this sample is not a substitute for "
        "the company filings."
    )
    return risks


def _catalysts(evidence: dict[str, Any]) -> list[str]:
    currency = str(evidence.get("currency") or "USD")
    net = evidence.get("news_net_sentiment")
    net_text = "0" if net is None else str(net)
    catalysts = [
        "Re-run the brief if a fresh sample of at least 8 headlines moves the "
        f"net sentiment count away from {net_text}.",
    ]
    price = evidence.get("last_price")
    high = evidence.get("high")
    low = evidence.get("low")
    if isinstance(price, (int, float)) and isinstance(high, (int, float)):
        low_text = fmt_price(float(low), currency) if isinstance(low, (int, float)) else "n/a"
        catalysts.append(
            f"The last price is {fmt_price(float(price), currency)}. A close through "
            f"the window high of {fmt_price(float(high), currency)}, or a break of "
            f"the window low at {low_text}, would change the stance."
        )
    target = evidence.get("target_mean")
    upside = evidence.get("target_upside")
    if isinstance(target, (int, float)) and isinstance(price, (int, float)):
        catalysts.append(
            f"The analyst mean target is {fmt_price(float(target), currency)}, "
            f"{fmt_pct(float(upside) if isinstance(upside, (int, float)) else None)} "
            "from the last price."
        )
    growth = evidence.get("revenue_growth")
    if isinstance(growth, (int, float)):
        catalysts.append(
            f"The stance depends on revenue growth staying near {fmt_pct(float(growth))}. "
            "A later print below 0% would weaken it."
        )
    return catalysts[:4]


def _prior_check(evidence: dict[str, Any], stance: str) -> str:
    note = evidence.get("prior_note") or {}
    if not isinstance(note, dict):
        note = {}
    prior_stance = note.get("stance") or "an earlier"
    prior_thesis = str(note.get("thesis") or "").strip()
    when = str(note.get("timestamp") or "the previous run")
    focus = []
    next_run = note.get("next_run") if isinstance(note.get("next_run"), dict) else {}
    if isinstance(next_run, dict):
        focus = _strings(next_run.get("focus"))
    relation = (
        "This run keeps that stance."
        if str(prior_stance) == stance
        else f"This run moves the stance from {prior_stance} to {stance}."
    )
    focus_text = " ".join(focus) if focus else "No extra focus was stored."
    thesis_text = prior_thesis or "No thesis text was stored."
    return (
        f"On {when} the prior stance was {prior_stance}. {relation} "
        f"Prior thesis: {thesis_text} Focus carried into this run: {focus_text}"
    )


def income_sentence(statement: dict[str, Any] | None) -> str:
    """Describe the revenue and margin path when a statement was retrieved."""
    if not isinstance(statement, dict) or not statement.get("available"):
        return ""
    periods = statement.get("periods") or []
    revenue = statement.get("revenue") or []
    if len(periods) < 2 or len(revenue) < 2:
        return ""
    first_margin = statement.get("net_margin_first")
    last_margin = statement.get("net_margin_last")
    return (
        f"Across the income statement from {periods[0]} to {periods[-1]}, revenue "
        f"moved from {fmt_big(revenue[0])} to {fmt_big(revenue[-1])} "
        f"({fmt_pct(statement.get('revenue_change'))}) and net margin moved from "
        f"{fmt_pct(first_margin)} to {fmt_pct(last_margin)}."
    )


def describe_multiple(value: float | None) -> str:
    """Format a price-to-earnings multiple."""
    if value is None:
        return "n/a"
    return f"{fmt_number(value, 1)}x"
