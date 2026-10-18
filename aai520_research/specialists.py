"""Specialist readings and the first-pass research brief.

The first draft is a narrative synthesis. It states a thesis and a few
figures, and it leaves the evidence list, the two-sided case, and the
catalysts short of the rubric on purpose. The optimizer in ``evaluator``
is what brings the brief up to the bar.
"""

from __future__ import annotations

from typing import Any

from aai520_research.evaluator import choose_stance, describe_multiple, income_sentence
from aai520_research.formatting import fmt_big, fmt_number, fmt_pct, fmt_price, fmt_volume


def market_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Read the price snapshot into findings and risks."""
    currency = str(payload.get("currency") or "USD")
    period = str(payload.get("period") or "1y")
    last_price = _num(payload.get("last_price"))
    period_return = _num(payload.get("period_return"))
    volatility = _num(payload.get("annualized_volatility"))
    drawdown = _num(payload.get("max_drawdown"))
    high = _num(payload.get("high"))
    low = _num(payload.get("low"))
    volume = _num(payload.get("average_volume"))
    findings = [
        (
            f"The last price is {fmt_price(last_price, currency)} and the {period} "
            f"total return is {fmt_pct(period_return)}."
        ),
        (
            f"Annualized volatility over the window is {fmt_pct(volatility)} and the "
            f"maximum drawdown is {fmt_pct(drawdown)}."
        ),
    ]
    if last_price and high and low and high > 0 and low > 0:
        findings.append(
            f"The last price is {fmt_pct(last_price / high - 1)} from the window high "
            f"of {fmt_price(high, currency)} and {fmt_pct(last_price / low - 1)} from "
            f"the window low of {fmt_price(low, currency)}."
        )
    if volume is not None:
        findings.append(f"Average daily volume over the window is {fmt_volume(volume)} shares.")
    sma_20 = _num(payload.get("sma_20"))
    sma_50 = _num(payload.get("sma_50"))
    rsi = _num(payload.get("rsi_14"))
    if last_price and sma_20 and sma_50:
        if last_price > sma_20 > sma_50:
            relation = "above both averages, so the short trend agrees with the longer one"
        elif last_price < sma_20 < sma_50:
            relation = "below both averages, so the short trend agrees with a downtrend"
        else:
            relation = "between the two averages, so the trend is mixed"
        findings.append(
            f"The 20-day SMA is {fmt_price(sma_20, currency)} and the 50-day SMA is "
            f"{fmt_price(sma_50, currency)}. Price is {relation}."
        )
    if rsi is not None:
        if rsi >= 70:
            zone = "overbought"
        elif rsi <= 30:
            zone = "oversold"
        else:
            zone = "inside the 30 to 70 band"
        findings.append(f"RSI(14) is {rsi:.1f}, {zone}.")
    headlines = _lines(payload.get("headlines"))
    if headlines:
        findings.append("Macro headlines routed here: " + "; ".join(headlines[:2]) + ".")
    risks = []
    if volatility is not None and volatility >= 0.30:
        risks.append(
            f"Volatility of {fmt_pct(volatility)} is high enough that position size matters."
        )
    if drawdown is not None and drawdown <= -0.20:
        risks.append(
            f"The window already includes a drawdown of {fmt_pct(drawdown)}."
        )
    if period_return is not None and period_return < -0.10:
        risks.append(f"The {period} return of {fmt_pct(period_return)} is a downtrend.")
    if len(risks) < 2:
        risks.append(
            "A one-year price window does not show whether the move is liquidity, "
            "multiple, or earnings."
        )
    if len(risks) < 2:
        risks.append("A failed retest of the window high would weaken the market read.")
    if rsi is not None and rsi >= 70:
        risks.append(f"RSI(14) at {rsi:.1f} is a stretched reading.")
    confidence = 0.8 if last_price is not None else 0.35
    return {"findings": findings[:6], "risks": risks[:3], "confidence": confidence}


def earnings_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Read fundamentals, the statement, and the analyst snapshot."""
    fundamentals = payload.get("fundamentals") or {}
    analyst = payload.get("analyst") or {}
    statement = payload.get("income_statement")
    name = fundamentals.get("short_name") or payload.get("symbol") or "The company"
    sector = fundamentals.get("sector") or "an unclassified sector"
    industry = fundamentals.get("industry") or "an unclassified industry"
    currency = str(fundamentals.get("currency") or payload.get("currency") or "USD")
    findings = [
        (
            f"{name} is in {sector} / {industry}, with a market cap of "
            f"{fmt_big(_num(fundamentals.get('market_cap')))}."
        ),
        (
            "Trailing P/E is "
            f"{describe_multiple(_num(fundamentals.get('trailing_pe')))} and forward "
            f"P/E is {describe_multiple(_num(fundamentals.get('forward_pe')))}."
        ),
        (
            f"Reported revenue growth is {fmt_pct(_num(fundamentals.get('revenue_growth')))} "
            f"and the profit margin is {fmt_pct(_num(fundamentals.get('profit_margin')))}."
        ),
    ]
    margin_path = income_sentence(statement if isinstance(statement, dict) else None)
    if margin_path:
        findings.append(margin_path)
    target = _num(analyst.get("target_mean"))
    if target is not None:
        opinions = analyst.get("opinions")
        opinion_text = (
            f" across {fmt_number(float(opinions), 0)} opinions"
            if isinstance(opinions, (int, float))
            else ""
        )
        recommendation = analyst.get("recommendation") or "no recommendation label"
        findings.append(
            f"The analyst mean target is {fmt_price(target, currency)}{opinion_text}, "
            f"with a vendor label of {recommendation}."
        )
    headlines = _lines(payload.get("headlines"))
    if headlines:
        findings.append("Earnings headlines routed here: " + "; ".join(headlines[:2]) + ".")
    risks = []
    trailing = _num(fundamentals.get("trailing_pe"))
    margin = _num(fundamentals.get("profit_margin"))
    growth = _num(fundamentals.get("revenue_growth"))
    debt = _num(fundamentals.get("debt_to_equity"))
    if trailing is not None and trailing > 40:
        risks.append(
            f"A trailing P/E of {describe_multiple(trailing)} prices in strong growth, "
            "so a miss can compress the multiple."
        )
    if margin is not None and margin < 0.05:
        risks.append(
            f"A profit margin of {fmt_pct(margin)} leaves little room for a slowdown."
        )
    if growth is not None and growth < 0:
        risks.append(f"Revenue growth of {fmt_pct(growth)} means the top line is shrinking.")
    if debt is not None and debt > 150:
        risks.append(
            f"Yahoo Finance reports debt to equity of {fmt_number(debt, 1)}, which is elevated."
        )
    if len(risks) < 2:
        risks.append(
            "Published ratios can be moved by one-time items; the filing is the check."
        )
    if len(risks) < 2:
        risks.append("Consensus targets are opinions, not a forecast the business must meet.")
    populated = sum(
        value is not None
        for value in (trailing, margin, growth, fundamentals.get("market_cap"))
    )
    confidence = 0.75 if populated >= 3 else 0.45
    return {"findings": findings[:5], "risks": risks[:3], "confidence": confidence}


def news_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Read the prompt-chain brief and the headlines routed to news."""
    brief = payload.get("brief") or {}
    summary = str(brief.get("summary") or "").strip()
    headlines = [str(item) for item in payload.get("headlines") or [] if str(item).strip()]
    counts = brief.get("sentiment_counts") or {}
    count = int(brief.get("item_count") or len(headlines) or 0)
    findings = []
    if summary:
        findings.append(summary)
    if headlines:
        findings.append("Headlines in this route include: " + "; ".join(headlines[:3]) + ".")
    if not findings:
        findings.append("No company headlines were available for this run.")
    positive = int(counts.get("positive") or 0)
    negative = int(counts.get("negative") or 0)
    neutral = int(counts.get("neutral") or 0)
    findings.append(
        f"The sentiment tally is {positive} positive, {negative} negative, and "
        f"{neutral} neutral across {count} headlines."
    )
    risks = []
    if negative > positive:
        risks.append("Negative headlines outnumber positive ones in this sample.")
    if count < 5:
        risks.append("The news sample is small, so one headline can dominate the read.")
    risks.append("Headlines omit context that a filing or transcript would supply.")
    confidence = min(0.85, 0.35 + 0.08 * count)
    return {"findings": findings[:4], "risks": risks[:3], "confidence": round(confidence, 2)}


def compose_draft(evidence: dict[str, Any]) -> dict[str, Any]:
    """Build the first brief. Later revisions are the optimizer's job."""
    symbol = str(evidence.get("symbol") or "")
    company = str(evidence.get("company_name") or symbol)
    currency = str(evidence.get("currency") or "USD")
    stance = choose_stance(evidence)
    specialists = evidence.get("specialists") or {}
    market = specialists.get("market") or {}
    earnings = specialists.get("earnings") or {}
    news = specialists.get("news") or {}
    market_findings = _lines(market.get("findings"))
    earnings_findings = _lines(earnings.get("findings"))
    news_findings = _lines(news.get("findings"))
    thesis = (
        f"{company} ({symbol}) screens as {stance.lower()} on this pass. "
        f"The {evidence.get('period') or '1y'} return is "
        f"{fmt_pct(_num(evidence.get('period_return')))}, "
        f"revenue growth is {fmt_pct(_num(evidence.get('revenue_growth')))}, and the news "
        f"sample has a net sentiment of {evidence.get('news_net_sentiment') or 0}."
    )
    market_section = _paragraph(market_findings) or (
        f"Price history for {symbol} did not produce a usable market section."
    )
    fundamentals_bits = list(earnings_findings)
    statement = income_sentence(evidence.get("income_statement"))
    if statement and statement not in " ".join(fundamentals_bits):
        fundamentals_bits.append(statement)
    fundamentals_section = _paragraph(fundamentals_bits) or (
        f"Fundamentals for {symbol} were not available from the tool call."
    )
    news_section = _paragraph(news_findings) or f"No headlines were retrieved for {symbol}."
    bull_case = _bull_points(evidence, currency)
    bear_source = _lines((market.get("risks"))) + _lines(earnings.get("risks"))
    bear_case = bear_source[:1] or ["The first pass names only the most immediate risk."]
    figures = list(evidence.get("figure_bank") or [])[:2]
    return {
        "symbol": symbol,
        "company_name": company,
        "stance": stance,
        "thesis": thesis,
        "market_section": market_section,
        "fundamentals_section": fundamentals_section,
        "news_section": news_section,
        "bull_case": bull_case[:2],
        "bear_case": bear_case[:1],
        "key_figures": figures,
        "catalysts": ["The next earnings release and any formal guidance update."],
        "prior_thesis_check": "",
        "revision": 0,
        "revision_notes": [],
    }


def _bull_points(evidence: dict[str, Any], currency: str) -> list[str]:
    """Supportive points only. Drawdown and volatility belong in the bear case."""
    points = []
    growth = _num(evidence.get("revenue_growth"))
    margin = _num(evidence.get("profit_margin"))
    upside = _num(evidence.get("target_upside"))
    period_return = _num(evidence.get("period_return"))
    if growth is not None and growth > 0:
        points.append(
            f"Revenue growth of {fmt_pct(growth)} means the top line is still expanding."
        )
    if margin is not None and margin > 0.10:
        points.append(
            "A profit margin of "
            f"{fmt_pct(margin)} shows the business keeps a meaningful share of revenue."
        )
    if upside is not None and upside > 0:
        points.append(f"The analyst mean target is {fmt_pct(upside)} above the last price.")
    if period_return is not None and period_return > 0.05:
        points.append(
            f"The {evidence.get('period') or 'window'} return of "
            f"{fmt_pct(period_return)} is positive."
        )
    if len(points) < 2 and _num(evidence.get("market_cap")) is not None:
        points.append(
            f"Market cap is {fmt_big(_num(evidence.get('market_cap')))}, "
            "so the name is widely covered."
        )
    if len(points) < 2:
        points.append(
            f"The last price of {fmt_price(_num(evidence.get('last_price')), currency)} "
            "is a concrete level the next revision can test."
        )
    return points[:2]


def _paragraph(lines: list[str]) -> str:
    parts = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        parts.append(text)
    return " ".join(parts)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
