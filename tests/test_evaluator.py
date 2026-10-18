"""Tests for the rubric and the optimizer that responds to it."""

from aai520_research.evaluator import apply_rubric, refine_report


def _evidence():
    return {
        "currency": "USD",
        "period": "1y",
        "last_price": 100.0,
        "period_return": 0.12,
        "annualized_volatility": 0.22,
        "max_drawdown": -0.08,
        "high": 110.0,
        "low": 80.0,
        "trailing_pe": 20.0,
        "profit_margin": 0.25,
        "revenue_growth": 0.1,
        "market_cap": 2e11,
        "target_mean": 120.0,
        "target_upside": 0.2,
        "news_count": 6,
        "news_net_sentiment": 1,
        "had_prior_thesis": False,
        "figure_bank": [
            "Last price USD 100.00",
            "1y total return 12.0%",
            "Annualized volatility 22.0%",
            "Maximum drawdown -8.0%",
            "Trailing P/E 20.0x",
            "Profit margin 25.0%",
        ],
        "specialists": {
            "market": {
                "findings": ["Prices are higher over the year."],
                "risks": ["The window is only one year.", "Volume can dry up around holidays."],
            },
            "earnings": {
                "findings": ["Margins are healthy."],
                "risks": ["Multiples can compress after a beat."],
            },
            "news": {"findings": ["Headlines lean positive."], "risks": ["The sample is small."]},
        },
    }


def _weak_report():
    paragraph = "This section is intentionally long enough to satisfy the coverage check. " * 2
    return {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "stance": "Constructive",
        "thesis": paragraph,
        "market_section": paragraph,
        "fundamentals_section": paragraph,
        "news_section": paragraph,
        "bull_case": ["Demand is firm.", "Margins expanded."],
        "bear_case": ["The multiple is not cheap."],
        "key_figures": ["Last price USD 100.00", "1y total return 12.0%"],
        "catalysts": ["The next earnings release and any formal guidance update."],
        "prior_thesis_check": "",
        "revision": 0,
        "revision_notes": [],
    }


def test_weak_draft_fails_and_revision_passes():
    evidence = _evidence()
    draft = _weak_report()
    first = apply_rubric(draft, evidence)
    assert first["passed"] is False
    assert first["overall"] < 4
    revised = refine_report(draft, evidence, first["feedback"])
    second = apply_rubric(revised, evidence)
    assert second["passed"] is True
    assert len(revised["key_figures"]) >= 4
    assert len(revised["bear_case"]) >= 2
    assert revised["revision"] == 1


def test_conflicting_stance_lowers_consistency():
    evidence = _evidence()
    evidence["period_return"] = -0.2
    evidence["news_net_sentiment"] = -2
    report = _weak_report()
    scores = apply_rubric(report, evidence)["scores"]
    assert scores["consistency"] == 2


def test_missing_prior_thesis_check_is_penalized():
    evidence = _evidence()
    evidence["had_prior_thesis"] = True
    evidence["prior_note"] = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "stance": "Balanced",
        "thesis": "Earlier brief was balanced.",
        "next_run": {"focus": ["Check margins."]},
    }
    report = _weak_report()
    first = apply_rubric(report, evidence)
    assert "prior thesis" in " ".join(first["feedback"]).lower()
    revised = refine_report(report, evidence, first["feedback"])
    assert "Balanced" in revised["prior_thesis_check"]
    assert apply_rubric(revised, evidence)["scores"]["consistency"] == 5
