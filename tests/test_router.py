"""Tests for specialist routing."""

from aai520_research.heuristic import HeuristicLLM
from aai520_research.router import ContentRouter


def test_structured_observations_have_fixed_specialists():
    router = ContentRouter(HeuristicLLM())
    prices = router.route(
        {"id": "prices", "kind": "market_data", "text": "price", "category": None}
    )
    fundamentals = router.route(
        {"id": "fundamentals", "kind": "fundamentals", "text": "margin", "category": None}
    )
    assert prices.route == "market"
    assert fundamentals.route == "earnings"


def test_headline_category_selects_the_specialist():
    router = ContentRouter(HeuristicLLM())
    earnings = router.route(
        {
            "id": "n0",
            "kind": "news",
            "text": "earnings beat",
            "category": "earnings",
            "sentiment": "positive",
        }
    )
    macro = router.route(
        {"id": "n1", "kind": "news", "text": "fed", "category": "macro", "sentiment": "neutral"}
    )
    product = router.route(
        {
            "id": "n2",
            "kind": "news",
            "text": "launch",
            "category": "product",
            "sentiment": "positive",
        }
    )
    assert earnings.route == "earnings"
    assert macro.route == "market"
    assert product.route == "news"
