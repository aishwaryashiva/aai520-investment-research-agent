"""Tests for news normalization and statement parsing."""

import pandas as pd

from aai520_research.tools import _series, normalize_article


def test_normalize_article_accepts_both_yahoo_shapes():
    search_shape = {
        "title": "Microsoft faces fresh regulatory pressure",
        "publisher": "Desk",
        "link": "https://finance.yahoo.com/story",
        "providerPublishTime": 1792264284,
        "relatedTickers": ["MSFT"],
    }
    stream_shape = {
        "content": {
            "title": "Microsoft updates its cloud lineup",
            "summary": "A short summary.",
            "pubDate": "2026-03-01T12:00:00Z",
            "provider": {"displayName": "Desk"},
            "canonicalUrl": {"url": "https://finance.yahoo.com/other"},
        }
    }
    first = normalize_article(search_shape, "msft")
    second = normalize_article(stream_shape, "MSFT")
    assert first is not None and first["title"].startswith("Microsoft faces")
    assert first["published"] is not None
    assert second is not None and second["summary"].startswith("A short")
    assert second["link"].endswith("/other")
    assert normalize_article({"title": "  "}, "MSFT") is None


def test_series_reads_revenue_in_chronological_order():
    frame = pd.DataFrame(
        {
            pd.Timestamp("2025-06-30"): [200.0, 20.0],
            pd.Timestamp("2024-06-30"): [180.0, 18.0],
        },
        index=["Total Revenue", "Net Income"],
    )
    revenue = _series(frame, ("Total Revenue",))
    assert revenue is not None
    assert [item[0] for item in revenue] == ["2024-06-30", "2025-06-30"]
    assert revenue[1][1] == 200.0
