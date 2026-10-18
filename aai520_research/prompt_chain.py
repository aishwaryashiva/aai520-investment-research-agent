"""Prompt-chaining workflow for financial news.

The stages are ingest, preprocess, classify, extract, and summarize.
Ingest and preprocess are deterministic. Classify, extract, and summarize
go through the language-model interface so a hosted model can replace the
local rules without changing the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aai520_research.models import ChainResult
from aai520_research.prompts import CLASSIFY_SYSTEM, EXTRACT_SYSTEM, SUMMARIZE_SYSTEM

CATEGORIES = ("earnings", "macro", "product", "legal", "analyst", "other")
SENTIMENTS = ("positive", "negative", "neutral")

_CATEGORY_TERMS = (
    (
        "legal",
        {
            "lawsuit", "regulator", "regulators", "regulatory", "probe",
            "investigation", "antitrust", "fine", "sec",
        },
    ),
    ("analyst", {"upgrade", "downgrade", "rating", "analyst", "analysts", "target"}),
    (
        "earnings",
        {
            "earnings", "eps", "revenue", "profit", "profits", "guidance",
            "quarter", "margin", "margins",
        },
    ),
    (
        "macro",
        {"fed", "inflation", "rates", "rate", "recession", "gdp", "treasury", "economy"},
    ),
    ("product", {"launch", "product", "partnership", "strategy", "ai", "cloud", "software"}),
)
_POSITIVE = {
    "beat", "beats", "surge", "rally", "upgrade", "record", "growth", "strong",
    "boost", "outperform", "raised", "raise", "gain", "gains", "accelerates",
    "acceleration", "high", "higher",
}
_NEGATIVE = {
    "miss", "misses", "fall", "falls", "drop", "drops", "downgrade", "weak",
    "cut", "lawsuit", "probe", "decline", "loss", "plunge", "warning",
    "layoff", "layoffs", "investigation", "pressure", "fine", "lower",
}


class NewsPromptChain:
    """Run headlines through the five-stage chain and keep every artifact."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, articles: list[dict[str, Any]]) -> ChainResult:
        ingested = ingest(articles)
        preprocessed, dropped = preprocess(ingested)
        classified = self._classify(preprocessed)
        facts = self._extract(classified)
        brief = self._summarize(classified, facts)
        return ChainResult(
            ingested=ingested,
            preprocessed=preprocessed,
            classified=classified,
            facts=facts,
            brief=brief,
            dropped=dropped,
        )

    def _classify(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        payload = {
            "items": [
                {"id": item["id"], "title": item["title"], "summary": item["summary"]}
                for item in items
            ]
        }
        raw = self.llm.complete_json(CLASSIFY_SYSTEM, json.dumps(payload))
        labels = raw.get("labels") if isinstance(raw, dict) else None
        return align_labels(items, labels)

    def _extract(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        raw = self.llm.complete_json(EXTRACT_SYSTEM, json.dumps({"items": items}))
        facts = raw.get("facts") if isinstance(raw, dict) else None
        cleaned = []
        if isinstance(facts, list):
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                statement = str(fact.get("statement") or "").strip()
                if not statement:
                    continue
                cleaned.append(
                    {
                        "statement": statement,
                        "category": fact.get("category")
                        if fact.get("category") in CATEGORIES
                        else "other",
                        "sentiment": fact.get("sentiment")
                        if fact.get("sentiment") in SENTIMENTS
                        else "neutral",
                        "source_title": str(fact.get("source_title") or ""),
                    }
                )
        return cleaned or [fact_from_item(item) for item in items]

    def _summarize(
        self,
        items: list[dict[str, Any]],
        facts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sentiment_counts = _count(items, "sentiment", SENTIMENTS)
        category_counts = _count(items, "category", CATEGORIES)
        payload = {
            "facts": facts,
            "sentiment_counts": sentiment_counts,
            "category_counts": category_counts,
            "item_count": len(items),
        }
        raw = self.llm.complete_json(SUMMARIZE_SYSTEM, json.dumps(payload))
        summary = str(raw.get("summary") or "").strip() if isinstance(raw, dict) else ""
        if not summary:
            summary = summarize_facts(payload)["summary"]
        return {
            "summary": summary,
            "facts": facts,
            "sentiment_counts": sentiment_counts,
            "category_counts": category_counts,
            "item_count": len(items),
        }


def ingest(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy raw articles into the chain and assign stable ids."""
    ingested = []
    for index, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        ingested.append(
            {
                "id": f"n{index}",
                "title": str(article.get("title") or "").strip(),
                "publisher": str(article.get("publisher") or "").strip(),
                "link": str(article.get("link") or "").strip(),
                "summary": str(article.get("summary") or "").strip(),
                "published": article.get("published"),
            }
        )
    return ingested


def preprocess(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop empty titles and duplicate headlines."""
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    for item in items:
        title = re.sub(r"\s+", " ", item.get("title") or "").strip()
        if len(title) < 8:
            dropped += 1
            continue
        key = title.lower()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        summary = re.sub(r"\s+", " ", item.get("summary") or "").strip()
        kept.append({**item, "title": title, "summary": summary})
    return kept, dropped


def classify_with_rules(payload: dict[str, Any]) -> dict[str, Any]:
    """Keyword classifier used by the heuristic provider."""
    labels = []
    for item in payload.get("items") or []:
        text = f"{item.get('title') or ''} {item.get('summary') or ''}"
        category = _category(text)
        sentiment = _sentiment(text)
        labels.append(
            {
                "id": item.get("id"),
                "category": category,
                "sentiment": sentiment,
                "rationale": f"Terms in the headline map to {category} and {sentiment}.",
            }
        )
    return {"labels": labels}


def extract_with_rules(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn each classified headline into one citable statement."""
    facts = []
    for item in payload.get("items") or []:
        facts.append(fact_from_item(item))
    return {"facts": facts}


def summarize_facts(payload: dict[str, Any]) -> dict[str, Any]:
    """Write the news brief from counts and the first extracted claims."""
    count = int(payload.get("item_count") or 0)
    if count == 0:
        return {
            "summary": (
                "No headlines were available, so this brief cannot support a "
                "sentiment view from news."
            )
        }
    sentiments = payload.get("sentiment_counts") or {}
    categories = payload.get("category_counts") or {}
    dominant = "other"
    if categories:
        dominant = max(categories, key=lambda name: categories[name])
    facts = payload.get("facts") or []
    examples = [str(fact.get("statement")) for fact in facts[:2] if isinstance(fact, dict)]
    example_text = "; ".join(examples) if examples else "No extracted claim was available."
    summary = (
        f"The sample has {count} headlines: {int(sentiments.get('positive') or 0)} positive, "
        f"{int(sentiments.get('negative') or 0)} negative, and "
        f"{int(sentiments.get('neutral') or 0)} neutral. "
        f"The dominant category is {dominant}. Representative claims: {example_text}."
    )
    return {"summary": summary}


def align_labels(items: list[dict[str, Any]], labels: Any) -> list[dict[str, Any]]:
    """Attach model labels, falling back when an id or field is unusable."""
    by_id = {}
    if isinstance(labels, list):
        for label in labels:
            if isinstance(label, dict) and label.get("id") is not None:
                by_id[str(label["id"])] = label
    aligned = []
    for item in items:
        label = by_id.get(str(item["id"]), {})
        category = label.get("category")
        sentiment = label.get("sentiment")
        aligned.append(
            {
                **item,
                "category": category if category in CATEGORIES else "other",
                "sentiment": sentiment if sentiment in SENTIMENTS else "neutral",
                "rationale": str(label.get("rationale") or "").strip(),
            }
        )
    return aligned


def fact_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Build one statement from a headline and, when useful, its summary."""
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()
    statement = title
    if summary and summary.lower() not in title.lower():
        sentence = re.split(r"(?<=[.!?])\s", summary)[0].strip()
        if sentence:
            statement = f"{title}. {sentence}"
    return {
        "statement": statement,
        "category": item.get("category") if item.get("category") in CATEGORIES else "other",
        "sentiment": item.get("sentiment") if item.get("sentiment") in SENTIMENTS else "neutral",
        "source_title": title,
    }


def _category(text: str) -> str:
    tokens = _tokens(text)
    for name, vocab in _CATEGORY_TERMS:
        if tokens & vocab:
            return name
    if "price" in tokens and "target" in tokens:
        return "analyst"
    return "other"


def _sentiment(text: str) -> str:
    tokens = _tokens(text)
    positive = len(tokens & _POSITIVE)
    negative = len(tokens & _NEGATIVE)
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _count(items: list[dict[str, Any]], field: str, allowed: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in allowed}
    for item in items:
        label = item.get(field)
        if label in counts:
            counts[str(label)] += 1
    return counts
