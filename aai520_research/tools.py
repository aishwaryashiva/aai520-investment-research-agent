"""Market, fundamental, news, and analyst tools backed by Yahoo Finance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from aai520_research.models import ToolResult


class ToolRegistry:
    """Catalog of tools the planner is allowed to call."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, fn: Any, description: str) -> None:
        self._tools[name] = {"fn": fn, "description": description}

    def names(self) -> set[str]:
        return set(self._tools)

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": meta["description"]}
            for name, meta in self._tools.items()
        ]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = dict(arguments or {})
        if name not in self._tools:
            return ToolResult(
                tool=name,
                arguments=arguments,
                ok=False,
                data={},
                error=f"Unknown tool: {name}",
            )
        try:
            data = self._tools[name]["fn"](**arguments)
        except TypeError as exc:
            return ToolResult(
                tool=name,
                arguments=arguments,
                ok=False,
                data={},
                error=f"Bad arguments for {name}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - tools must fail into a result
            return ToolResult(
                tool=name,
                arguments=arguments,
                ok=False,
                data={},
                error=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(data, dict):
            return ToolResult(
                tool=name,
                arguments=arguments,
                ok=False,
                data={},
                error=f"{name} did not return a dictionary.",
            )
        return ToolResult(tool=name, arguments=arguments, ok=True, data=data)


def as_float(value: Any) -> float | None:
    """Convert a vendor value to float, treating blanks and NaN as missing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_article(raw: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    """Map either Yahoo news shape onto title, publisher, link, and summary."""
    if not isinstance(raw, dict):
        return None
    content = raw.get("content") if isinstance(raw.get("content"), dict) else None
    if content is not None:
        title = str(content.get("title") or "").strip()
        provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
        publisher = str(provider.get("displayName") or "").strip()
        url_block = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = str(url_block.get("url") or "").strip() if isinstance(url_block, dict) else ""
        summary = str(content.get("summary") or content.get("description") or "").strip()
        published = _published_text(content.get("pubDate"))
        related = content.get("relatedTickers") or raw.get("relatedTickers") or []
    else:
        title = str(raw.get("title") or "").strip()
        publisher = str(raw.get("publisher") or "").strip()
        link = str(raw.get("link") or "").strip()
        summary = str(raw.get("summary") or "").strip()
        published = _published_text(raw.get("providerPublishTime"))
        related = raw.get("relatedTickers") or []
    if not title:
        return None
    related_symbols = [str(item).upper() for item in related] if isinstance(related, list) else []
    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "summary": summary[:600],
        "published": published,
        "related_symbols": related_symbols,
        "symbol": symbol.upper(),
    }


def _name_tokens(symbol: str, short_name: Any) -> set[str]:
    """Ticker plus distinctive words from the company name."""
    stop = {
        "CORPORATION", "COMPANY", "INCORPORATED", "GROUP", "HOLDINGS",
        "LIMITED", "CLASS", "INC", "PLC", "THE",
    }
    tokens = {symbol.upper()}
    for word in str(short_name or "").replace(",", " ").replace(".", " ").split():
        cleaned = "".join(char for char in word.upper() if char.isalpha())
        if len(cleaned) >= 4 and cleaned not in stop:
            tokens.add(cleaned)
    return tokens


def _mentions(article: dict[str, Any], tokens: set[str]) -> bool:
    blob = f"{article.get('title') or ''} {article.get('summary') or ''}".upper()
    return any(token in blob for token in tokens)


def _moving_average(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    value = close.rolling(window).mean().iloc[-1]
    return as_float(value)


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    """Wilder-style RSI from the last rolling average of gains and losses."""
    if len(close) < window + 1:
        return None
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta.clip(upper=0)).rolling(window).mean()
    avg_gain = as_float(gains.iloc[-1])
    avg_loss = as_float(losses.iloc[-1])
    if avg_gain is None or avg_loss is None:
        return None
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else None
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _published_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            moment = datetime.fromtimestamp(int(value), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
        return moment.date().isoformat()
    text = str(value).strip()
    return text[:10] if text else None


class YahooClient:
    """Cached Yahoo Finance access used by the market tools."""

    def __init__(self) -> None:
        self._tickers: dict[str, yf.Ticker] = {}
        self._info: dict[str, dict[str, Any]] = {}

    def ticker(self, symbol: str) -> yf.Ticker:
        key = symbol.upper().strip()
        if key not in self._tickers:
            self._tickers[key] = yf.Ticker(key)
        return self._tickers[key]

    def info(self, symbol: str) -> dict[str, Any]:
        key = symbol.upper().strip()
        if key not in self._info:
            try:
                raw = self.ticker(key).info or {}
            except Exception:
                raw = {}
            self._info[key] = raw if isinstance(raw, dict) else {}
        return self._info[key]

    def price_history(self, symbol: str, period: str = "1y") -> dict[str, Any]:
        key = symbol.upper().strip()
        history = self.ticker(key).history(period=period, auto_adjust=True)
        if history is None or history.empty or "Close" not in history.columns:
            raise RuntimeError(f"No price history returned for {key}.")
        close = history["Close"].dropna()
        if close.empty:
            raise RuntimeError(f"No closing prices returned for {key}.")
        returns = close.pct_change().dropna()
        last_price = float(close.iloc[-1])
        first_price = float(close.iloc[0])
        period_return = (last_price / first_price - 1.0) if first_price else None
        volatility = None
        if len(returns) >= 5:
            volatility = float(returns.std() * (252 ** 0.5))
        drawdown = float((close / close.cummax() - 1.0).min())
        profile = self.info(key)
        high = as_float(profile.get("fiftyTwoWeekHigh"))
        low = as_float(profile.get("fiftyTwoWeekLow"))
        if high is None:
            high = float(close.max())
        if low is None:
            low = float(close.min())
        monthly = close.resample("ME").last().dropna()
        monthly_closes = [
            {"date": index.strftime("%Y-%m-%d"), "close": round(float(value), 2)}
            for index, value in monthly.items()
        ]
        volume = history["Volume"].dropna() if "Volume" in history.columns else None
        sma_20 = _moving_average(close, 20)
        sma_50 = _moving_average(close, 50)
        return {
            "symbol": key,
            "period": period,
            "currency": str(profile.get("currency") or "USD"),
            "last_price": round(last_price, 4),
            "period_return": period_return,
            "annualized_volatility": volatility,
            "max_drawdown": drawdown,
            "average_volume": float(volume.mean()) if volume is not None and len(volume) else None,
            "high": high,
            "low": low,
            "sma_20": None if sma_20 is None else round(sma_20, 4),
            "sma_50": None if sma_50 is None else round(sma_50, 4),
            "rsi_14": _rsi(close, 14),
            "observations": int(close.shape[0]),
            "monthly_closes": monthly_closes,
        }

    def fundamentals(self, symbol: str) -> dict[str, Any]:
        key = symbol.upper().strip()
        profile = self.info(key)
        if not profile:
            raise RuntimeError(f"No fundamentals returned for {key}.")
        return {
            "symbol": key,
            "short_name": profile.get("shortName") or profile.get("longName") or key,
            "sector": profile.get("sector") or "Unknown",
            "industry": profile.get("industry") or "Unknown",
            "market_cap": as_float(profile.get("marketCap")),
            "trailing_pe": as_float(profile.get("trailingPE")),
            "forward_pe": as_float(profile.get("forwardPE")),
            "profit_margin": as_float(profile.get("profitMargins")),
            "revenue_growth": as_float(profile.get("revenueGrowth")),
            "debt_to_equity": as_float(profile.get("debtToEquity")),
            "return_on_equity": as_float(profile.get("returnOnEquity")),
            "free_cash_flow": as_float(profile.get("freeCashflow")),
            "total_revenue": as_float(profile.get("totalRevenue")),
            "trailing_eps": as_float(profile.get("trailingEps")),
            "currency": str(profile.get("currency") or "USD"),
        }

    def company_news(self, symbol: str, limit: int = 8) -> dict[str, Any]:
        key = symbol.upper().strip()
        requested = min(max(int(limit), 1), 20)
        articles, source = self._fetch_articles(key, min(20, max(requested * 2, 8)))
        normalized = []
        for raw in articles:
            item = normalize_article(raw, key)
            if item is not None:
                normalized.append(item)
        matched = [
            item
            for item in normalized
            if not item["related_symbols"] or key in item["related_symbols"]
        ]
        pool = matched or normalized
        names = _name_tokens(key, self.info(key).get("shortName"))
        mentioned = [item for item in pool if _mentions(item, names)]
        chosen = mentioned or pool
        items = [
            {
                "title": item["title"],
                "publisher": item["publisher"],
                "link": item["link"],
                "summary": item["summary"],
                "published": item["published"],
            }
            for item in chosen[:requested]
        ]
        return {"symbol": key, "source": source, "items": items}

    def analyst_snapshot(self, symbol: str) -> dict[str, Any]:
        key = symbol.upper().strip()
        profile = self.info(key)
        return {
            "symbol": key,
            "target_mean": as_float(profile.get("targetMeanPrice")),
            "target_high": as_float(profile.get("targetHighPrice")),
            "target_low": as_float(profile.get("targetLowPrice")),
            "recommendation": profile.get("recommendationKey"),
            "recommendation_mean": as_float(profile.get("recommendationMean")),
            "opinions": as_float(profile.get("numberOfAnalystOpinions")),
            "currency": str(profile.get("currency") or "USD"),
        }

    def income_statement(self, symbol: str) -> dict[str, Any]:
        key = symbol.upper().strip()
        frame = self.ticker(key).income_stmt
        empty = {
            "symbol": key,
            "available": False,
            "periods": [],
            "revenue": [],
            "net_income": [],
            "net_margin": [],
            "revenue_change": None,
            "net_margin_first": None,
            "net_margin_last": None,
        }
        if frame is None or getattr(frame, "empty", True):
            return empty
        revenue = _series(frame, ("Total Revenue", "Operating Revenue"))
        income = _series(frame, ("Net Income", "Net Income Common Stockholders"))
        if revenue is None or income is None:
            return empty
        periods = []
        revenues = []
        incomes = []
        margins = []
        for stamp, revenue_value in revenue:
            income_value = dict(income).get(stamp)
            if revenue_value is None or income_value is None:
                continue
            periods.append(stamp)
            revenues.append(revenue_value)
            incomes.append(income_value)
            margins.append(income_value / revenue_value if revenue_value else None)
        if len(periods) < 2:
            return empty
        first_revenue = revenues[0]
        change = (revenues[-1] / first_revenue - 1.0) if first_revenue else None
        return {
            "symbol": key,
            "available": True,
            "periods": periods,
            "revenue": revenues,
            "net_income": incomes,
            "net_margin": margins,
            "revenue_change": change,
            "net_margin_first": margins[0],
            "net_margin_last": margins[-1],
        }

    def _fetch_articles(self, symbol: str, limit: int) -> tuple[list[dict[str, Any]], str]:
        try:
            search = yf.Search(symbol, news_count=limit)
            if search.news:
                return list(search.news), "yahoo-search"
        except Exception:
            pass
        try:
            ticker = yf.Ticker(symbol)
            ticker._news = []
            raw_news = ticker.get_news(count=limit)
            if raw_news:
                return list(raw_news), "yahoo-news"
        except Exception:
            pass
        return [], "none"


def _series(frame: pd.DataFrame, names: tuple[str, ...]) -> list[tuple[str, float | None]] | None:
    labels = {str(index).lower(): index for index in frame.index}
    row = None
    for name in names:
        if name in frame.index:
            row = frame.loc[name]
            break
        if name.lower() in labels:
            row = frame.loc[labels[name.lower()]]
            break
    if row is None:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    points = []
    for column, value in row.items():
        stamp = pd.Timestamp(column).strftime("%Y-%m-%d")
        points.append((stamp, as_float(value)))
    points.sort(key=lambda item: item[0])
    return points


def build_market_registry(client: YahooClient | None = None) -> ToolRegistry:
    """Register the Yahoo Finance tools the planner can select."""
    client = client or YahooClient()
    registry = ToolRegistry()
    registry.register(
        "get_price_history",
        client.price_history,
        "Daily prices plus return, volatility, drawdown, and monthly closes.",
    )
    registry.register(
        "get_fundamentals",
        client.fundamentals,
        "Company profile, valuation, growth, profitability, and cash flow.",
    )
    registry.register(
        "get_company_news",
        client.company_news,
        "Recent headlines for the news prompt chain.",
    )
    registry.register(
        "get_analyst_snapshot",
        client.analyst_snapshot,
        "Consensus recommendation and published price targets.",
    )
    registry.register(
        "get_income_statement",
        client.income_statement,
        "Multi-period revenue, net income, and net margin.",
    )
    return registry
