"""Display helpers for prices, percents, and large dollar amounts."""

from __future__ import annotations


def fmt_pct(value: float | None, digits: int = 1) -> str:
    """Format a decimal ratio as a percent string."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def fmt_number(value: float | None, digits: int = 2) -> str:
    """Format a plain number, or ``n/a`` when it is missing."""
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def fmt_big(value: float | None) -> str:
    """Format a dollar amount with thousands, millions, billions, or trillions."""
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{sign}${magnitude / 1e12:.2f}T"
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:.2f}M"
    return f"{sign}${magnitude:,.0f}"


def fmt_price(value: float | None, currency: str = "USD") -> str:
    """Format a traded price with its currency code."""
    if value is None:
        return "n/a"
    return f"{currency} {value:,.2f}"


def fmt_volume(value: float | None) -> str:
    """Format a share volume as a whole number."""
    if value is None:
        return "n/a"
    return f"{value:,.0f}"
