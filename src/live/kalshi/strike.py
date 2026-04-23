from __future__ import annotations

import re
from typing import Any

from src.indexers.kalshi.models import Market

_STRIKE_PATTERNS = (
    re.compile(r"(?:price to beat|target price):\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*target", re.IGNORECASE),
)


def extract_strike_price_dollars(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value)
    for pattern in _STRIKE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            continue
    return None


def extract_price_to_beat_dollars(value: str | None) -> float | None:
    return extract_strike_price_dollars(value)


def strike_price_from_market(market: Market | dict[str, Any] | None) -> float | None:
    if market is None:
        return None
    if isinstance(market, dict):
        candidates = (
            market.get("yes_sub_title"),
            market.get("no_sub_title"),
            market.get("title"),
        )
    else:
        candidates = (
            market.yes_sub_title,
            market.no_sub_title,
            market.title,
        )
    for value in candidates:
        strike_price = extract_strike_price_dollars(str(value) if value is not None else None)
        if strike_price is not None:
            return strike_price
    return None
