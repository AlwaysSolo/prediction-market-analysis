from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from src.live.kalshi.signal_risk import KalshiTradeIntent

TradeIntentCallback = Callable[[KalshiTradeIntent], Awaitable[None] | None]


class KalshiTradeIntentSource(Protocol):
    def subscribe_trade_intent_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiTradeIntent]:
        ...

    def subscribe_trade_intent_callback(self, callback: TradeIntentCallback) -> None:
        ...

    def unsubscribe_trade_intent_callback(self, callback: TradeIntentCallback) -> None:
        ...
