from __future__ import annotations

import asyncio
from typing import Protocol

from src.live.kalshi.signal_risk import KalshiTradeIntent


class KalshiTradeIntentSource(Protocol):
    def subscribe_trade_intent_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiTradeIntent]:
        ...
