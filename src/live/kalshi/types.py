from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class KalshiTickerState:
    ticker: str
    last_yes_price_cents: int | None
    last_trade_time: datetime | None
    previous_yes_price_cents: int | None
    close_time: datetime | None
    is_open: bool
    open_time: datetime | None = None
    trade_id: str | None = None
    count: int | None = None
    taker_side: str | None = None
    last_price_cents: int | None = None
    yes_bid_cents: int | None = None
    yes_ask_cents: int | None = None
    no_bid_cents: int | None = None
    no_ask_cents: int | None = None
    yes_bid_size: int | None = None
    yes_ask_size: int | None = None
    no_bid_size: int | None = None
    no_ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    dollar_volume: int | None = None
    dollar_open_interest: int | None = None
    ticker_update_time: datetime | None = None
    received_at: datetime | None = field(default=None, compare=False)
    event_id: str = field(default="", compare=False)
    raw_event_id: str | None = field(default=None, compare=False)

    @property
    def feature_price_cents(self) -> int | None:
        price_cents = self.last_yes_price_cents
        if price_cents is None:
            price_cents = self.last_price_cents
        return price_cents

    @property
    def market_prob(self) -> float | None:
        price_cents = self.feature_price_cents
        if price_cents is None:
            return None
        return price_cents / 100.0

    @property
    def trade_yes_prob(self) -> float | None:
        if self.last_yes_price_cents is not None:
            return self.last_yes_price_cents / 100.0
        if self.last_price_cents is not None:
            return self.last_price_cents / 100.0
        return None

    @property
    def quote_mid_prob(self) -> float | None:
        if self.yes_bid_cents is None or self.yes_ask_cents is None:
            if self.no_bid_cents is None or self.no_ask_cents is None:
                return None
            return (200.0 - self.no_bid_cents - self.no_ask_cents) / 200.0
        return (self.yes_bid_cents + self.yes_ask_cents) / 200.0

    @property
    def quote_spread_cents(self) -> int | None:
        if self.yes_bid_cents is None or self.yes_ask_cents is None:
            if self.no_bid_cents is None or self.no_ask_cents is None:
                return None
            return self.no_ask_cents - self.no_bid_cents
        return self.yes_ask_cents - self.yes_bid_cents

    @property
    def buy_yes_price_cents(self) -> int | None:
        if self.yes_ask_cents is not None:
            return self.yes_ask_cents
        if self.no_bid_cents is not None:
            return 100 - self.no_bid_cents
        return None

    @property
    def buy_no_price_cents(self) -> int | None:
        if self.no_ask_cents is not None:
            return self.no_ask_cents
        if self.yes_bid_cents is None:
            return None
        return 100 - self.yes_bid_cents

    @property
    def buy_yes_size(self) -> int | None:
        if self.yes_ask_size is not None:
            return self.yes_ask_size
        return self.no_bid_size

    @property
    def buy_no_size(self) -> int | None:
        if self.no_ask_size is not None:
            return self.no_ask_size
        return self.yes_bid_size

    @property
    def last_to_mid_gap(self) -> float | None:
        trade_yes_prob = self.trade_yes_prob
        quote_mid_prob = self.quote_mid_prob
        if trade_yes_prob is None or quote_mid_prob is None:
            return None
        return trade_yes_prob - quote_mid_prob

    def quote_age_seconds(self, now: datetime | None = None) -> float | None:
        if self.ticker_update_time is None:
            return None
        now = now or utc_now()
        return max(0.0, (now - self.ticker_update_time).total_seconds())

    @property
    def previous_market_prob(self) -> float | None:
        if self.previous_yes_price_cents is None:
            return None
        return self.previous_yes_price_cents / 100.0

    @property
    def price_momentum(self) -> float | None:
        if self.market_prob is None or self.previous_market_prob is None:
            return None
        return self.market_prob - self.previous_market_prob

    def tau_minutes(self, now: datetime | None = None) -> float | None:
        if self.close_time is None:
            return None
        now = now or utc_now()
        return (self.close_time - now).total_seconds() / 60.0


@dataclass(frozen=True)
class KalshiTickerUpdate:
    ticker: str
    event_time: datetime
    last_yes_price_cents: int | None
    previous_yes_price_cents: int | None
    close_time: datetime | None
    is_open: bool
    market_prob: float | None
    previous_market_prob: float | None
    price_momentum: float | None
    tau_minutes: float | None
    open_time: datetime | None = None
    trade_id: str | None = None
    count: int | None = None
    taker_side: str | None = None
    last_trade_time: datetime | None = None
    last_price_cents: int | None = None
    yes_bid_cents: int | None = None
    yes_ask_cents: int | None = None
    no_bid_cents: int | None = None
    no_ask_cents: int | None = None
    yes_bid_size: int | None = None
    yes_ask_size: int | None = None
    no_bid_size: int | None = None
    no_ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    dollar_volume: int | None = None
    dollar_open_interest: int | None = None
    ticker_update_time: datetime | None = None
    source: str = "snapshot"
    received_at: datetime | None = None
    event_id: str = ""
    raw_event_id: str | None = None


@dataclass(frozen=True)
class KalshiRawStreamEvent:
    channel: str
    market_ticker: str | None
    exchange_event_time: datetime | None
    received_at: datetime
    session_id: str
    message_index: int
    payload: dict
    raw_event_id: str = ""


def state_to_update(
    state: KalshiTickerState,
    event_time: datetime,
    now: datetime | None = None,
    source: str = "snapshot",
) -> KalshiTickerUpdate:
    return KalshiTickerUpdate(
        ticker=state.ticker,
        event_time=event_time,
        last_yes_price_cents=state.last_yes_price_cents,
        previous_yes_price_cents=state.previous_yes_price_cents,
        close_time=state.close_time,
        open_time=state.open_time,
        is_open=state.is_open,
        market_prob=state.market_prob,
        previous_market_prob=state.previous_market_prob,
        price_momentum=state.price_momentum,
        tau_minutes=state.tau_minutes(now=now),
        trade_id=state.trade_id,
        count=state.count,
        taker_side=state.taker_side,
        last_trade_time=state.last_trade_time,
        last_price_cents=state.last_price_cents,
        yes_bid_cents=state.yes_bid_cents,
        yes_ask_cents=state.yes_ask_cents,
        no_bid_cents=state.no_bid_cents,
        no_ask_cents=state.no_ask_cents,
        yes_bid_size=state.yes_bid_size,
        yes_ask_size=state.yes_ask_size,
        no_bid_size=state.no_bid_size,
        no_ask_size=state.no_ask_size,
        volume=state.volume,
        open_interest=state.open_interest,
        dollar_volume=state.dollar_volume,
        dollar_open_interest=state.dollar_open_interest,
        ticker_update_time=state.ticker_update_time,
        source=source,
        received_at=state.received_at,
        event_id=state.event_id,
        raw_event_id=state.raw_event_id,
    )
