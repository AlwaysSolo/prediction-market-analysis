from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect

from src.indexers.kalshi.models import Market, parse_count, parse_price_cents
from src.live.kalshi.auth import build_auth_headers
from src.live.kalshi.client import KalshiLiveRestClient
from src.live.kalshi.config import KalshiCollectorConfig, KalshiEnvironment
from src.live.kalshi.types import (
    KalshiRawStreamEvent,
    KalshiTickerState,
    KalshiTickerUpdate,
    state_to_update,
)

Callback = Callable[[KalshiTickerUpdate], Awaitable[None] | None]
RawCallback = Callable[[KalshiRawStreamEvent], Awaitable[None] | None]


def _is_market_open(status: str | None) -> bool:
    return status in {"active", "open"}


def _has_final_result(market: Market | None) -> bool:
    if market is None:
        return False
    return market.result.strip().upper() in {"YES", "NO"}


def _merge_quote_side(
    *,
    direct_bid_cents: int | None,
    direct_ask_cents: int | None,
    opposite_direct_bid_cents: int | None,
    opposite_direct_ask_cents: int | None,
    fallback_bid_cents: int | None,
    fallback_ask_cents: int | None,
) -> tuple[int | None, int | None]:
    bid_cents = direct_bid_cents
    if bid_cents is None and opposite_direct_ask_cents is not None:
        bid_cents = 100 - opposite_direct_ask_cents
    if bid_cents is None:
        bid_cents = fallback_bid_cents

    ask_cents = direct_ask_cents
    if ask_cents is None and opposite_direct_bid_cents is not None:
        ask_cents = 100 - opposite_direct_bid_cents
    if ask_cents is None:
        ask_cents = fallback_ask_cents

    return bid_cents, ask_cents


def _merge_binary_quote_fields(
    *,
    source_yes_bid_cents: int | None,
    source_yes_ask_cents: int | None,
    source_no_bid_cents: int | None,
    source_no_ask_cents: int | None,
    fallback_yes_bid_cents: int | None,
    fallback_yes_ask_cents: int | None,
    fallback_no_bid_cents: int | None,
    fallback_no_ask_cents: int | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    yes_bid_cents, yes_ask_cents = _merge_quote_side(
        direct_bid_cents=source_yes_bid_cents,
        direct_ask_cents=source_yes_ask_cents,
        opposite_direct_bid_cents=source_no_bid_cents,
        opposite_direct_ask_cents=source_no_ask_cents,
        fallback_bid_cents=fallback_yes_bid_cents,
        fallback_ask_cents=fallback_yes_ask_cents,
    )
    no_bid_cents, no_ask_cents = _merge_quote_side(
        direct_bid_cents=source_no_bid_cents,
        direct_ask_cents=source_no_ask_cents,
        opposite_direct_bid_cents=source_yes_bid_cents,
        opposite_direct_ask_cents=source_yes_ask_cents,
        fallback_bid_cents=fallback_no_bid_cents,
        fallback_ask_cents=fallback_no_ask_cents,
    )
    return yes_bid_cents, yes_ask_cents, no_bid_cents, no_ask_cents


class JsonlEventLogger:
    def __init__(self, base_dir: Path, environment: str):
        self.base_dir = base_dir
        self.environment = environment
        self._lock = asyncio.Lock()

    async def write(self, event_type: str, payload: dict[str, Any], event_time: datetime | None = None) -> None:
        event_time = event_time or datetime.now(UTC)
        date_dir = self.base_dir / self.environment / event_time.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / "events.jsonl"
        row = {
            "logged_at": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        async with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")


class KalshiMarketDataCollector:
    def __init__(self, config: KalshiCollectorConfig):
        self.config = config
        self._rest_client = KalshiLiveRestClient(config)
        self._logger = JsonlEventLogger(config.log_dir, config.environment.value)
        self._states: dict[str, KalshiTickerState] = {}
        self._markets: dict[str, Market] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiTickerUpdate]] = []
        self._raw_callbacks: list[RawCallback] = []
        self._raw_queues: list[asyncio.Queue[KalshiRawStreamEvent]] = []
        self._tasks: list[asyncio.Task[Any]] = []
        self._active_websocket: Any | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._subscribed_tickers: tuple[str, ...] = ()
        self._message_id = 1
        self._stream_session_id = ""
        self._stream_message_index = 0

    async def _log_stream_message(self, event_type: str, payload: dict[str, Any], event_time: datetime | None = None) -> None:
        if not self.config.log_stream_messages:
            return
        await self._logger.write(event_type, payload, event_time)

    async def start(self) -> None:
        if self._tasks:
            return

        await self._logger.write("collector_started", {"environment": self.config.environment.value})
        await self.refresh_market_metadata()
        self._tasks = [
            asyncio.create_task(self._stream_loop(), name="kalshi-stream-loop"),
            asyncio.create_task(self._metadata_refresh_loop(), name="kalshi-metadata-refresh"),
        ]
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self._rest_client.close()
        await self._logger.write("collector_stopped", {"environment": self.config.environment.value})

    def get_state(self, ticker: str) -> KalshiTickerState | None:
        return self._states.get(ticker)

    def get_market(self, ticker: str) -> Market | None:
        return self._markets.get(ticker)

    async def refresh_market_snapshot(self, ticker: str) -> Market | None:
        try:
            market = await asyncio.to_thread(self._rest_client.get_market, ticker)
        except Exception as exc:
            await self._logger.write(
                "rest_market_snapshot_failed",
                {"ticker": ticker, "error": repr(exc)},
            )
            return self._markets.get(ticker)

        if (
            self.config.environment is KalshiEnvironment.DEMO
            and not _has_final_result(market)
            and market.status in {"closed", "determined", "finalized", "settled"}
        ):
            try:
                public_market = await asyncio.to_thread(
                    self._rest_client.get_public_market,
                    ticker,
                    KalshiEnvironment.PRODUCTION,
                )
            except Exception as exc:
                await self._logger.write(
                    "rest_market_snapshot_public_fallback_failed",
                    {"ticker": ticker, "error": repr(exc)},
                )
            else:
                if _has_final_result(public_market):
                    market = public_market
                    await self._logger.write(
                        "rest_market_snapshot_public_fallback",
                        {
                            "ticker": ticker,
                            "status": market.status,
                            "result": market.result,
                            "close_time": market.close_time.isoformat() if market.close_time else None,
                        },
                    )

        await self._logger.write(
            "rest_market_snapshot",
            {
                "ticker": ticker,
                "status": market.status,
                "result": market.result,
                "close_time": market.close_time.isoformat() if market.close_time else None,
            },
        )

        now = datetime.now(UTC)
        current = self._states.get(ticker)
        yes_bid_cents, yes_ask_cents, no_bid_cents, no_ask_cents = _merge_binary_quote_fields(
            source_yes_bid_cents=market.yes_bid,
            source_yes_ask_cents=market.yes_ask,
            source_no_bid_cents=market.no_bid,
            source_no_ask_cents=market.no_ask,
            fallback_yes_bid_cents=(current.yes_bid_cents if current else None),
            fallback_yes_ask_cents=(current.yes_ask_cents if current else None),
            fallback_no_bid_cents=(current.no_bid_cents if current else None),
            fallback_no_ask_cents=(current.no_ask_cents if current else None),
        )
        updated_state = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=(
                current.last_yes_price_cents
                if current and current.last_yes_price_cents is not None
                else market.last_price
            ),
            last_trade_time=current.last_trade_time if current else None,
            previous_yes_price_cents=current.previous_yes_price_cents if current else None,
            close_time=market.close_time or (current.close_time if current else None),
            open_time=market.open_time or (current.open_time if current else None),
            is_open=_is_market_open(market.status),
            trade_id=current.trade_id if current else None,
            count=current.count if current else None,
            taker_side=current.taker_side if current else None,
            last_price_cents=market.last_price if market.last_price is not None else (current.last_price_cents if current else None),
            yes_bid_cents=yes_bid_cents,
            yes_ask_cents=yes_ask_cents,
            no_bid_cents=no_bid_cents,
            no_ask_cents=no_ask_cents,
            volume=market.volume if market.volume or market.volume == 0 else (current.volume if current else None),
            open_interest=(
                market.open_interest
                if market.open_interest or market.open_interest == 0
                else (current.open_interest if current else None)
            ),
            dollar_volume=current.dollar_volume if current else None,
            dollar_open_interest=current.dollar_open_interest if current else None,
            ticker_update_time=current.ticker_update_time if current else None,
            received_at=now,
            event_id=self._next_event_id("market_snapshot"),
            raw_event_id=None,
        )
        self._markets[ticker] = market
        if updated_state != current:
            self._states[ticker] = updated_state
            await self._publish_update(state_to_update(updated_state, now, now=now, source="market_snapshot"))
        return market

    def snapshot_states(self) -> dict[str, KalshiTickerState]:
        return dict(self._states)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiTickerUpdate]:
        queue: asyncio.Queue[KalshiTickerUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def subscribe_raw_stream(self, callback: RawCallback) -> None:
        self._raw_callbacks.append(callback)

    def subscribe_raw_stream_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiRawStreamEvent]:
        queue: asyncio.Queue[KalshiRawStreamEvent] = asyncio.Queue(maxsize=maxsize)
        self._raw_queues.append(queue)
        return queue

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def _next_event_id(self, prefix: str) -> str:
        return f"{prefix}:{uuid.uuid4().hex}"

    async def _publish_raw_stream_event(self, event: KalshiRawStreamEvent) -> None:
        for callback in self._raw_callbacks:
            result = callback(event)
            if inspect.isawaitable(result):
                await result
        for queue in self._raw_queues:
            await queue.put(event)

    def _raw_stream_event_time(self, channel: str, msg: dict[str, Any]) -> datetime | None:
        if channel == "trade":
            ts_value = msg.get("ts")
            if ts_value is None:
                return None
            return datetime.fromtimestamp(int(ts_value), tz=UTC)
        if channel == "ticker":
            if msg.get("time"):
                return datetime.fromisoformat(str(msg["time"]).replace("Z", "+00:00"))
            if msg.get("ts") is not None:
                return datetime.fromtimestamp(int(msg["ts"]), tz=UTC)
            return None
        if channel == "market_lifecycle_v2":
            if msg.get("ts") is not None:
                return datetime.fromtimestamp(int(msg["ts"]), tz=UTC)
            if msg.get("time"):
                return datetime.fromisoformat(str(msg["time"]).replace("Z", "+00:00"))
            if msg.get("close_ts") is not None:
                return datetime.fromtimestamp(int(msg["close_ts"]), tz=UTC)
            return None
        return None

    async def refresh_market_metadata(self) -> None:
        await self._logger.write("metadata_refresh_started", {})
        previous_subscribed_tickers = self._subscribed_tickers
        filtered_markets: dict[str, Market] = {}
        metadata_updates: list[KalshiTickerUpdate] = []
        refresh_time = datetime.now(UTC)
        for payload, markets in await self._fetch_market_pages():
            await self._logger.write(
                "rest_markets_page",
                {
                    "cursor": payload.get("cursor"),
                    "market_count": len(markets),
                    "first_ticker": markets[0].ticker if markets else None,
                    "last_ticker": markets[-1].ticker if markets else None,
                },
            )
            for market in markets:
                if self._market_matches_filters(market):
                    filtered_markets[market.ticker] = market

        self._markets = filtered_markets
        self._subscribed_tickers = tuple(sorted(filtered_markets))
        for market in filtered_markets.values():
            existing = self._states.get(market.ticker)
            yes_bid_cents, yes_ask_cents, no_bid_cents, no_ask_cents = _merge_binary_quote_fields(
                source_yes_bid_cents=(existing.yes_bid_cents if existing and existing.yes_bid_cents is not None else market.yes_bid),
                source_yes_ask_cents=(existing.yes_ask_cents if existing and existing.yes_ask_cents is not None else market.yes_ask),
                source_no_bid_cents=(existing.no_bid_cents if existing and existing.no_bid_cents is not None else market.no_bid),
                source_no_ask_cents=(existing.no_ask_cents if existing and existing.no_ask_cents is not None else market.no_ask),
                fallback_yes_bid_cents=None,
                fallback_yes_ask_cents=None,
                fallback_no_bid_cents=None,
                fallback_no_ask_cents=None,
            )
            updated = KalshiTickerState(
                ticker=market.ticker,
                last_yes_price_cents=existing.last_yes_price_cents if existing else None,
                last_trade_time=existing.last_trade_time if existing else None,
                previous_yes_price_cents=existing.previous_yes_price_cents if existing else None,
                close_time=market.close_time,
                open_time=market.open_time,
                is_open=_is_market_open(market.status),
                trade_id=existing.trade_id if existing else None,
                count=existing.count if existing else None,
                taker_side=existing.taker_side if existing else None,
                last_price_cents=existing.last_price_cents if existing and existing.last_price_cents is not None else market.last_price,
                yes_bid_cents=yes_bid_cents,
                yes_ask_cents=yes_ask_cents,
                no_bid_cents=no_bid_cents,
                no_ask_cents=no_ask_cents,
                volume=existing.volume if existing and existing.volume is not None else market.volume,
                open_interest=(
                    existing.open_interest if existing and existing.open_interest is not None else market.open_interest
                ),
                dollar_volume=existing.dollar_volume if existing else None,
                dollar_open_interest=existing.dollar_open_interest if existing else None,
                ticker_update_time=existing.ticker_update_time if existing else None,
                received_at=refresh_time,
                event_id=self._next_event_id("metadata_refresh"),
                raw_event_id=None,
            )
            self._states[market.ticker] = updated
            if updated != existing:
                metadata_updates.append(
                    state_to_update(
                        updated,
                        refresh_time,
                        now=refresh_time,
                        source="metadata_refresh",
                    )
                )
        await self._logger.write(
            "metadata_refresh_completed",
            {"market_count": len(self._subscribed_tickers)},
        )
        for update in metadata_updates:
            await self._publish_update(update)
        if self._should_resubscribe_trade_stream(previous_subscribed_tickers):
            await self._logger.write(
                "ws_resubscribe_requested",
                {
                    "previous_ticker_count": len(previous_subscribed_tickers),
                    "new_ticker_count": len(self._subscribed_tickers),
                },
            )
            if self._active_websocket is not None:
                await self._active_websocket.close(code=1000, reason="subscription_refresh")

    async def _fetch_market_pages(self) -> list[tuple[dict[str, Any], list[Market]]]:
        get_markets_page = getattr(self._rest_client, "get_markets_page", None)
        page_limit = 1000
        if callable(get_markets_page):
            pages: list[tuple[dict[str, Any], list[Market]]] = []
            filter_sets = self._market_page_filters()
            if not filter_sets:
                filter_sets = [{"limit": page_limit, "status": "open"}]

            for filter_kwargs in filter_sets:
                cursor: str | None = None
                while True:
                    payload, markets = await asyncio.to_thread(
                        get_markets_page,
                        cursor=cursor,
                        **filter_kwargs,
                    )
                    pages.append((payload, markets))
                    cursor = payload.get("cursor")
                    if not cursor:
                        break
            return pages

        return await asyncio.to_thread(
            lambda: list(
                self._rest_client.iter_markets(
                    limit=page_limit,
                    status="open",
                )
            )
        )

    def _market_page_filters(self) -> list[dict[str, Any]]:
        configured_tickers = self.config.normalized_market_tickers()
        if configured_tickers:
            return [
                {
                    "limit": 1000,
                    "status": "open",
                    "tickers": configured_tickers,
                }
            ]

        configured_series = self.config.normalized_series_tickers()
        if configured_series:
            return [
                {
                    "limit": 1000,
                    "status": "open",
                    "series_ticker": series_ticker,
                }
                for series_ticker in configured_series
            ]

        return []

    def _market_matches_filters(self, market: Market) -> bool:
        configured_tickers = self.config.normalized_market_tickers()
        if configured_tickers:
            return market.ticker in configured_tickers

        configured_series = self.config.normalized_series_tickers()
        if configured_series:
            return any(
                market.ticker.startswith(f"{series}-") or market.event_ticker.startswith(series)
                for series in configured_series
            )

        return True

    async def _metadata_refresh_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.config.metadata_refresh_interval_seconds)
                if self._stop_event.is_set():
                    break
                await self.refresh_market_metadata()
        except asyncio.CancelledError:
            raise

    async def _stream_loop(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            try:
                await self._connect_and_consume()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                await self._logger.write(
                    "ws_disconnected",
                    {"attempt": attempt, "error": repr(exc)},
                )
                delay = min(
                    self.config.reconnect.initial_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.reconnect.max_backoff_seconds,
                )
                await self._logger.write("ws_reconnect_scheduled", {"delay_seconds": delay, "attempt": attempt})
                await asyncio.sleep(delay)
                await self.refresh_market_metadata()

    async def _connect_and_consume(self) -> None:
        path = "/trade-api/ws/v2"
        timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
        headers = build_auth_headers(self.config.credentials, "GET", path, timestamp_ms)
        await self._logger.write(
            "ws_connecting",
            {
                "url": self.config.environment.websocket_url,
                "ticker_count": len(self._subscribed_tickers),
            },
        )
        async with connect(
            self.config.environment.websocket_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            self._active_websocket = websocket
            self._stream_session_id = uuid.uuid4().hex
            self._stream_message_index = 0
            await self._logger.write("ws_connected", {"ticker_count": len(self._subscribed_tickers)})
            await self._subscribe_channels(websocket)
            try:
                async for message in websocket:
                    payload = json.loads(message)
                    await self._handle_message(payload)
            finally:
                self._active_websocket = None

    async def _subscribe_channels(self, websocket: Any) -> None:
        for message in self._subscription_messages():
            await websocket.send(json.dumps(message))
            await self._logger.write("ws_subscribed", message)

    def _subscription_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        trade_params: dict[str, Any] = {"channels": ["trade"]}
        if self._subscribed_tickers:
            trade_params["market_tickers"] = list(self._subscribed_tickers)
        messages.append({"id": self._message_id, "cmd": "subscribe", "params": trade_params})
        self._message_id += 1

        ticker_params: dict[str, Any] = {"channels": ["ticker"]}
        if self._subscribed_tickers:
            ticker_params["market_tickers"] = list(self._subscribed_tickers)
        messages.append({"id": self._message_id, "cmd": "subscribe", "params": ticker_params})
        self._message_id += 1

        # Kalshi's lifecycle feed is all-markets only; ticker filters are not supported.
        messages.append(
            {
                "id": self._message_id,
                "cmd": "subscribe",
                "params": {"channels": ["market_lifecycle_v2"]},
            }
        )
        self._message_id += 1
        return messages

    async def _handle_message(self, payload: dict[str, Any]) -> None:
        message_type = payload.get("type")
        msg = payload.get("msg", {})
        received_at = datetime.now(UTC)
        raw_event_id: str | None = None
        if message_type in {"trade", "ticker", "market_lifecycle_v2"} and isinstance(msg, dict):
            self._stream_message_index += 1
            raw_event_id = f"{self._stream_session_id}:{self._stream_message_index}"
            await self._publish_raw_stream_event(
                KalshiRawStreamEvent(
                    channel=message_type,
                    market_ticker=msg.get("market_ticker"),
                    exchange_event_time=self._raw_stream_event_time(message_type, msg),
                    received_at=received_at,
                    session_id=self._stream_session_id,
                    message_index=self._stream_message_index,
                    payload=payload,
                    raw_event_id=raw_event_id,
                )
            )
        if message_type == "trade":
            await self._handle_trade_message(payload["msg"], received_at=received_at, raw_event_id=raw_event_id)
        elif message_type == "ticker":
            await self._handle_ticker_message(payload["msg"], received_at=received_at, raw_event_id=raw_event_id)
        elif message_type == "market_lifecycle_v2":
            await self._handle_market_lifecycle_message(payload["msg"], received_at=received_at, raw_event_id=raw_event_id)
        elif message_type in {"subscribed", "ok", "error"}:
            return

    async def _handle_trade_message(
        self,
        msg: dict[str, Any],
        *,
        received_at: datetime | None = None,
        raw_event_id: str | None = None,
    ) -> None:
        ticker = msg["market_ticker"]
        event_time = datetime.fromtimestamp(int(msg["ts"]), tz=UTC)
        received_at = received_at or datetime.now(UTC)
        await self._log_stream_message("ws_trade_message", msg, event_time)
        yes_price_cents = parse_price_cents(msg.get("yes_price", msg.get("yes_price_dollars")))
        count = parse_count(msg.get("count"), msg.get("count_fp"))
        if yes_price_cents is None:
            return
        if self._is_stale_event_time(event_time):
            return
        current = self._states.get(ticker)
        market = self._markets.get(ticker)
        previous_yes = current.last_yes_price_cents if current else None
        yes_bid_cents, yes_ask_cents, no_bid_cents, no_ask_cents = _merge_binary_quote_fields(
            source_yes_bid_cents=None,
            source_yes_ask_cents=None,
            source_no_bid_cents=None,
            source_no_ask_cents=None,
            fallback_yes_bid_cents=(current.yes_bid_cents if current else (market.yes_bid if market else None)),
            fallback_yes_ask_cents=(current.yes_ask_cents if current else (market.yes_ask if market else None)),
            fallback_no_bid_cents=(current.no_bid_cents if current else (market.no_bid if market else None)),
            fallback_no_ask_cents=(current.no_ask_cents if current else (market.no_ask if market else None)),
        )
        updated_state = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=yes_price_cents,
            last_trade_time=event_time,
            previous_yes_price_cents=previous_yes,
            close_time=(current.close_time if current and current.close_time else (market.close_time if market else None)),
            open_time=(current.open_time if current and current.open_time else (market.open_time if market else None)),
            is_open=(current.is_open if current else (_is_market_open(market.status) if market else True)),
            trade_id=msg.get("trade_id"),
            count=count,
            taker_side=msg.get("taker_side"),
            last_price_cents=yes_price_cents,
            yes_bid_cents=yes_bid_cents,
            yes_ask_cents=yes_ask_cents,
            no_bid_cents=no_bid_cents,
            no_ask_cents=no_ask_cents,
            volume=current.volume if current else (market.volume if market else None),
            open_interest=current.open_interest if current else (market.open_interest if market else None),
            dollar_volume=current.dollar_volume if current else None,
            dollar_open_interest=current.dollar_open_interest if current else None,
            ticker_update_time=current.ticker_update_time if current else None,
            received_at=received_at,
            event_id=self._next_event_id("trade"),
            raw_event_id=raw_event_id,
        )
        if updated_state == current:
            return
        self._states[ticker] = updated_state
        await self._publish_update(state_to_update(updated_state, event_time, now=event_time, source="trade"))

    async def _handle_ticker_message(
        self,
        msg: dict[str, Any],
        *,
        received_at: datetime | None = None,
        raw_event_id: str | None = None,
    ) -> None:
        ticker = msg["market_ticker"]
        received_at = received_at or datetime.now(UTC)
        current = self._states.get(ticker)
        market = self._markets.get(ticker)
        last_price_cents = parse_price_cents(msg.get("price", msg.get("price_dollars")))
        yes_bid_cents = parse_price_cents(msg.get("yes_bid", msg.get("yes_bid_dollars")))
        yes_ask_cents = parse_price_cents(msg.get("yes_ask", msg.get("yes_ask_dollars")))
        no_bid_cents = parse_price_cents(msg.get("no_bid", msg.get("no_bid_dollars")))
        no_ask_cents = parse_price_cents(msg.get("no_ask", msg.get("no_ask_dollars")))
        volume = parse_count(msg.get("volume"), msg.get("volume_fp"))
        open_interest = parse_count(msg.get("open_interest"), msg.get("open_interest_fp"))
        dollar_volume_raw = msg.get("dollar_volume")
        dollar_open_interest_raw = msg.get("dollar_open_interest")
        event_time = (
            datetime.fromisoformat(msg["time"].replace("Z", "+00:00"))
            if msg.get("time")
            else datetime.fromtimestamp(int(msg["ts"]), tz=UTC)
        )
        await self._log_stream_message("ws_ticker_message", msg, event_time)
        if self._is_stale_event_time(event_time):
            return

        preserved_trade_price = current.last_yes_price_cents if current else None
        if preserved_trade_price is None and last_price_cents is not None:
            preserved_trade_price = last_price_cents

        resolved_yes_bid_cents, resolved_yes_ask_cents, resolved_no_bid_cents, resolved_no_ask_cents = _merge_binary_quote_fields(
            source_yes_bid_cents=yes_bid_cents,
            source_yes_ask_cents=yes_ask_cents,
            source_no_bid_cents=no_bid_cents,
            source_no_ask_cents=no_ask_cents,
            fallback_yes_bid_cents=(current.yes_bid_cents if current else None),
            fallback_yes_ask_cents=(current.yes_ask_cents if current else None),
            fallback_no_bid_cents=(current.no_bid_cents if current else (market.no_bid if market else None)),
            fallback_no_ask_cents=(current.no_ask_cents if current else (market.no_ask if market else None)),
        )
        updated_state = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=preserved_trade_price,
            last_trade_time=current.last_trade_time if current else None,
            previous_yes_price_cents=current.previous_yes_price_cents if current else None,
            close_time=(current.close_time if current and current.close_time else (market.close_time if market else None)),
            open_time=(current.open_time if current and current.open_time else (market.open_time if market else None)),
            is_open=(current.is_open if current else (_is_market_open(market.status) if market else True)),
            trade_id=current.trade_id if current else None,
            count=current.count if current else None,
            taker_side=current.taker_side if current else None,
            last_price_cents=last_price_cents if last_price_cents is not None else (current.last_price_cents if current else None),
            yes_bid_cents=resolved_yes_bid_cents,
            yes_ask_cents=resolved_yes_ask_cents,
            no_bid_cents=resolved_no_bid_cents,
            no_ask_cents=resolved_no_ask_cents,
            volume=volume if volume or volume == 0 else (current.volume if current else (market.volume if market else None)),
            open_interest=(
                open_interest
                if open_interest or open_interest == 0
                else (current.open_interest if current else (market.open_interest if market else None))
            ),
            dollar_volume=int(dollar_volume_raw) if dollar_volume_raw is not None else (current.dollar_volume if current else None),
            dollar_open_interest=(
                int(dollar_open_interest_raw)
                if dollar_open_interest_raw is not None
                else (current.dollar_open_interest if current else None)
            ),
            ticker_update_time=event_time,
            received_at=received_at,
            event_id=self._next_event_id("ticker"),
            raw_event_id=raw_event_id,
        )
        if updated_state == current:
            return
        self._states[ticker] = updated_state
        await self._publish_update(state_to_update(updated_state, event_time, now=event_time, source="ticker"))

    async def _handle_market_lifecycle_message(
        self,
        msg: dict[str, Any],
        *,
        received_at: datetime | None = None,
        raw_event_id: str | None = None,
    ) -> None:
        ticker = msg["market_ticker"]
        received_at = received_at or datetime.now(UTC)
        current = self._states.get(ticker)
        market = self._markets.get(ticker)
        close_ts = msg.get("close_ts") or msg.get("expected_expiration_ts")
        existing_close_time = current.close_time if current else None
        close_time = datetime.fromtimestamp(int(close_ts), tz=UTC) if close_ts is not None else existing_close_time
        event_time = close_time or received_at
        await self._log_stream_message("ws_market_lifecycle_message", msg, event_time)
        event_type = msg.get("event_type", "")
        if event_type in {"deactivated", "determined", "settled"}:
            is_open = False
        elif event_type in {"created", "activated", "close_date_updated"}:
            is_open = True
        elif current:
            is_open = current.is_open
        else:
            is_open = True

        updated_state = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=current.last_yes_price_cents if current else None,
            last_trade_time=current.last_trade_time if current else None,
            previous_yes_price_cents=current.previous_yes_price_cents if current else None,
            close_time=close_time,
            open_time=current.open_time if current else (market.open_time if market else None),
            is_open=is_open,
            trade_id=current.trade_id if current else None,
            count=current.count if current else None,
            taker_side=current.taker_side if current else None,
            last_price_cents=current.last_price_cents if current else None,
            yes_bid_cents=current.yes_bid_cents if current else None,
            yes_ask_cents=current.yes_ask_cents if current else None,
            no_bid_cents=current.no_bid_cents if current else (market.no_bid if market else None),
            no_ask_cents=current.no_ask_cents if current else (market.no_ask if market else None),
            volume=current.volume if current else (market.volume if market else None),
            open_interest=current.open_interest if current else (market.open_interest if market else None),
            dollar_volume=current.dollar_volume if current else None,
            dollar_open_interest=current.dollar_open_interest if current else None,
            ticker_update_time=current.ticker_update_time if current else None,
            received_at=received_at,
            event_id=self._next_event_id("market_lifecycle"),
            raw_event_id=raw_event_id,
        )
        if market:
            self._markets[ticker] = Market(
                ticker=market.ticker,
                event_ticker=market.event_ticker,
                market_type=market.market_type,
                title=market.title,
                yes_sub_title=market.yes_sub_title,
                no_sub_title=market.no_sub_title,
                status="open" if updated_state.is_open else event_type or market.status,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                no_bid=market.no_bid,
                no_ask=market.no_ask,
                last_price=market.last_price,
                volume=market.volume,
                volume_24h=market.volume_24h,
                open_interest=market.open_interest,
                result=str(msg.get("result") or market.result or ""),
                created_time=market.created_time,
                open_time=market.open_time,
                close_time=updated_state.close_time,
            )
        if updated_state == current:
            return
        self._states[ticker] = updated_state
        await self._publish_update(state_to_update(updated_state, event_time, now=event_time, source="market_lifecycle"))

    async def _publish_update(self, update: KalshiTickerUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)

    def _should_resubscribe_trade_stream(self, previous_subscribed_tickers: tuple[str, ...]) -> bool:
        if not self._tasks:
            return False
        if previous_subscribed_tickers == self._subscribed_tickers:
            return False
        return True

    def _is_stale_event_time(self, event_time: datetime) -> bool:
        max_age_seconds = self.config.max_stale_message_age_seconds
        if max_age_seconds <= 0:
            return False
        return (datetime.now(UTC) - event_time).total_seconds() > max_age_seconds
