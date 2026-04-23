from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect

Callback = Callable[["BTCSpotUpdate"], Awaitable[None] | None]

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"


def _parse_exchange_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _JsonlLogger:
    def __init__(self, base_dir: Path, environment: str):
        self.base_dir = base_dir
        self.environment = environment
        self._queue: asyncio.Queue[object] | None = None
        self._writer_task: asyncio.Task[None] | None = None

    def _ensure_writer(self) -> None:
        if self._writer_task is not None and not self._writer_task.done():
            return
        self._queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._writer_loop(), name=f"btc-spot-jsonl-{self.environment}")

    async def write(self, event_type: str, payload: dict[str, Any], event_time: datetime | None = None) -> None:
        event_time = event_time or _utc_now()
        path = self.base_dir / self.environment / event_time.strftime("%Y-%m-%d") / "events.jsonl"
        row = {"logged_at": _utc_now().isoformat(), "event_type": event_type, "payload": payload}
        self._ensure_writer()
        assert self._queue is not None
        await self._queue.put((path, row))

    async def close(self) -> None:
        if self._queue is None or self._writer_task is None:
            return
        await self._queue.join()
        await self._queue.put(None)
        await self._writer_task
        self._queue = None
        self._writer_task = None

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                path, row = item
                assert isinstance(path, Path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, default=str) + "\n")
            finally:
                self._queue.task_done()


@dataclass(frozen=True)
class BTCSpotFeedConfig:
    environment: str = "production"
    log_dir: Path = Path("output") / "live" / "external_spot"
    venues: tuple[str, ...] = ("coinbase", "kraken")
    max_age_ms: int = 500
    venue_divergence_bps_gate: float = 25.0
    hard_fail_seconds: float = 5.0
    twap_window_seconds: int = 60
    history_window_seconds: int = 31 * 60
    clock_drift_warn_ms: float = 200.0
    archive_raw: bool = True
    archive_normalized: bool = True
    coinbase_url: str = COINBASE_WS_URL
    kraken_url: str = KRAKEN_WS_URL


@dataclass(frozen=True)
class BTCSpotVenueQuote:
    venue: str
    symbol: str
    best_bid: float
    best_ask: float
    best_bid_size: float
    best_ask_size: float
    exchange_event_time: datetime
    received_at: datetime
    sequence: int | None = None
    is_resyncing: bool = False

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return max(self.best_ask - self.best_bid, 0.0)


@dataclass(frozen=True)
class BTCSpotUpdate:
    event_time: datetime
    received_at: datetime
    btc_spot_price: float | None
    btc_spot_twap_60s: float | None
    btc_spot_age_ms: float
    btc_spot_is_fresh: bool
    btc_spot_venues_fresh: int
    btc_spot_venue_divergence_bps: float
    btc_vol_effective_sample_size: float
    btc_spot_source: str
    btc_spot_return_30s: float | None
    btc_spot_return_120s: float | None
    btc_spot_return_300s: float | None
    btc_spot_return_900s: float | None
    btc_spot_vol_120s: float | None
    btc_spot_vol_300s: float | None
    btc_spot_vol_900s: float | None
    btc_spot_vol_1800s: float | None
    btc_spot_vol_ewma_hl300: float | None


def parse_coinbase_ticker_message(payload: dict[str, Any], *, received_at: datetime | None = None) -> BTCSpotVenueQuote | None:
    if payload.get("channel") != "ticker":
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        tickers = event.get("tickers")
        if not isinstance(tickers, list):
            continue
        for ticker in tickers:
            if str(ticker.get("product_id")) != "BTC-USD":
                continue
            best_bid = ticker.get("best_bid")
            best_ask = ticker.get("best_ask")
            if best_bid is None or best_ask is None:
                continue
            event_time = _parse_exchange_timestamp(str(payload.get("timestamp")))
            if event_time is None:
                continue
            return BTCSpotVenueQuote(
                venue="coinbase",
                symbol="BTC-USD",
                best_bid=float(best_bid),
                best_ask=float(best_ask),
                best_bid_size=float(ticker.get("best_bid_quantity", 0.0) or 0.0),
                best_ask_size=float(ticker.get("best_ask_quantity", 0.0) or 0.0),
                exchange_event_time=event_time,
                received_at=received_at or _utc_now(),
                sequence=int(payload["sequence_num"]) if payload.get("sequence_num") is not None else None,
            )
    return None


def parse_kraken_book_message(payload: dict[str, Any], *, received_at: datetime | None = None) -> BTCSpotVenueQuote | None:
    if payload.get("channel") != "book":
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    book = data[0]
    if not isinstance(book, dict) or str(book.get("symbol")) != "BTC/USD":
        return None
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return None
    best_bid = bids[0]
    best_ask = asks[0]
    event_time = _parse_exchange_timestamp(book.get("timestamp"))
    if event_time is None:
        return None
    return BTCSpotVenueQuote(
        venue="kraken",
        symbol="BTC/USD",
        best_bid=float(best_bid["price"]),
        best_ask=float(best_ask["price"]),
        best_bid_size=float(best_bid.get("qty", 0.0) or 0.0),
        best_ask_size=float(best_ask.get("qty", 0.0) or 0.0),
        exchange_event_time=event_time,
        received_at=received_at or _utc_now(),
        sequence=None,
    )


class BTCSpotFeed:
    def __init__(self, config: BTCSpotFeedConfig | None = None):
        self.config = config or BTCSpotFeedConfig()
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[BTCSpotUpdate]] = []
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._venue_quotes: dict[str, BTCSpotVenueQuote] = {}
        self._updates: deque[BTCSpotUpdate] = deque()
        self._resampled_history: deque[tuple[datetime, float]] = deque()
        self._last_resampled_second: datetime | None = None
        self._latest_update: BTCSpotUpdate | None = None
        self._raw_logger = _JsonlLogger(self.config.log_dir / "external_spot_raw", self.config.environment)
        self._normalized_logger = _JsonlLogger(self.config.log_dir / "external_spot_normalized", self.config.environment)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[BTCSpotUpdate]:
        queue: asyncio.Queue[BTCSpotUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def snapshot_state(self) -> BTCSpotUpdate | None:
        return self._latest_update

    def lookup_at_or_before(self, event_time: datetime, *, max_age_ms: int | None = None) -> BTCSpotUpdate | None:
        limit_ms = self.config.max_age_ms if max_age_ms is None else int(max_age_ms)
        for update in reversed(self._updates):
            if update.event_time <= event_time:
                age_ms = max(0.0, (event_time - update.event_time).total_seconds() * 1000.0)
                if age_ms <= limit_ms and update.btc_spot_is_fresh:
                    return update
                return None
        return None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        self._task = asyncio.create_task(self._run(), name="btc-spot-feed")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._raw_logger.close()
        await self._normalized_logger.close()

    async def wait_until_ready(self) -> None:
        if self._ready_event.is_set():
            return
        if self._task is None:
            raise RuntimeError("BTC spot feed has not been started.")
        ready_wait = asyncio.create_task(self._ready_event.wait(), name="btc-spot-feed-ready-wait")
        try:
            done, _pending = await asyncio.wait(
                {ready_wait, self._task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if ready_wait in done:
                await ready_wait
                return
            exception = self._task.exception()
            if exception is not None:
                raise exception
            raise RuntimeError("BTC spot feed stopped before publishing a ready snapshot.")
        finally:
            ready_wait.cancel()
            try:
                await ready_wait
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        tasks = []
        if "coinbase" in self.config.venues:
            tasks.append(asyncio.create_task(self._coinbase_loop(), name="btc-spot-coinbase"))
        if "kraken" in self.config.venues:
            tasks.append(asyncio.create_task(self._kraken_loop(), name="btc-spot-kraken"))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

    async def _coinbase_loop(self) -> None:
        async with connect(self.config.coinbase_url, max_size=None) as websocket:
            await websocket.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"}))
            await websocket.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))
            async for raw_message in websocket:
                if self._stop_event.is_set():
                    return
                received_at = _utc_now()
                payload = json.loads(raw_message)
                if self.config.archive_raw:
                    await self._raw_logger.write("coinbase_ws_message", payload, received_at)
                quote = parse_coinbase_ticker_message(payload, received_at=received_at)
                if quote is not None:
                    await self._publish_quote(quote)

    async def _kraken_loop(self) -> None:
        async with connect(self.config.kraken_url, max_size=None) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "book",
                            "symbol": ["BTC/USD"],
                            "depth": 10,
                            "snapshot": True,
                        },
                    }
                )
            )
            async for raw_message in websocket:
                if self._stop_event.is_set():
                    return
                received_at = _utc_now()
                payload = json.loads(raw_message)
                if self.config.archive_raw:
                    await self._raw_logger.write("kraken_ws_message", payload, received_at)
                quote = parse_kraken_book_message(payload, received_at=received_at)
                if quote is not None:
                    await self._publish_quote(quote)

    async def _publish_quote(self, quote: BTCSpotVenueQuote) -> None:
        update = self._ingest_quote(quote)
        if update is None:
            return
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)
        if self.config.archive_normalized:
            await self._normalized_logger.write(
                "btc_spot_update",
                {
                    "event_time": update.event_time.isoformat(),
                    "btc_spot_price": update.btc_spot_price,
                    "btc_spot_twap_60s": update.btc_spot_twap_60s,
                    "btc_spot_age_ms": update.btc_spot_age_ms,
                    "btc_spot_is_fresh": update.btc_spot_is_fresh,
                    "btc_spot_venues_fresh": update.btc_spot_venues_fresh,
                    "btc_spot_venue_divergence_bps": update.btc_spot_venue_divergence_bps,
                    "btc_spot_source": update.btc_spot_source,
                },
                update.event_time,
            )

    def _ingest_quote(self, quote: BTCSpotVenueQuote) -> BTCSpotUpdate | None:
        self._venue_quotes[quote.venue] = quote
        update = self._build_update(reference_time=quote.exchange_event_time)
        if update is None:
            return None
        self._latest_update = update
        self._updates.append(update)
        self._prune_updates(reference_time=quote.exchange_event_time)
        self._ready_event.set()
        return update

    def _build_update(self, *, reference_time: datetime) -> BTCSpotUpdate | None:
        fresh_quotes: list[BTCSpotVenueQuote] = []
        for quote in self._venue_quotes.values():
            age_ms = max(0.0, (reference_time - quote.exchange_event_time).total_seconds() * 1000.0)
            if age_ms <= float(self.config.max_age_ms) and not quote.is_resyncing:
                fresh_quotes.append(quote)
        if not fresh_quotes:
            return None
        selected_quotes = list(fresh_quotes)
        venue_divergence_bps = 0.0
        if len(fresh_quotes) >= 2:
            mids = [quote.mid_price for quote in fresh_quotes]
            venue_divergence_bps = abs(max(mids) - min(mids)) / ((max(mids) + min(mids)) / 2.0) * 10000.0
            if venue_divergence_bps > float(self.config.venue_divergence_bps_gate):
                selected_quotes = [min(fresh_quotes, key=lambda quote: (quote.spread, quote.venue))]
        consolidated_mid = self._consolidated_mid(selected_quotes)
        if consolidated_mid is None:
            return None
        self._record_resampled_point(reference_time, consolidated_mid)
        metrics = self._compute_metrics(reference_time)
        return BTCSpotUpdate(
            event_time=reference_time,
            received_at=_utc_now(),
            btc_spot_price=consolidated_mid,
            btc_spot_twap_60s=metrics["twap_60s"],
            btc_spot_age_ms=0.0,
            btc_spot_is_fresh=True,
            btc_spot_venues_fresh=len(fresh_quotes),
            btc_spot_venue_divergence_bps=venue_divergence_bps,
            btc_vol_effective_sample_size=metrics["effective_sample_size"],
            btc_spot_source="quote",
            btc_spot_return_30s=metrics["return_30s"],
            btc_spot_return_120s=metrics["return_120s"],
            btc_spot_return_300s=metrics["return_300s"],
            btc_spot_return_900s=metrics["return_900s"],
            btc_spot_vol_120s=metrics["vol_120s"],
            btc_spot_vol_300s=metrics["vol_300s"],
            btc_spot_vol_900s=metrics["vol_900s"],
            btc_spot_vol_1800s=metrics["vol_1800s"],
            btc_spot_vol_ewma_hl300=metrics["vol_ewma_hl300"],
        )

    def _consolidated_mid(self, quotes: list[BTCSpotVenueQuote]) -> float | None:
        if not quotes:
            return None
        if len(quotes) == 1:
            return quotes[0].mid_price
        tick_floor = 0.01
        weights = [1.0 / max(quote.spread, tick_floor) for quote in quotes]
        return float(sum(weight * quote.mid_price for quote, weight in zip(quotes, weights)) / sum(weights))

    def _record_resampled_point(self, event_time: datetime, price: float) -> None:
        bucket = event_time.replace(microsecond=0)
        if self._resampled_history and self._resampled_history[-1][0] == bucket:
            self._resampled_history[-1] = (bucket, float(price))
        else:
            self._resampled_history.append((bucket, float(price)))
        self._last_resampled_second = bucket
        cutoff = bucket - timedelta(seconds=self.config.history_window_seconds)
        while self._resampled_history and self._resampled_history[0][0] < cutoff:
            self._resampled_history.popleft()

    def _prune_updates(self, *, reference_time: datetime) -> None:
        cutoff = reference_time - timedelta(seconds=self.config.history_window_seconds)
        while self._updates and self._updates[0].event_time < cutoff:
            self._updates.popleft()

    def _compute_metrics(self, reference_time: datetime) -> dict[str, float | None]:
        points = list(self._resampled_history)
        if not points:
            return {
                "twap_60s": None,
                "effective_sample_size": 0.0,
                "return_30s": None,
                "return_120s": None,
                "return_300s": None,
                "return_900s": None,
                "vol_120s": None,
                "vol_300s": None,
                "vol_900s": None,
                "vol_1800s": None,
                "vol_ewma_hl300": None,
            }
        series = [(timestamp, price) for timestamp, price in points if price > 0.0]
        if not series:
            return {
                "twap_60s": None,
                "effective_sample_size": 0.0,
                "return_30s": None,
                "return_120s": None,
                "return_300s": None,
                "return_900s": None,
                "vol_120s": None,
                "vol_300s": None,
                "vol_900s": None,
                "vol_1800s": None,
                "vol_ewma_hl300": None,
            }
        timestamps = [timestamp for timestamp, _price in series]
        prices = np.asarray([price for _timestamp, price in series], dtype=np.float64)
        log_returns = np.diff(np.log(prices)) if len(prices) > 1 else np.asarray([], dtype=np.float64)
        annualizer = math.sqrt(365.0 * 24.0 * 60.0 * 60.0)

        def latest_price_before(cutoff_seconds: int) -> float | None:
            cutoff_time = reference_time - timedelta(seconds=cutoff_seconds)
            eligible = [price for timestamp, price in series if timestamp <= cutoff_time]
            if not eligible:
                return None
            return float(eligible[-1])

        def return_window(seconds: int) -> float | None:
            previous = latest_price_before(seconds)
            if previous is None or prices[-1] <= 0.0:
                return None
            return float(math.log(prices[-1] / previous))

        def vol_window(seconds: int) -> float | None:
            cutoff_time = reference_time - timedelta(seconds=seconds)
            start_index = next((index for index, timestamp in enumerate(timestamps) if timestamp >= cutoff_time), None)
            if start_index is None:
                return None
            window_prices = prices[start_index:]
            if len(window_prices) <= 1:
                return 0.0
            window_returns = np.diff(np.log(window_prices))
            if len(window_returns) == 0:
                return 0.0
            return float(window_returns.std() * annualizer)

        def ewma_vol() -> float | None:
            if len(log_returns) == 0:
                return 0.0
            half_life = 300.0
            deltas = []
            last_time = timestamps[1:]
            for timestamp in last_time:
                delta = (reference_time - timestamp).total_seconds()
                deltas.append(max(delta, 0.0))
            weights = np.asarray([math.exp(math.log(0.5) * delta / half_life) for delta in deltas], dtype=np.float64)
            if weights.sum() <= 0.0:
                return 0.0
            variance = float(np.dot(weights, np.square(log_returns)) / weights.sum())
            return float(math.sqrt(max(variance, 0.0)) * annualizer)

        twap_cutoff = reference_time - timedelta(seconds=self.config.twap_window_seconds)
        twap_prices = [price for timestamp, price in series if timestamp >= twap_cutoff]
        return {
            "twap_60s": float(np.mean(twap_prices)) if twap_prices else float(prices[-1]),
            "effective_sample_size": float(len(log_returns)),
            "return_30s": return_window(30),
            "return_120s": return_window(120),
            "return_300s": return_window(300),
            "return_900s": return_window(900),
            "vol_120s": vol_window(120),
            "vol_300s": vol_window(300),
            "vol_900s": vol_window(900),
            "vol_1800s": vol_window(1800),
            "vol_ewma_hl300": ewma_vol(),
        }
