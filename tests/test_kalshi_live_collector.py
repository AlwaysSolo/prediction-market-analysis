from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.indexers.kalshi.models import Market
from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiMarketDataCollector,
    KalshiTickerState,
)
from src.live.kalshi.auth import build_auth_headers
from src.live.kalshi.types import state_to_update


def _config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
        max_stale_message_age_seconds=0.0,
    )


def test_environment_urls():
    assert KalshiEnvironment.DEMO.api_base_url.endswith("/trade-api/v2")
    assert KalshiEnvironment.PRODUCTION.websocket_url.startswith("wss://")


def test_state_derivatives():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    state = KalshiTickerState(
        ticker="KXBTC15M-TEST",
        last_yes_price_cents=55,
        last_trade_time=now,
        previous_yes_price_cents=54,
        close_time=now + timedelta(minutes=7),
        is_open=True,
    )
    update = state_to_update(state, now, now=now)
    assert update.market_prob == 0.55
    assert update.previous_market_prob == 0.54
    assert round(update.price_momentum, 2) == 0.01
    assert round(update.tau_minutes, 2) == 7.00


def test_auth_headers_include_required_fields(tmp_path: Path):
    key_path = tmp_path / "demo.pem"
    key_path.write_bytes(
        rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    headers = build_auth_headers(
        KalshiCredentials(api_key_id="demo", private_key_path=key_path),
        "GET",
        "/trade-api/ws/v2",
        1234567890,
    )
    assert headers["KALSHI-ACCESS-KEY"] == "demo"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1234567890"
    assert headers["KALSHI-ACCESS-SIGNATURE"]


def test_trade_message_updates_state(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="open",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=None,
        volume=0,
        volume_24h=0,
        open_interest=0,
        result="",
        created_time=None,
        open_time=None,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
    )

    async def run() -> None:
        await collector._handle_trade_message(
            {
                "market_ticker": ticker,
                "yes_price_dollars": "0.540",
                "ts": 1767225600,
            }
        )
        await collector._handle_trade_message(
            {
                "market_ticker": ticker,
                "yes_price_dollars": "0.550",
                "ts": 1767225660,
            }
        )

    asyncio.run(run())
    state = collector.get_state(ticker)
    assert state is not None
    assert state.last_yes_price_cents == 55
    assert state.previous_yes_price_cents == 54
    assert round(state.price_momentum or 0.0, 2) == 0.01
    assert state.count == 0


def test_stale_trade_message_is_ignored(tmp_path: Path):
    collector = KalshiMarketDataCollector(
        KalshiCollectorConfig(
            environment=KalshiEnvironment.DEMO,
            credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
            log_dir=tmp_path / "logs",
            series_tickers=("KXBTC15M",),
            metadata_refresh_interval_seconds=3600.0,
            max_stale_message_age_seconds=1.0,
        )
    )
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="open",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=None,
        volume=0,
        volume_24h=0,
        open_interest=0,
        result="",
        created_time=None,
        open_time=None,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
    )

    stale_timestamp = int((datetime.now(UTC) - timedelta(seconds=10)).timestamp())

    async def run() -> None:
        await collector._handle_trade_message(
            {
                "market_ticker": ticker,
                "yes_price_dollars": "0.540",
                "ts": stale_timestamp,
            }
        )

    asyncio.run(run())
    assert collector.get_state(ticker) is None


def test_ticker_message_updates_quote_fields(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
        is_open=True,
    )

    async def run() -> None:
        await collector._handle_ticker_message(
            {
                "market_ticker": ticker,
                "price_dollars": "0.560",
                "yes_bid_dollars": "0.540",
                "yes_ask_dollars": "0.570",
                "yes_bid_size_fp": "2.00",
                "yes_ask_size_fp": "5.00",
                "volume_fp": "123.00",
                "open_interest_fp": "77.00",
                "dollar_volume": 456,
                "dollar_open_interest": 210,
                "time": "2026-01-01T12:01:00Z",
            }
        )

    asyncio.run(run())
    state = collector.get_state(ticker)
    assert state is not None
    assert state.last_yes_price_cents == 55
    assert state.last_price_cents == 56
    assert state.yes_bid_cents == 54
    assert state.yes_ask_cents == 57
    assert state.no_bid_cents == 43
    assert state.no_ask_cents == 46
    assert state.yes_bid_size == 2
    assert state.yes_ask_size == 5
    assert state.no_bid_size == 5
    assert state.no_ask_size == 2
    assert state.volume == 123
    assert state.open_interest == 77
    assert state.dollar_volume == 456
    assert state.dollar_open_interest == 210
    assert state.ticker_update_time == datetime(2026, 1, 1, 12, 1, tzinfo=UTC)


def test_ticker_message_replaces_stale_no_side_with_yes_implied_quotes(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=36,
        last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        previous_yes_price_cents=35,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
        is_open=True,
        yes_bid_cents=36,
        yes_ask_cents=37,
        no_bid_cents=63,
        no_ask_cents=64,
    )

    async def run() -> None:
        await collector._handle_ticker_message(
            {
                "market_ticker": ticker,
                "price_dollars": "0.090",
                "yes_bid_dollars": "0.080",
                "yes_ask_dollars": "0.100",
                "yes_bid_size_fp": "1.00",
                "yes_ask_size_fp": "4.00",
                "time": "2026-01-01T12:01:00Z",
            }
        )

    asyncio.run(run())
    state = collector.get_state(ticker)
    assert state is not None
    assert state.yes_bid_cents == 8
    assert state.yes_ask_cents == 10
    assert state.no_bid_cents == 90
    assert state.no_ask_cents == 92
    assert state.yes_bid_size == 1
    assert state.yes_ask_size == 4
    assert state.no_bid_size == 4
    assert state.no_ask_size == 1


def test_get_market_returns_cached_market_and_lifecycle_updates_result(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="open",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=None,
        volume=0,
        volume_24h=0,
        open_interest=0,
        result="",
        created_time=None,
        open_time=None,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
    )

    async def run() -> None:
        await collector._handle_market_lifecycle_message(
            {
                "market_ticker": ticker,
                "event_type": "settled",
                "result": "YES",
                "close_ts": int(datetime(2026, 1, 1, 12, 10, tzinfo=UTC).timestamp()),
            }
        )

    asyncio.run(run())
    market = collector.get_market(ticker)
    assert market is not None
    assert market.result == "YES"


def test_refresh_market_snapshot_fetches_latest_result(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"

    class FakeRestClient:
        def get_market(self, market_ticker: str) -> Market:
            assert market_ticker == ticker
            return Market(
                ticker=ticker,
                event_ticker="KXBTC15M",
                market_type="binary",
                title="t",
                yes_sub_title="y",
                no_sub_title="n",
                status="settled",
                yes_bid=54,
                yes_ask=56,
                no_bid=44,
                no_ask=46,
                last_price=55,
                volume=0,
                volume_24h=0,
                open_interest=0,
                result="YES",
                created_time=None,
                open_time=None,
                close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
            )

    collector._rest_client = FakeRestClient()
    asyncio.run(collector.refresh_market_snapshot(ticker))

    market = collector.get_market(ticker)
    state = collector.get_state(ticker)
    assert market is not None
    assert market.result == "YES"
    assert state is not None
    assert state.is_open is False


def test_refresh_market_snapshot_demo_falls_back_to_public_market_result(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"

    class FakeRestClient:
        def get_market(self, market_ticker: str) -> Market:
            assert market_ticker == ticker
            return Market(
                ticker=ticker,
                event_ticker="KXBTC15M",
                market_type="binary",
                title="t",
                yes_sub_title="y",
                no_sub_title="n",
                status="closed",
                yes_bid=54,
                yes_ask=56,
                no_bid=44,
                no_ask=46,
                last_price=55,
                volume=0,
                volume_24h=0,
                open_interest=0,
                result="",
                created_time=None,
                open_time=None,
                close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
            )

        def get_public_market(self, market_ticker: str, environment: KalshiEnvironment | None = None) -> Market:
            assert market_ticker == ticker
            assert environment is KalshiEnvironment.PRODUCTION
            return Market(
                ticker=ticker,
                event_ticker="KXBTC15M",
                market_type="binary",
                title="t",
                yes_sub_title="y",
                no_sub_title="n",
                status="finalized",
                yes_bid=99,
                yes_ask=100,
                no_bid=0,
                no_ask=1,
                last_price=99,
                volume=0,
                volume_24h=0,
                open_interest=0,
                result="YES",
                created_time=None,
                open_time=None,
                close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
            )

    collector._rest_client = FakeRestClient()
    asyncio.run(collector.refresh_market_snapshot(ticker))

    market = collector.get_market(ticker)
    state = collector.get_state(ticker)
    assert market is not None
    assert market.result == "YES"
    assert market.status == "finalized"
    assert state is not None
    assert state.is_open is False


def test_collector_logs_raw_stream_messages_when_enabled(tmp_path: Path):
    collector = KalshiMarketDataCollector(
        KalshiCollectorConfig(
            environment=KalshiEnvironment.DEMO,
            credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
            log_dir=tmp_path / "logs",
            series_tickers=("KXBTC15M",),
            metadata_refresh_interval_seconds=3600.0,
            max_stale_message_age_seconds=0.0,
            log_stream_messages=True,
        )
    )
    ticker = "KXBTC15M-TEST"

    async def run() -> None:
        await collector._handle_ticker_message(
            {
                "market_ticker": ticker,
                "price_dollars": "0.560",
                "yes_bid_dollars": "0.540",
                "yes_ask_dollars": "0.570",
                "volume_fp": "123.00",
                "open_interest_fp": "77.00",
                "time": "2026-01-01T12:01:00Z",
            }
        )

    asyncio.run(run())
    path = tmp_path / "logs" / "demo" / "2026-01-01" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["event_type"] == "ws_ticker_message" for row in rows)


def test_market_lifecycle_updates_close_time_and_open_flag(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
        is_open=True,
    )

    async def run() -> None:
        await collector._handle_market_lifecycle_message(
            {
                "market_ticker": ticker,
                "event_type": "settled",
                "close_ts": 1767226200,
            }
        )

    asyncio.run(run())
    state = collector.get_state(ticker)
    assert state is not None
    assert state.is_open is False
    assert state.close_time == datetime.fromtimestamp(1767226200, tz=UTC)


def test_raw_jsonl_logging(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))

    async def run() -> None:
        await collector._logger.write("test_event", {"x": 1}, datetime(2026, 1, 1, tzinfo=UTC))

    asyncio.run(run())
    path = tmp_path / "logs" / "demo" / "2026-01-01" / "events.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["event_type"] == "test_event"
    assert row["payload"] == {"x": 1}


def test_subscription_messages_split_trade_and_lifecycle_filters(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    collector._subscribed_tickers = ("KXBTC15M-1", "KXBTC15M-2")

    assert collector._subscription_messages() == [
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["trade"], "market_tickers": ["KXBTC15M-1", "KXBTC15M-2"]},
        },
        {
            "id": 2,
            "cmd": "subscribe",
            "params": {"channels": ["ticker"], "market_tickers": ["KXBTC15M-1", "KXBTC15M-2"]},
        },
        {
            "id": 3,
            "cmd": "subscribe",
            "params": {"channels": ["market_lifecycle_v2"]},
        },
    ]


def test_refresh_market_metadata_preserves_trade_state(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
        is_open=True,
    )

    class FakeRestClient:
        def iter_markets(self, **_kwargs):
            yield (
                {"markets": [{"ticker": ticker}]},
                [
                    Market(
                        ticker=ticker,
                        event_ticker="KXBTC15M",
                        market_type="binary",
                        title="t",
                        yes_sub_title="y",
                        no_sub_title="n",
                        status="open",
                        yes_bid=None,
                        yes_ask=None,
                        no_bid=None,
                        no_ask=None,
                        last_price=None,
                        volume=0,
                        volume_24h=0,
                        open_interest=0,
                        result="",
                        created_time=None,
                        open_time=None,
                        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
                    )
                ],
            )

    collector._rest_client = FakeRestClient()
    asyncio.run(collector.refresh_market_metadata())

    state = collector.get_state(ticker)
    assert state is not None
    assert state.last_yes_price_cents == 55
    assert state.previous_yes_price_cents == 54
    assert state.close_time == datetime(2026, 1, 1, 12, 15, tzinfo=UTC)


def test_refresh_market_metadata_marks_active_market_open(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"

    class FakeRestClient:
        def iter_markets(self, **_kwargs):
            yield (
                {"markets": [{"ticker": ticker}]},
                [
                    Market(
                        ticker=ticker,
                        event_ticker="KXBTC15M",
                        market_type="binary",
                        title="t",
                        yes_sub_title="y",
                        no_sub_title="n",
                        status="active",
                        yes_bid=None,
                        yes_ask=None,
                        no_bid=None,
                        no_ask=None,
                        last_price=None,
                        volume=0,
                        volume_24h=0,
                        open_interest=0,
                        result="",
                        created_time=None,
                        open_time=None,
                        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
                    )
                ],
            )

    collector._rest_client = FakeRestClient()
    asyncio.run(collector.refresh_market_metadata())

    state = collector.get_state(ticker)
    assert state is not None
    assert state.is_open is True


def test_refresh_market_metadata_requests_resubscribe_when_ticker_set_changes(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    collector._tasks = [object()]
    collector._subscribed_tickers = ("KXBTC15M-OLD",)

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self.close_code = None
            self.close_reason = None

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

    fake_websocket = _FakeWebSocket()
    collector._active_websocket = fake_websocket
    ticker = "KXBTC15M-NEW"

    class FakeRestClient:
        def iter_markets(self, **_kwargs):
            yield (
                {"markets": [{"ticker": ticker}]},
                [
                    Market(
                        ticker=ticker,
                        event_ticker="KXBTC15M",
                        market_type="binary",
                        title="t",
                        yes_sub_title="y",
                        no_sub_title="n",
                        status="active",
                        yes_bid=None,
                        yes_ask=None,
                        no_bid=None,
                        no_ask=None,
                        last_price=None,
                        volume=0,
                        volume_24h=0,
                        open_interest=0,
                        result="",
                        created_time=None,
                        open_time=None,
                        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
                    )
                ],
            )

    collector._rest_client = FakeRestClient()

    async def run() -> None:
        await collector.refresh_market_metadata()

    asyncio.run(run())

    assert collector._subscribed_tickers == (ticker,)
    assert fake_websocket.closed is True
    assert fake_websocket.close_reason == "subscription_refresh"


def test_refresh_market_metadata_uses_series_filter_for_server_side_market_lookup(tmp_path: Path):
    collector = KalshiMarketDataCollector(_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    calls: list[dict[str, object]] = []

    class FakeRestClient:
        def get_markets_page(
            self,
            *,
            limit: int,
            status: str,
            cursor: str | None = None,
            event_ticker: str | None = None,
            series_ticker: str | None = None,
            tickers: tuple[str, ...] | None = None,
        ):
            calls.append(
                {
                    "limit": limit,
                    "status": status,
                    "cursor": cursor,
                    "event_ticker": event_ticker,
                    "series_ticker": series_ticker,
                    "tickers": tickers,
                }
            )
            return (
                {"markets": [{"ticker": ticker}]},
                [
                    Market(
                        ticker=ticker,
                        event_ticker="KXBTC15M",
                        market_type="binary",
                        title="t",
                        yes_sub_title="y",
                        no_sub_title="n",
                        status="active",
                        yes_bid=None,
                        yes_ask=None,
                        no_bid=None,
                        no_ask=None,
                        last_price=None,
                        volume=0,
                        volume_24h=0,
                        open_interest=0,
                        result="",
                        created_time=None,
                        open_time=None,
                        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
                    )
                ],
            )

    collector._rest_client = FakeRestClient()

    async def run() -> None:
        await collector.refresh_market_metadata()

    asyncio.run(run())

    assert calls == [
        {
            "limit": 1000,
            "status": "open",
            "cursor": None,
            "event_ticker": None,
            "series_ticker": "KXBTC15M",
            "tickers": None,
        }
    ]
    assert collector._subscribed_tickers == (ticker,)
