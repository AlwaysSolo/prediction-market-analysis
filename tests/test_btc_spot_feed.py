from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.live.data.btc_spot_feed import (
    BTCSpotFeed,
    BTCSpotFeedConfig,
    BTCSpotVenueQuote,
    parse_coinbase_ticker_message,
    parse_kraken_book_message,
)


def _quote(
    *,
    venue: str,
    bid: float,
    ask: float,
    event_time: datetime,
    bid_size: float = 1.0,
    ask_size: float = 1.0,
) -> BTCSpotVenueQuote:
    return BTCSpotVenueQuote(
        venue=venue,
        symbol="BTC-USD" if venue == "coinbase" else "BTC/USD",
        best_bid=bid,
        best_ask=ask,
        best_bid_size=bid_size,
        best_ask_size=ask_size,
        exchange_event_time=event_time,
        received_at=event_time + timedelta(milliseconds=25),
        sequence=101,
        is_resyncing=False,
    )


def test_parse_coinbase_ticker_message_returns_top_of_book_quote() -> None:
    payload = {
        "channel": "ticker",
        "timestamp": "2026-04-18T12:00:00.100000Z",
        "sequence_num": 42,
        "events": [
            {
                "type": "update",
                "tickers": [
                    {
                        "product_id": "BTC-USD",
                        "best_bid": "85000.12",
                        "best_bid_quantity": "1.25",
                        "best_ask": "85001.34",
                        "best_ask_quantity": "0.75",
                    }
                ],
            }
        ],
    }

    quote = parse_coinbase_ticker_message(payload, received_at=datetime(2026, 4, 18, 12, 0, 0, 150000, tzinfo=UTC))

    assert quote is not None
    assert quote.venue == "coinbase"
    assert quote.symbol == "BTC-USD"
    assert quote.best_bid == pytest.approx(85000.12)
    assert quote.best_ask == pytest.approx(85001.34)
    assert quote.best_bid_size == pytest.approx(1.25)
    assert quote.best_ask_size == pytest.approx(0.75)
    assert quote.exchange_event_time == datetime(2026, 4, 18, 12, 0, 0, 100000, tzinfo=UTC)
    assert quote.sequence == 42


def test_parse_kraken_book_message_returns_top_of_book_quote() -> None:
    payload = {
        "channel": "book",
        "type": "snapshot",
        "data": [
            {
                "symbol": "BTC/USD",
                "bids": [
                    {"price": 84999.5, "qty": 2.0},
                    {"price": 84999.0, "qty": 1.0},
                ],
                "asks": [
                    {"price": 85000.5, "qty": 3.5},
                    {"price": 85001.0, "qty": 1.5},
                ],
                "checksum": 1234,
                "timestamp": "2026-04-18T12:00:00.200000Z",
            }
        ],
    }

    quote = parse_kraken_book_message(payload, received_at=datetime(2026, 4, 18, 12, 0, 0, 260000, tzinfo=UTC))

    assert quote is not None
    assert quote.venue == "kraken"
    assert quote.symbol == "BTC/USD"
    assert quote.best_bid == pytest.approx(84999.5)
    assert quote.best_ask == pytest.approx(85000.5)
    assert quote.best_bid_size == pytest.approx(2.0)
    assert quote.best_ask_size == pytest.approx(3.5)
    assert quote.exchange_event_time == datetime(2026, 4, 18, 12, 0, 0, 200000, tzinfo=UTC)
    assert quote.sequence is None


def test_consolidated_mid_uses_inverse_spread_weighting_and_outlier_gate(tmp_path: Path) -> None:
    feed = BTCSpotFeed(
        BTCSpotFeedConfig(
            environment="test",
            log_dir=tmp_path / "logs",
            archive_raw=False,
            archive_normalized=False,
            max_age_ms=500,
            venue_divergence_bps_gate=25.0,
        )
    )
    event_time = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)

    feed._ingest_quote(  # type: ignore[attr-defined]
        _quote(venue="coinbase", bid=85000.0, ask=85001.0, bid_size=1.0, ask_size=1.0, event_time=event_time)
    )
    feed._ingest_quote(  # type: ignore[attr-defined]
        _quote(venue="kraken", bid=85000.0, ask=85002.0, bid_size=1.0, ask_size=1.0, event_time=event_time)
    )

    snapshot = feed.snapshot_state()
    assert snapshot is not None
    expected_mid = ((1.0 / 1.0) * 85000.5 + (1.0 / 2.0) * 85001.0) / ((1.0 / 1.0) + (1.0 / 2.0))
    assert snapshot.btc_spot_price == pytest.approx(expected_mid)
    assert snapshot.btc_spot_venues_fresh == 2
    assert snapshot.btc_spot_is_fresh is True

    feed._ingest_quote(  # type: ignore[attr-defined]
        _quote(
            venue="kraken",
            bid=85280.0,
            ask=85282.0,
            bid_size=1.0,
            ask_size=1.0,
            event_time=event_time + timedelta(milliseconds=100),
        )
    )
    gated_snapshot = feed.snapshot_state()
    assert gated_snapshot is not None
    assert gated_snapshot.btc_spot_price == pytest.approx(85000.5)
    assert gated_snapshot.btc_spot_venue_divergence_bps > 25.0


def test_lookup_at_or_before_uses_event_time_and_max_age(tmp_path: Path) -> None:
    feed = BTCSpotFeed(
        BTCSpotFeedConfig(
            environment="test",
            log_dir=tmp_path / "logs",
            archive_raw=False,
            archive_normalized=False,
            max_age_ms=500,
        )
    )
    event_time = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    feed._ingest_quote(  # type: ignore[attr-defined]
        _quote(venue="coinbase", bid=85000.0, ask=85001.0, event_time=event_time)
    )

    assert feed.lookup_at_or_before(event_time + timedelta(milliseconds=300), max_age_ms=500) is not None
    assert feed.lookup_at_or_before(event_time + timedelta(milliseconds=700), max_age_ms=500) is None


def test_wait_until_ready_blocks_until_first_quote(tmp_path: Path) -> None:
    class _IdleFeed(BTCSpotFeed):
        async def _run(self) -> None:  # pragma: no cover - exercised via start/stop lifecycle
            await self._stop_event.wait()

    feed = _IdleFeed(
        BTCSpotFeedConfig(
            environment="test",
            log_dir=tmp_path / "logs",
            archive_raw=False,
            archive_normalized=False,
        )
    )
    event_time = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)

    async def run() -> None:
        await feed.start()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(feed.wait_until_ready(), timeout=0.05)
        feed._ingest_quote(  # type: ignore[attr-defined]
            _quote(venue="coinbase", bid=85000.0, ask=85001.0, event_time=event_time)
        )
        await asyncio.wait_for(feed.wait_until_ready(), timeout=0.05)
        await feed.stop()

    asyncio.run(run())
