from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from src.indexers.kalshi.models import Market, Trade
from src.live.kalshi.data_download import KXBTC15MQuoteAwareDataDownloader


def _http_404() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(404, request=request)
    return httpx.HTTPStatusError("not found", request=request, response=response)


class FakeKalshiClient:
    def __init__(self, payloads: dict[str, object]):
        self.payloads = payloads

    def __enter__(self) -> FakeKalshiClient:
        return self

    def __exit__(self, *args) -> None:
        return None

    def close(self) -> None:
        return None

    def get_historical_cutoff(self) -> dict[str, str]:
        return {
            "market_settled_ts": "2026-03-01T00:00:00Z",
            "trades_created_ts": "2026-03-01T00:00:00Z",
            "orders_updated_ts": "2026-03-01T00:00:00Z",
        }

    def iter_markets_pages(self, **kwargs):
        status = kwargs.get("status")
        market = self.payloads.get(f"market_page:{status}")
        if market is None:
            return iter([])
        return iter([({"cursor": ""}, [market])])

    def get_market(self, ticker: str) -> Market:
        value = self.payloads.get(f"live_market:{ticker}", _http_404())
        if isinstance(value, Exception):
            raise value
        return value

    def get_historical_market(self, ticker: str) -> Market:
        value = self.payloads.get(f"historical_market:{ticker}", _http_404())
        if isinstance(value, Exception):
            raise value
        return value

    def iter_historical_trades_pages(self, **kwargs):
        ticker = kwargs["ticker"]
        return iter(self.payloads.get(f"historical_trades:{ticker}", [({"cursor": ""}, [])]))

    def iter_market_trades_pages(self, **kwargs):
        ticker = kwargs["ticker"]
        return iter(self.payloads.get(f"live_trades:{ticker}", [({"cursor": ""}, [])]))

    def get_market_candlesticks(self, *, ticker: str, **kwargs):
        value = self.payloads.get(f"live_candles:{ticker}", _http_404())
        if isinstance(value, Exception):
            raise value
        return value

    def get_historical_market_candlesticks(self, *, ticker: str, **kwargs):
        value = self.payloads.get(f"historical_candles:{ticker}", _http_404())
        if isinstance(value, Exception):
            raise value
        return value


def _market(
    ticker: str,
    *,
    status: str = "settled",
    open_time: datetime | None = None,
    close_time: datetime | None = None,
) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        market_type="binary",
        title=ticker,
        yes_sub_title="yes",
        no_sub_title="no",
        status=status,
        yes_bid=45,
        yes_ask=55,
        no_bid=45,
        no_ask=55,
        last_price=50,
        volume=100,
        volume_24h=100,
        open_interest=20,
        result="yes" if status == "settled" else "",
        created_time=open_time,
        open_time=open_time,
        close_time=close_time,
    )


def _trade(trade_id: str, ticker: str, *, created_time: datetime, yes_price: int) -> Trade:
    return Trade(
        trade_id=trade_id,
        ticker=ticker,
        count=1,
        yes_price=yes_price,
        no_price=100 - yes_price,
        taker_side="yes",
        created_time=created_time,
    )


def test_build_ticker_universe_merges_local_archives_and_live_markets(tmp_path: Path):
    existing_root = tmp_path / "existing"
    trades_dir = existing_root / "KXBTC15M" / "trades"
    trades_dir.mkdir(parents=True)
    pd.DataFrame([{"trade_id": "1"}]).to_parquet(trades_dir / "KXBTC15M-ARCHIVE-00.parquet")
    pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-LOCAL-15",
                "event_ticker": "KXBTC15M-LOCAL",
                "status": "settled",
                "open_time": datetime(2026, 2, 1, tzinfo=UTC),
                "close_time": datetime(2026, 2, 1, 0, 15, tzinfo=UTC),
                "result": "yes",
            }
        ]
    ).to_parquet(existing_root / "KXBTC15M_markets.parquet")

    live_market = _market(
        "KXBTC15M-LIVE-30",
        status="open",
        open_time=datetime(2026, 3, 28, 12, 15, tzinfo=UTC),
        close_time=datetime(2026, 3, 28, 12, 30, tzinfo=UTC),
    )
    downloader = KXBTC15MQuoteAwareDataDownloader(
        root_dir=tmp_path / "dataset",
        existing_roots=(existing_root,),
        client_factory=lambda: FakeKalshiClient({"market_page:open": live_market}),
    )

    manifest = downloader.build_ticker_universe()
    tickers = {item["ticker"] for item in manifest}
    assert tickers == {"KXBTC15M-ARCHIVE-00", "KXBTC15M-LOCAL-15", "KXBTC15M-LIVE-30"}


def test_sync_merges_live_and_historical_trades_and_writes_candles(tmp_path: Path):
    ticker = "KXBTC15M-26FEB010015-15"
    open_time = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    close_time = datetime(2026, 2, 1, 0, 15, tzinfo=UTC)
    payloads = {
        "market_page:settled": _market(ticker, status="settled", open_time=open_time, close_time=close_time),
        f"live_market:{ticker}": _http_404(),
        f"historical_market:{ticker}": _market(ticker, status="settled", open_time=open_time, close_time=close_time),
        f"historical_trades:{ticker}": [
            ({"cursor": ""}, [_trade("1", ticker, created_time=open_time, yes_price=53)])
        ],
        f"live_trades:{ticker}": [
            (
                {"cursor": ""},
                [
                    _trade("1", ticker, created_time=open_time, yes_price=53),
                    _trade("2", ticker, created_time=close_time, yes_price=54),
                ],
            )
        ],
        f"historical_candles:{ticker}": {
            "ticker": ticker,
            "candlesticks": [
                {
                    "end_period_ts": int(close_time.timestamp()),
                    "yes_bid": {"open": "0.51", "low": "0.50", "high": "0.52", "close": "0.51"},
                    "yes_ask": {"open": "0.53", "low": "0.52", "high": "0.54", "close": "0.53"},
                    "price": {
                        "open": "0.52",
                        "low": "0.52",
                        "high": "0.53",
                        "close": "0.53",
                        "mean": "0.525",
                        "previous": "0.51",
                    },
                    "volume": "10.00",
                    "open_interest": "4.00",
                }
            ],
        },
    }
    downloader = KXBTC15MQuoteAwareDataDownloader(
        root_dir=tmp_path / "dataset",
        existing_roots=(),
        client_factory=lambda: FakeKalshiClient(payloads),
    )

    report = downloader.sync(max_workers=1)
    trades_df = pd.read_parquet(tmp_path / "dataset" / "curated" / "trades" / f"{ticker}.parquet")
    candles_df = pd.read_parquet(tmp_path / "dataset" / "curated" / "candles_1m" / f"{ticker}.parquet")
    markets_df = pd.read_parquet(tmp_path / "dataset" / "curated" / "markets" / "KXBTC15M_markets.parquet")

    assert len(trades_df) == 2
    assert set(trades_df["trade_id"]) == {"1", "2"}
    assert len(candles_df) == 1
    assert set(markets_df["ticker"]) == {ticker}
    assert report["summary"]["trade_ready_count"] == 1
    assert (tmp_path / "dataset" / "raw" / "trades" / f"{ticker}.json").exists()


def test_materialize_live_quote_logs_parses_ws_messages(tmp_path: Path):
    downloader = KXBTC15MQuoteAwareDataDownloader(
        root_dir=tmp_path / "dataset",
        existing_roots=(),
        client_factory=lambda: FakeKalshiClient({}),
    )
    raw_dir = tmp_path / "dataset" / "raw" / "live_ticker" / "production" / "2026-03-28"
    raw_dir.mkdir(parents=True)
    events = [
        {
            "event_type": "ws_ticker_message",
            "payload": {
                "market_ticker": "KXBTC15M-TEST-00",
                "time": "2026-03-28T12:00:00Z",
                "price_dollars": "0.55",
                "yes_bid_dollars": "0.54",
                "yes_ask_dollars": "0.56",
                "volume_fp": "12.00",
                "open_interest_fp": "3.00",
            },
        },
        {
            "event_type": "ws_trade_message",
            "payload": {
                "market_ticker": "KXBTC15M-TEST-00",
                "ts": int(datetime(2026, 3, 28, 12, 0, tzinfo=UTC).timestamp()),
                "trade_id": "123",
                "count_fp": "1.00",
                "yes_price_dollars": "0.55",
                "taker_side": "yes",
            },
        },
    ]
    with (raw_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row) + "\n")

    rows_written = downloader.materialize_live_quote_logs()
    output = tmp_path / "dataset" / "curated" / "live_quotes" / "production" / "2026-03-28.parquet"
    df = pd.read_parquet(output)

    assert rows_written == 2
    assert list(df["event_type"]) == ["ticker", "trade"]
    assert df.loc[0, "yes_bid_cents"] == 54
    assert df.loc[0, "yes_ask_cents"] == 56
    assert df.loc[1, "trade_id"] == "123"
