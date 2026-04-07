from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd

from src.indexers.kalshi.client import KalshiClient
from src.indexers.kalshi.models import Market, Trade, parse_count, parse_datetime, parse_price_cents

DEFAULT_DATA_ROOT = Path("output/kalshi_kxbtc15m_data")
DEFAULT_EXISTING_ROOTS = (
    Path("output/kalshi_series"),
    Path("output/kalshi_series_backfill"),
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_parse_datetime(value: Any) -> datetime | None:
    if value in (None, "", pd.NaT):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        return parse_datetime(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.to_pydatetime().replace(tzinfo=UTC)
        return value.to_pydatetime().astimezone(UTC)
    return None


def _to_unix_seconds(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp())


def _series_from_ticker(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def _event_from_ticker(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0]


def _price_dollars_to_cents(value: Any) -> int | None:
    return parse_price_cents(value)


def _count_fp_to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(Decimal(str(value)))


def _merge_values(left: Any, right: Any) -> Any:
    return right if right not in (None, "", []) else left


class SharedRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.requests_per_second = max(0.01, requests_per_second)
        self._interval_seconds = 1.0 / self.requests_per_second
        self._lock = threading.Lock()
        self._last_request_monotonic = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_monotonic
            delay = self._interval_seconds - elapsed
            if delay > 0:
                time.sleep(delay)
            self._last_request_monotonic = time.monotonic()


@dataclass
class HistoricalCutoff:
    market_settled_time: datetime
    trades_created_time: datetime
    orders_updated_time: datetime

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HistoricalCutoff:
        return cls(
            market_settled_time=parse_datetime(payload["market_settled_ts"]),
            trades_created_time=parse_datetime(payload["trades_created_ts"]),
            orders_updated_time=parse_datetime(payload["orders_updated_ts"]),
        )


@dataclass
class TickerSyncResult:
    ticker: str
    manifest_entry: dict[str, Any]
    market_record: dict[str, Any] | None
    trade_rows: int
    candle_rows: int
    live_quote_rows: int
    state_update: dict[str, Any]


def market_to_record(market: Market, *, source: str, fetched_at: datetime) -> dict[str, Any]:
    return {
        "ticker": market.ticker,
        "event_ticker": market.event_ticker or _event_from_ticker(market.ticker),
        "series_ticker": _series_from_ticker(market.ticker),
        "market_type": market.market_type,
        "title": market.title,
        "yes_sub_title": market.yes_sub_title,
        "no_sub_title": market.no_sub_title,
        "status": market.status,
        "result": market.result,
        "created_time": market.created_time,
        "open_time": market.open_time,
        "close_time": market.close_time,
        "yes_bid_cents": market.yes_bid,
        "yes_ask_cents": market.yes_ask,
        "no_bid_cents": market.no_bid,
        "no_ask_cents": market.no_ask,
        "last_price_cents": market.last_price,
        "volume": market.volume,
        "volume_24h": market.volume_24h,
        "open_interest": market.open_interest,
        "metadata_source": source,
        "metadata_fetched_at": fetched_at,
    }


def trade_to_record(trade: Trade, *, source: str, fetched_at: datetime) -> dict[str, Any]:
    return {
        "trade_id": trade.trade_id,
        "ticker": trade.ticker,
        "event_ticker": _event_from_ticker(trade.ticker),
        "series_ticker": _series_from_ticker(trade.ticker),
        "count": trade.count,
        "yes_price_cents": trade.yes_price,
        "no_price_cents": trade.no_price,
        "taker_side": trade.taker_side,
        "created_time": trade.created_time,
        "source": source,
        "fetched_at": fetched_at,
    }


def _normalize_bid_ask_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_cents": _price_dollars_to_cents(payload.get("open_dollars", payload.get("open"))),
        "low_cents": _price_dollars_to_cents(payload.get("low_dollars", payload.get("low"))),
        "high_cents": _price_dollars_to_cents(payload.get("high_dollars", payload.get("high"))),
        "close_cents": _price_dollars_to_cents(payload.get("close_dollars", payload.get("close"))),
    }


def _normalize_price_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_cents": _price_dollars_to_cents(payload.get("open_dollars", payload.get("open"))),
        "low_cents": _price_dollars_to_cents(payload.get("low_dollars", payload.get("low"))),
        "high_cents": _price_dollars_to_cents(payload.get("high_dollars", payload.get("high"))),
        "close_cents": _price_dollars_to_cents(payload.get("close_dollars", payload.get("close"))),
        "mean_cents": _price_dollars_to_cents(payload.get("mean_dollars", payload.get("mean"))),
        "previous_cents": _price_dollars_to_cents(payload.get("previous_dollars", payload.get("previous"))),
    }


def candle_payload_to_records(
    payload: dict[str, Any],
    *,
    ticker: str,
    series_ticker: str,
    source: str,
    fetched_at: datetime,
    period_interval_minutes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candle in payload.get("candlesticks", []):
        yes_bid = _normalize_bid_ask_distribution(candle.get("yes_bid", {}))
        yes_ask = _normalize_bid_ask_distribution(candle.get("yes_ask", {}))
        price = _normalize_price_distribution(candle.get("price", {}))
        rows.append(
            {
                "ticker": ticker,
                "event_ticker": _event_from_ticker(ticker),
                "series_ticker": series_ticker,
                "period_interval_minutes": period_interval_minutes,
                "end_period_time": datetime.fromtimestamp(int(candle["end_period_ts"]), tz=UTC),
                "yes_bid_open_cents": yes_bid["open_cents"],
                "yes_bid_low_cents": yes_bid["low_cents"],
                "yes_bid_high_cents": yes_bid["high_cents"],
                "yes_bid_close_cents": yes_bid["close_cents"],
                "yes_ask_open_cents": yes_ask["open_cents"],
                "yes_ask_low_cents": yes_ask["low_cents"],
                "yes_ask_high_cents": yes_ask["high_cents"],
                "yes_ask_close_cents": yes_ask["close_cents"],
                "price_open_cents": price["open_cents"],
                "price_low_cents": price["low_cents"],
                "price_high_cents": price["high_cents"],
                "price_close_cents": price["close_cents"],
                "price_mean_cents": price["mean_cents"],
                "price_previous_cents": price["previous_cents"],
                "volume": _count_fp_to_int(candle.get("volume_fp", candle.get("volume"))),
                "open_interest": _count_fp_to_int(candle.get("open_interest_fp", candle.get("open_interest"))),
                "source": source,
                "fetched_at": fetched_at,
            }
        )
    return rows


class KXBTC15MQuoteAwareDataDownloader:
    def __init__(
        self,
        *,
        root_dir: Path = DEFAULT_DATA_ROOT,
        series_ticker: str = "KXBTC15M",
        existing_roots: tuple[Path, ...] = DEFAULT_EXISTING_ROOTS,
        client_factory: Callable[[], KalshiClient] | None = None,
        requests_per_second: float = 6.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.series_ticker = series_ticker
        self.existing_roots = tuple(Path(root) for root in existing_roots)
        self._client_factory = client_factory or KalshiClient
        self._rate_limiter = SharedRateLimiter(requests_per_second)

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.root_dir / "curated"

    @property
    def manifest_dir(self) -> Path:
        return self.root_dir / "manifests"

    @property
    def manifest_path(self) -> Path:
        return self.manifest_dir / "ticker_universe.json"

    @property
    def sync_state_path(self) -> Path:
        return self.manifest_dir / "sync_state.json"

    @property
    def coverage_report_path(self) -> Path:
        return self.manifest_dir / "coverage_report.json"

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._rate_limiter.wait()
        return fn(*args, **kwargs)

    def load_manifest(self) -> list[dict[str, Any]]:
        return _read_json(self.manifest_path, [])

    def save_manifest(self, manifest: list[dict[str, Any]]) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")

    def load_sync_state(self) -> dict[str, Any]:
        return _read_json(
            self.sync_state_path,
            {
                "series_ticker": self.series_ticker,
                "updated_at": None,
                "tickers": {},
            },
        )

    def save_sync_state(self, sync_state: dict[str, Any]) -> None:
        sync_state["updated_at"] = utc_now().isoformat()
        _write_json(self.sync_state_path, sync_state)

    def fetch_historical_cutoff(self) -> HistoricalCutoff:
        with self._client_factory() as client:
            payload = self._call(client.get_historical_cutoff)
        return HistoricalCutoff.from_payload(payload)

    def build_ticker_universe(self) -> list[dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for entry in self.load_manifest():
            entries[entry["ticker"]] = dict(entry)

        for market_record in self._scan_existing_market_metadata():
            self._merge_manifest_entry(entries, market_record["ticker"], market_record, "local_market_archive")

        for ticker in self._scan_existing_trade_tickers():
            self._merge_manifest_entry(
                entries,
                ticker,
                {
                    "ticker": ticker,
                    "event_ticker": _event_from_ticker(ticker),
                    "series_ticker": self.series_ticker,
                },
                "local_trade_archive",
            )

        with self._client_factory() as client:
            for status in ("open", "settled"):
                pages = self._call(
                    lambda: list(
                        client.iter_markets_pages(
                            limit=1000,
                            series_ticker=self.series_ticker,
                            status=status,
                        )
                    )
                )
                for _, markets in pages:
                    for market in markets:
                        self._merge_manifest_entry(
                            entries,
                            market.ticker,
                            market_to_record(market, source=f"live_{status}", fetched_at=utc_now()),
                            f"live_{status}",
                        )

        manifest = sorted(entries.values(), key=lambda item: (item.get("open_time") or "", item["ticker"]))
        self.save_manifest(manifest)
        return manifest

    def sync(
        self,
        *,
        tickers: tuple[str, ...] | None = None,
        sync_markets: bool = True,
        sync_trades: bool = True,
        sync_candles: bool = True,
        materialize_live_quotes: bool = True,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        manifest = self.build_ticker_universe()
        requested = set(tickers or ())
        selected = [entry for entry in manifest if not requested or entry["ticker"] in requested]
        cutoff = self.fetch_historical_cutoff()
        sync_state = self.load_sync_state()
        market_records_by_ticker = self._load_existing_market_records()
        manifest_by_ticker = {entry["ticker"]: dict(entry) for entry in manifest}

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(
                    self._sync_single_ticker,
                    entry,
                    cutoff,
                    sync_markets,
                    sync_trades,
                    sync_candles,
                ): entry["ticker"]
                for entry in selected
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    sync_state.setdefault("tickers", {})[ticker] = {
                        "market": {"status": "error", "error": repr(exc), "updated_at": utc_now().isoformat()},
                        "trades": {"status": "error", "error": repr(exc), "updated_at": utc_now().isoformat()},
                        "candles_1m": {"status": "error", "error": repr(exc), "updated_at": utc_now().isoformat()},
                        "live_quote_capture": {"status": "unknown", "row_count": 0, "updated_at": utc_now().isoformat()},
                    }
                    continue
                if result.market_record is not None:
                    market_records_by_ticker[ticker] = result.market_record
                sync_state.setdefault("tickers", {})[ticker] = result.state_update
                manifest_by_ticker[ticker] = result.manifest_entry

        refreshed_manifest = sorted(manifest_by_ticker.values(), key=lambda item: (item.get("open_time") or "", item["ticker"]))
        self.save_manifest(refreshed_manifest)
        self.save_sync_state(sync_state)
        self._write_curated_markets(list(market_records_by_ticker.values()))
        if materialize_live_quotes:
            self.materialize_live_quote_logs()
        return self.generate_coverage_report()

    def materialize_live_quote_logs(self) -> int:
        raw_live_root = self.raw_dir / "live_ticker"
        if not raw_live_root.exists():
            return 0

        rows_written = 0
        for environment_dir in sorted(path for path in raw_live_root.iterdir() if path.is_dir()):
            for date_dir in sorted(path for path in environment_dir.iterdir() if path.is_dir()):
                events_path = date_dir / "events.jsonl"
                rows = self._normalize_live_quote_events(_read_jsonl(events_path))
                if not rows:
                    continue
                df = pd.DataFrame(rows).sort_values(["event_time", "ticker", "event_type"]).reset_index(drop=True)
                output_path = self.curated_dir / "live_quotes" / environment_dir.name / f"{date_dir.name}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)
                rows_written += len(df)
        return rows_written

    def generate_coverage_report(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        market_records = self._load_existing_market_records()
        live_quote_index = self._build_live_quote_index()
        ticker_reports: list[dict[str, Any]] = []
        for entry in manifest:
            ticker = entry["ticker"]
            trade_path = self.curated_dir / "trades" / f"{ticker}.parquet"
            candle_path = self.curated_dir / "candles_1m" / f"{ticker}.parquet"
            trade_rows = self._safe_parquet_row_count(trade_path)
            candle_rows = self._safe_parquet_row_count(candle_path)
            quote_info = live_quote_index.get(ticker, {"row_count": 0, "latest_event_time": None})
            ticker_reports.append(
                {
                    "ticker": ticker,
                    "event_ticker": entry.get("event_ticker"),
                    "status": entry.get("status"),
                    "open_time": entry.get("open_time"),
                    "close_time": entry.get("close_time"),
                    "market_metadata_ready": ticker in market_records,
                    "trade_rows": trade_rows,
                    "candle_rows": candle_rows,
                    "live_quote_rows": quote_info["row_count"],
                    "latest_live_quote_time": quote_info["latest_event_time"],
                }
            )

        report = {
            "generated_at": utc_now().isoformat(),
            "series_ticker": self.series_ticker,
            "summary": {
                "ticker_count": len(ticker_reports),
                "market_metadata_ready_count": sum(1 for item in ticker_reports if item["market_metadata_ready"]),
                "trade_ready_count": sum(1 for item in ticker_reports if item["trade_rows"] > 0),
                "candles_ready_count": sum(1 for item in ticker_reports if item["candle_rows"] > 0),
                "live_quote_ready_count": sum(1 for item in ticker_reports if item["live_quote_rows"] > 0),
            },
            "tickers": sorted(ticker_reports, key=lambda item: item["ticker"]),
        }
        _write_json(self.coverage_report_path, report)
        return report

    def _sync_single_ticker(
        self,
        entry: dict[str, Any],
        cutoff: HistoricalCutoff,
        sync_markets: bool,
        sync_trades: bool,
        sync_candles: bool,
    ) -> TickerSyncResult:
        ticker = entry["ticker"]
        state_update: dict[str, Any] = {}
        market_record: dict[str, Any] | None = None
        trade_rows = 0
        candle_rows = 0
        fetched_at = utc_now()
        updated_entry = dict(entry)

        with self._client_factory() as client:
            if sync_markets:
                market_record, market_state = self._sync_market_metadata(client, ticker, entry, cutoff, fetched_at)
                state_update["market"] = market_state
                if market_record is not None:
                    updated_entry = self._merge_manifest_values(updated_entry, market_record, market_state.get("source"))

            if sync_trades:
                trade_rows, trades_state = self._sync_trades(client, ticker, cutoff, fetched_at)
                state_update["trades"] = trades_state

            if sync_candles:
                candle_rows, candles_state = self._sync_candles(
                    client,
                    ticker,
                    updated_entry,
                    cutoff,
                    fetched_at,
                )
                state_update["candles_1m"] = candles_state

        state_update["live_quote_capture"] = {
            "status": "unknown",
            "row_count": 0,
            "updated_at": utc_now().isoformat(),
        }
        updated_entry["last_synced_at"] = utc_now().isoformat()
        return TickerSyncResult(
            ticker=ticker,
            manifest_entry=updated_entry,
            market_record=market_record,
            trade_rows=trade_rows,
            candle_rows=candle_rows,
            live_quote_rows=0,
            state_update=state_update,
        )

    def _sync_market_metadata(
        self,
        client: KalshiClient,
        ticker: str,
        entry: dict[str, Any],
        cutoff: HistoricalCutoff,
        fetched_at: datetime,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        close_time = _safe_parse_datetime(entry.get("close_time"))
        preferred_sources = ["live", "historical"]
        if close_time is not None and close_time < cutoff.market_settled_time:
            preferred_sources = ["historical", "live"]
        last_error: str | None = None

        for source in preferred_sources:
            try:
                market = self._call(client.get_market, ticker) if source == "live" else self._call(client.get_historical_market, ticker)
                _write_json(
                    self.raw_dir / "markets" / f"{ticker}.json",
                    {
                        "ticker": ticker,
                        "fetched_at": fetched_at.isoformat(),
                        "source": source,
                        "market": asdict(market),
                    },
                )
                return (
                    market_to_record(market, source=source, fetched_at=fetched_at),
                    {
                        "status": "ok",
                        "source": source,
                        "updated_at": fetched_at.isoformat(),
                        "error": None,
                    },
                )
            except httpx.HTTPStatusError as exc:
                last_error = f"{source}:{exc.response.status_code}"
                if exc.response.status_code != 404:
                    raise

        return (
            None,
            {
                "status": "missing",
                "source": None,
                "updated_at": fetched_at.isoformat(),
                "error": last_error,
            },
        )

    def _sync_trades(
        self,
        client: KalshiClient,
        ticker: str,
        cutoff: HistoricalCutoff,
        fetched_at: datetime,
    ) -> tuple[int, dict[str, Any]]:
        cutoff_ts = _to_unix_seconds(cutoff.trades_created_time)
        historical_payloads: list[dict[str, Any]] = []
        live_payloads: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []

        historical_pages = self._call(
            lambda: list(
                client.iter_historical_trades_pages(
                    ticker=ticker,
                    limit=1000,
                    max_ts=cutoff_ts - 1 if cutoff_ts is not None else None,
                )
            )
        )
        for payload, trades in historical_pages:
            historical_payloads.append(payload)
            all_rows.extend(trade_to_record(trade, source="historical", fetched_at=fetched_at) for trade in trades)

        live_pages = self._call(
            lambda: list(
                client.iter_market_trades_pages(
                    ticker=ticker,
                    limit=1000,
                    min_ts=cutoff_ts,
                )
            )
        )
        for payload, trades in live_pages:
            live_payloads.append(payload)
            all_rows.extend(trade_to_record(trade, source="live", fetched_at=fetched_at) for trade in trades)

        _write_json(
            self.raw_dir / "trades" / f"{ticker}.json",
            {
                "ticker": ticker,
                "fetched_at": fetched_at.isoformat(),
                "historical_pages": historical_payloads,
                "live_pages": live_payloads,
            },
        )

        output_path = self.curated_dir / "trades" / f"{ticker}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not all_rows:
            pd.DataFrame(
                columns=[
                    "trade_id",
                    "ticker",
                    "event_ticker",
                    "series_ticker",
                    "count",
                    "yes_price_cents",
                    "no_price_cents",
                    "taker_side",
                    "created_time",
                    "source",
                    "fetched_at",
                ]
            ).to_parquet(output_path, index=False)
            return 0, {"status": "ok", "row_count": 0, "updated_at": fetched_at.isoformat(), "error": None}

        df = (
            pd.DataFrame(all_rows)
            .sort_values(["created_time", "trade_id", "source"])
            .drop_duplicates(subset=["trade_id"], keep="last")
            .reset_index(drop=True)
        )
        df.to_parquet(output_path, index=False)
        return len(df), {"status": "ok", "row_count": len(df), "updated_at": fetched_at.isoformat(), "error": None}

    def _sync_candles(
        self,
        client: KalshiClient,
        ticker: str,
        entry: dict[str, Any],
        cutoff: HistoricalCutoff,
        fetched_at: datetime,
    ) -> tuple[int, dict[str, Any]]:
        open_time = _safe_parse_datetime(entry.get("open_time"))
        close_time = _safe_parse_datetime(entry.get("close_time"))
        if open_time is None or close_time is None:
            return 0, {
                "status": "missing_market_times",
                "row_count": 0,
                "updated_at": fetched_at.isoformat(),
                "error": "open_time_or_close_time_missing",
            }

        start_ts = _to_unix_seconds(open_time)
        end_ts = _to_unix_seconds(close_time)
        if start_ts is None or end_ts is None or end_ts < start_ts:
            return 0, {
                "status": "invalid_time_range",
                "row_count": 0,
                "updated_at": fetched_at.isoformat(),
                "error": "invalid_market_time_range",
            }

        preferred_sources = ["live", "historical"]
        if close_time < cutoff.market_settled_time:
            preferred_sources = ["historical", "live"]

        payload: dict[str, Any] | None = None
        source_used: str | None = None
        last_error: str | None = None
        for source in preferred_sources:
            try:
                if source == "live":
                    payload = self._call(
                        client.get_market_candlesticks,
                        series_ticker=self.series_ticker,
                        ticker=ticker,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        period_interval=1,
                    )
                else:
                    payload = self._call(
                        client.get_historical_market_candlesticks,
                        ticker=ticker,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        period_interval=1,
                    )
                source_used = source
                break
            except httpx.HTTPStatusError as exc:
                last_error = f"{source}:{exc.response.status_code}"
                if exc.response.status_code != 404:
                    raise

        if payload is None or source_used is None:
            return 0, {
                "status": "missing",
                "row_count": 0,
                "updated_at": fetched_at.isoformat(),
                "error": last_error,
            }

        _write_json(
            self.raw_dir / "candles_1m" / f"{ticker}.json",
            {
                "ticker": ticker,
                "fetched_at": fetched_at.isoformat(),
                "source": source_used,
                "payload": payload,
            },
        )
        rows = candle_payload_to_records(
            payload,
            ticker=ticker,
            series_ticker=self.series_ticker,
            source=source_used,
            fetched_at=fetched_at,
            period_interval_minutes=1,
        )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["end_period_time"]).reset_index(drop=True)
        output_path = self.curated_dir / "candles_1m" / f"{ticker}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        return len(df), {
            "status": "ok",
            "row_count": len(df),
            "source": source_used,
            "updated_at": fetched_at.isoformat(),
            "error": None,
        }

    def _scan_existing_market_metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for root in self.existing_roots:
            path = root / f"{self.series_ticker}_markets.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if "ticker" not in df.columns:
                continue
            for record in df.to_dict(orient="records"):
                ticker = record.get("ticker")
                if not ticker:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "event_ticker": record.get("event_ticker") or _event_from_ticker(ticker),
                        "series_ticker": record.get("series_ticker") or self.series_ticker,
                        "status": record.get("status"),
                        "open_time": _safe_parse_datetime(record.get("open_time")),
                        "close_time": _safe_parse_datetime(record.get("close_time")),
                        "result": record.get("result"),
                        "first_seen_source": "local_market_archive",
                    }
                )
        return rows

    def _scan_existing_trade_tickers(self) -> set[str]:
        tickers: set[str] = set()
        for root in self.existing_roots:
            trades_dir = root / self.series_ticker / "trades"
            if not trades_dir.exists():
                continue
            for parquet_path in trades_dir.glob("*.parquet"):
                tickers.add(parquet_path.stem)
        return tickers

    def _merge_manifest_entry(
        self,
        entries: dict[str, dict[str, Any]],
        ticker: str,
        payload: dict[str, Any],
        source: str,
    ) -> None:
        existing = dict(entries.get(ticker, {}))
        entries[ticker] = self._merge_manifest_values(existing, payload, source)

    def _merge_manifest_values(
        self,
        existing: dict[str, Any],
        payload: dict[str, Any],
        source: str | None,
    ) -> dict[str, Any]:
        ticker = payload.get("ticker") or existing.get("ticker")
        merged = dict(existing)
        merged["ticker"] = ticker
        merged["event_ticker"] = _merge_values(existing.get("event_ticker"), payload.get("event_ticker")) or _event_from_ticker(
            ticker
        )
        merged["series_ticker"] = _merge_values(existing.get("series_ticker"), payload.get("series_ticker")) or self.series_ticker
        merged["status"] = _merge_values(existing.get("status"), payload.get("status"))
        open_time = _safe_parse_datetime(payload.get("open_time")) or _safe_parse_datetime(existing.get("open_time"))
        close_time = _safe_parse_datetime(payload.get("close_time")) or _safe_parse_datetime(existing.get("close_time"))
        merged["open_time"] = open_time.isoformat() if open_time else None
        merged["close_time"] = close_time.isoformat() if close_time else None
        merged["result"] = _merge_values(existing.get("result"), payload.get("result"))
        merged["first_seen_source"] = existing.get("first_seen_source") or source
        merged["last_synced_source"] = source or existing.get("last_synced_source")
        merged["last_synced_at"] = utc_now().isoformat()
        return merged

    def _load_existing_market_records(self) -> dict[str, dict[str, Any]]:
        path = self.curated_dir / "markets" / f"{self.series_ticker}_markets.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return {row["ticker"]: row for row in df.to_dict(orient="records")}

    def _write_curated_markets(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values(["close_time", "ticker"]).reset_index(drop=True)
        output_path = self.curated_dir / "markets" / f"{self.series_ticker}_markets.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    def _normalize_live_quote_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events:
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            if event_type not in {"ws_ticker_message", "ws_trade_message"}:
                continue
            if event_type == "ws_ticker_message":
                event_time = (
                    parse_datetime(payload["time"])
                    if payload.get("time")
                    else datetime.fromtimestamp(int(payload["ts"]), tz=UTC)
                )
                rows.append(
                    {
                        "event_type": "ticker",
                        "ticker": payload.get("market_ticker"),
                        "event_time": event_time,
                        "last_price_cents": _price_dollars_to_cents(payload.get("price_dollars", payload.get("price"))),
                        "yes_bid_cents": _price_dollars_to_cents(payload.get("yes_bid_dollars", payload.get("yes_bid"))),
                        "yes_ask_cents": _price_dollars_to_cents(payload.get("yes_ask_dollars", payload.get("yes_ask"))),
                        "volume": _count_fp_to_int(payload.get("volume_fp", payload.get("volume"))),
                        "open_interest": _count_fp_to_int(payload.get("open_interest_fp", payload.get("open_interest"))),
                        "dollar_volume": payload.get("dollar_volume"),
                        "dollar_open_interest": payload.get("dollar_open_interest"),
                        "trade_id": None,
                        "count": None,
                        "taker_side": None,
                    }
                )
            else:
                event_time = datetime.fromtimestamp(int(payload["ts"]), tz=UTC)
                rows.append(
                    {
                        "event_type": "trade",
                        "ticker": payload.get("market_ticker"),
                        "event_time": event_time,
                        "last_price_cents": _price_dollars_to_cents(
                            payload.get("yes_price_dollars", payload.get("yes_price"))
                        ),
                        "yes_bid_cents": None,
                        "yes_ask_cents": None,
                        "volume": None,
                        "open_interest": None,
                        "dollar_volume": None,
                        "dollar_open_interest": None,
                        "trade_id": payload.get("trade_id"),
                        "count": _count_fp_to_int(payload.get("count_fp", payload.get("count"))),
                        "taker_side": payload.get("taker_side"),
                    }
                )
        return rows

    def _build_live_quote_index(self) -> dict[str, dict[str, Any]]:
        live_quotes_dir = self.curated_dir / "live_quotes"
        if not live_quotes_dir.exists():
            return {}
        index: dict[str, dict[str, Any]] = {}
        for parquet_path in live_quotes_dir.rglob("*.parquet"):
            df = pd.read_parquet(parquet_path, columns=["ticker", "event_time"])
            if df.empty:
                continue
            grouped = df.groupby("ticker")["event_time"].agg(["count", "max"]).reset_index()
            for row in grouped.to_dict(orient="records"):
                ticker = row["ticker"]
                latest = row["max"]
                if isinstance(latest, pd.Timestamp):
                    latest = latest.to_pydatetime()
                existing = index.get(ticker)
                latest_value = latest.isoformat() if isinstance(latest, datetime) else str(latest)
                if existing is None:
                    index[ticker] = {"row_count": int(row["count"]), "latest_event_time": latest_value}
                    continue
                existing["row_count"] += int(row["count"])
                if existing["latest_event_time"] is None or latest_value > existing["latest_event_time"]:
                    existing["latest_event_time"] = latest_value
        return index

    @staticmethod
    def _safe_parquet_row_count(path: Path) -> int:
        if not path.exists():
            return 0
        return len(pd.read_parquet(path))
