from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.live.data.btc_spot_feed import BTCSpotUpdate

LOGGER = logging.getLogger(__name__)

COINBASE_TRADES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/trades"
KRAKEN_TRADES_URL = "https://api.kraken.com/0/public/Trades"
DEFAULT_OUTPUT_ROOT = Path("output") / "kalshi_kxbtc15m_data" / "curated"
DEFAULT_RAW_CACHE_ROOT = Path("output") / "kalshi_kxbtc15m_data" / "raw" / "external_spot_public_trades"
DEFAULT_ENVIRONMENT = "backfill"
DEFAULT_MARKETS_CANDIDATES = (
    Path("output") / "kalshi_series_backfill" / "KXBTC15M_markets.parquet",
    Path("output") / "kalshi_series" / "KXBTC15M_markets.parquet",
)
_NORMALIZED_SPOT_COLUMNS = tuple(field.name for field in fields(BTCSpotUpdate))


@dataclass(frozen=True)
class HistoricalBTCSpotBackfillConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    raw_cache_root: Path = DEFAULT_RAW_CACHE_ROOT
    environment: str = DEFAULT_ENVIRONMENT
    venues: tuple[str, ...] = ("coinbase", "kraken")
    freshness_bound_ms: int = 500
    rolling_median_window_seconds: int = 2
    twap_window_seconds: int = 60
    history_window_seconds: int = 31 * 60
    venue_divergence_bps_gate: float = 25.0
    latency_gamma_shape: float = 2.0
    latency_gamma_scale_ms: float = 50.0
    max_latency_ms: int = 500
    requests_per_second: float = 3.0
    http_timeout_seconds: float = 30.0
    random_seed: int = 42


@dataclass(frozen=True)
class BackfillDateResult:
    date: date
    row_count: int
    skipped: bool


def normalized_spot_column_order() -> tuple[str, ...]:
    return _NORMALIZED_SPOT_COLUMNS


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _ensure_trade_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "event_time" not in frame.columns or "price" not in frame.columns:
        raise ValueError("Trade frame must include event_time and price columns.")
    if "size" not in frame.columns:
        frame["size"] = 0.0
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["event_time", "price"]).sort_values("event_time").reset_index(drop=True)
    return frame


def build_venue_proxy_frame(trades: pd.DataFrame, *, venue: str, window_seconds: int = 2) -> pd.DataFrame:
    frame = _ensure_trade_frame_columns(trades)
    if frame.empty:
        return pd.DataFrame(
            columns=["venue", "event_time", "price", "size", "proxy_mid", "rolling_notional_2s"]
        )
    indexed = frame.set_index("event_time")
    notional = indexed["price"] * indexed["size"]
    rolling_window = f"{int(window_seconds)}s"
    proxy_mid = indexed["price"].rolling(rolling_window).median()
    rolling_notional = notional.rolling(rolling_window).sum()
    proxied = frame.copy()
    proxied["venue"] = venue
    proxied["proxy_mid"] = proxy_mid.to_numpy(dtype=np.float64)
    proxied["rolling_notional_2s"] = rolling_notional.to_numpy(dtype=np.float64)
    proxied["rolling_notional_2s"] = proxied["rolling_notional_2s"].fillna(0.0)
    return proxied


def _venue_rng(config: HistoricalBTCSpotBackfillConfig, venue: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{int(config.random_seed)}:{venue}".encode("utf-8")).digest()
    venue_seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)
    return np.random.default_rng(venue_seed)


def _apply_latency(
    frame: pd.DataFrame,
    *,
    venue: str,
    config: HistoricalBTCSpotBackfillConfig,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(received_at=pd.Series(dtype="datetime64[ns, UTC]"), latency_ms=pd.Series(dtype=float))
    enriched = frame.copy()
    rng = _venue_rng(config, venue)
    latencies = rng.gamma(config.latency_gamma_shape, config.latency_gamma_scale_ms, len(enriched))
    latencies = np.clip(latencies, 0.0, float(config.max_latency_ms))
    enriched["latency_ms"] = latencies.astype(np.float64)
    enriched["received_at"] = enriched["event_time"] + pd.to_timedelta(enriched["latency_ms"], unit="ms")
    return enriched


class _ResampledMetricState:
    def __init__(self, *, history_window_seconds: int, twap_window_seconds: int):
        self.history_window_seconds = int(history_window_seconds)
        self.twap_window_seconds = int(twap_window_seconds)
        self._buckets: dict[datetime, float] = {}

    def record(self, event_time: datetime, price: float) -> None:
        bucket = event_time.astimezone(UTC).replace(microsecond=0)
        self._buckets[bucket] = float(price)
        cutoff = bucket - timedelta(seconds=self.history_window_seconds)
        stale_keys = [timestamp for timestamp in self._buckets if timestamp < cutoff]
        for timestamp in stale_keys:
            self._buckets.pop(timestamp, None)

    def compute(self, reference_time: datetime) -> dict[str, float | None]:
        points = sorted((timestamp, price) for timestamp, price in self._buckets.items() if timestamp <= reference_time and price > 0.0)
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
        timestamps = [timestamp for timestamp, _price in points]
        prices = np.asarray([price for _timestamp, price in points], dtype=np.float64)
        log_returns = np.diff(np.log(prices)) if len(prices) > 1 else np.asarray([], dtype=np.float64)
        annualizer = math.sqrt(365.0 * 24.0 * 60.0 * 60.0)

        def latest_price_before(cutoff_seconds: int) -> float | None:
            cutoff_time = reference_time - timedelta(seconds=cutoff_seconds)
            eligible = [price for timestamp, price in points if timestamp <= cutoff_time]
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
            weights = np.asarray(
                [
                    math.exp(math.log(0.5) * max((reference_time - timestamp).total_seconds(), 0.0) / half_life)
                    for timestamp in timestamps[1:]
                ],
                dtype=np.float64,
            )
            if weights.sum() <= 0.0:
                return 0.0
            variance = float(np.dot(weights, np.square(log_returns)) / weights.sum())
            return float(math.sqrt(max(variance, 0.0)) * annualizer)

        twap_cutoff = reference_time - timedelta(seconds=self.twap_window_seconds)
        twap_prices = [price for timestamp, price in points if timestamp >= twap_cutoff]
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


def _weighted_consolidated_price(rows: list[pd.Series]) -> float:
    if len(rows) == 1:
        return float(rows[0]["proxy_mid"])
    weights = [max(float(row.get("rolling_notional_2s", 0.0) or 0.0), 1e-9) for row in rows]
    prices = [float(row["proxy_mid"]) for row in rows]
    return float(np.average(prices, weights=weights))


def _daterange(start_date: date, end_date: date) -> Iterator[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, dtime.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


class HistoricalBTCSpotBackfillRunner:
    def __init__(self, config: HistoricalBTCSpotBackfillConfig | None = None):
        self.config = config or HistoricalBTCSpotBackfillConfig()

    def output_path_for_date(self, day: date) -> Path:
        return (
            Path(self.config.output_root)
            / "external_spot_normalized"
            / self.config.environment
            / day.isoformat()
            / "events.parquet"
        )

    def build_date_range(
        self,
        *,
        start_date: date,
        end_date: date,
        trade_loader: Callable[[str, datetime, datetime], pd.DataFrame],
        overwrite: bool = False,
    ) -> list[BackfillDateResult]:
        results: list[BackfillDateResult] = []
        for day in _daterange(start_date, end_date):
            output_path = self.output_path_for_date(day)
            if output_path.exists() and not overwrite:
                results.append(BackfillDateResult(date=day, row_count=0, skipped=True))
                continue
            day_start, day_end = _day_bounds(day)
            window_start = day_start - timedelta(seconds=self.config.history_window_seconds)
            venue_frames: dict[str, pd.DataFrame] = {}
            for venue in self.config.venues:
                trades = trade_loader(venue, window_start, day_end)
                venue_frames[venue] = _apply_latency(
                    build_venue_proxy_frame(
                        trades,
                        venue=venue,
                        window_seconds=self.config.rolling_median_window_seconds,
                    ),
                    venue=venue,
                    config=self.config,
                )
            frame = self._build_normalized_frame(
                venue_frames=venue_frames,
                day_start=day_start,
                day_end=day_end,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output_path, index=False)
            results.append(BackfillDateResult(date=day, row_count=len(frame), skipped=False))
        return results

    def _build_normalized_frame(
        self,
        *,
        venue_frames: dict[str, pd.DataFrame],
        day_start: datetime,
        day_end: datetime,
    ) -> pd.DataFrame:
        event_frames = []
        for venue, frame in venue_frames.items():
            if frame.empty:
                continue
            selected = frame.loc[:, ["venue", "event_time", "received_at", "proxy_mid", "rolling_notional_2s", "latency_ms"]].copy()
            selected["venue"] = venue
            event_frames.append(selected)
        if not event_frames:
            return pd.DataFrame(columns=normalized_spot_column_order())
        events = pd.concat(event_frames, ignore_index=True)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        events["received_at"] = pd.to_datetime(events["received_at"], utc=True)
        events = events.sort_values(["received_at", "event_time", "venue"]).reset_index(drop=True)
        latest_by_venue: dict[str, pd.Series] = {}
        metrics = _ResampledMetricState(
            history_window_seconds=self.config.history_window_seconds,
            twap_window_seconds=self.config.twap_window_seconds,
        )
        rows: list[dict[str, object]] = []
        for row in events.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            latest_by_venue[str(row_series["venue"])] = row_series
            reference_received = _utc_timestamp(row_series["received_at"])
            fresh_rows = [
                candidate
                for candidate in latest_by_venue.values()
                if 0.0
                <= (reference_received - _utc_timestamp(candidate["received_at"])).total_seconds() * 1000.0
                <= float(self.config.freshness_bound_ms)
            ]
            if not fresh_rows:
                continue
            venue_divergence_bps = 0.0
            selected_rows = list(fresh_rows)
            if len(fresh_rows) >= 2:
                mids = [float(candidate["proxy_mid"]) for candidate in fresh_rows]
                midpoint = (max(mids) + min(mids)) / 2.0
                if midpoint > 0.0:
                    venue_divergence_bps = abs(max(mids) - min(mids)) / midpoint * 10000.0
                if venue_divergence_bps > float(self.config.venue_divergence_bps_gate):
                    selected_rows = [
                        max(
                            fresh_rows,
                            key=lambda candidate: (
                                float(candidate.get("rolling_notional_2s", 0.0) or 0.0),
                                _utc_timestamp(candidate["event_time"]),
                                str(candidate["venue"]),
                            ),
                        )
                    ]
            consolidated_price = _weighted_consolidated_price(selected_rows)
            reference_event_time = max(_utc_timestamp(candidate["event_time"]) for candidate in selected_rows)
            reference_event_time_dt = reference_event_time.to_pydatetime()
            if not (day_start <= reference_event_time_dt < day_end):
                metrics.record(reference_event_time_dt, consolidated_price)
                continue
            metrics.record(reference_event_time_dt, consolidated_price)
            metric_values = metrics.compute(reference_event_time_dt)
            age_ms = max(0.0, (reference_received - reference_event_time).total_seconds() * 1000.0)
            if age_ms > float(self.config.freshness_bound_ms):
                continue
            rows.append(
                {
                    "event_time": reference_event_time_dt,
                    "received_at": reference_received,
                    "btc_spot_price": consolidated_price,
                    "btc_spot_twap_60s": metric_values["twap_60s"],
                    "btc_spot_age_ms": age_ms,
                    "btc_spot_is_fresh": True,
                    "btc_spot_venues_fresh": int(len(fresh_rows)),
                    "btc_spot_venue_divergence_bps": float(venue_divergence_bps),
                    "btc_vol_effective_sample_size": float(metric_values["effective_sample_size"] or 0.0),
                    "btc_spot_source": "trade_proxy",
                    "btc_spot_return_30s": metric_values["return_30s"],
                    "btc_spot_return_120s": metric_values["return_120s"],
                    "btc_spot_return_300s": metric_values["return_300s"],
                    "btc_spot_return_900s": metric_values["return_900s"],
                    "btc_spot_vol_120s": metric_values["vol_120s"],
                    "btc_spot_vol_300s": metric_values["vol_300s"],
                    "btc_spot_vol_900s": metric_values["vol_900s"],
                    "btc_spot_vol_1800s": metric_values["vol_1800s"],
                    "btc_spot_vol_ewma_hl300": metric_values["vol_ewma_hl300"],
                }
            )
        frame = pd.DataFrame(rows, columns=normalized_spot_column_order())
        if frame.empty:
            return frame
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, format="ISO8601")
        frame["received_at"] = pd.to_datetime(frame["received_at"], utc=True)
        return frame.sort_values(["received_at", "event_time"]).reset_index(drop=True)


class _RawTradeCache:
    def __init__(self, config: HistoricalBTCSpotBackfillConfig):
        self.config = config
        self.session = requests.Session()

    def load_window(self, venue: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        self.ensure_range(venue, start_time, end_time)
        return self._read_cached_window(venue, start_time, end_time)

    def ensure_range(self, venue: str, start_time: datetime, end_time: datetime) -> None:
        venue = str(venue).strip().lower()
        if venue == "coinbase":
            self._ensure_coinbase_range(start_time, end_time)
            return
        if venue == "kraken":
            self._ensure_kraken_range(start_time, end_time)
            return
        raise ValueError(f"Unsupported venue: {venue}")

    def _ensure_coinbase_range(self, start_time: datetime, end_time: datetime) -> None:
        checkpoint_path = self._checkpoint_path("coinbase")
        checkpoint = self._read_checkpoint(checkpoint_path)
        oldest_fetched_time = _parse_iso_datetime(checkpoint.get("oldest_fetched_time")) if checkpoint else None
        if oldest_fetched_time is not None and oldest_fetched_time <= start_time:
            return
        cursor = checkpoint.get("after_cursor") if checkpoint else None
        min_spacing = 1.0 / max(float(self.config.requests_per_second), 0.1)
        while True:
            params: dict[str, object] = {"limit": 1000}
            if cursor:
                params["after"] = cursor
            response = self.session.get(COINBASE_TRADES_URL, params=params, timeout=self.config.http_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            rows = [
                {
                    "trade_id": item.get("trade_id"),
                    "event_time": item.get("time"),
                    "price": item.get("price"),
                    "size": item.get("size"),
                    "side": item.get("side"),
                }
                for item in payload
                if item.get("time") is not None
            ]
            if rows:
                self._append_trade_rows("coinbase", rows, upper_bound=end_time)
                oldest_in_page = min(_utc_timestamp(row["event_time"]).to_pydatetime() for row in rows)
            else:
                oldest_in_page = None
            cursor = response.headers.get("cb-after")
            self._write_checkpoint(
                checkpoint_path,
                {
                    "after_cursor": cursor,
                    "oldest_fetched_time": oldest_in_page.isoformat() if oldest_in_page is not None else None,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            if oldest_in_page is None or oldest_in_page <= start_time or not cursor:
                break
            time.sleep(min_spacing)

    def _ensure_kraken_range(self, start_time: datetime, end_time: datetime) -> None:
        checkpoint_path = self._checkpoint_path("kraken")
        checkpoint = self._read_checkpoint(checkpoint_path)
        latest_fetched_time = _parse_iso_datetime(checkpoint.get("latest_fetched_time")) if checkpoint else None
        if latest_fetched_time is not None and latest_fetched_time >= end_time:
            return
        since = checkpoint.get("last_since_token") if checkpoint else None
        if since is None:
            since = str(int(start_time.timestamp() * 1_000_000_000))
        min_spacing = 1.0 / max(float(self.config.requests_per_second), 0.1)
        while True:
            response = self.session.get(
                KRAKEN_TRADES_URL,
                params={"pair": "XBTUSD", "since": since},
                timeout=self.config.http_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("error", [])
            if errors:
                raise RuntimeError(f"Kraken public trades request failed: {errors}")
            result = payload.get("result", {})
            pair_key = next((key for key in result.keys() if key != "last"), None)
            if pair_key is None:
                break
            rows_payload = result.get(pair_key) or []
            rows = [
                {
                    "trade_id": item[6] if len(item) > 6 else None,
                    "event_time": datetime.fromtimestamp(float(item[2]), tz=UTC).isoformat(),
                    "price": item[0],
                    "size": item[1],
                    "side": item[3],
                }
                for item in rows_payload
                if len(item) >= 3
            ]
            if rows:
                self._append_trade_rows("kraken", rows, upper_bound=end_time)
                latest_in_page = max(_utc_timestamp(row["event_time"]).to_pydatetime() for row in rows)
            else:
                latest_in_page = None
            next_since = str(result.get("last") or since)
            self._write_checkpoint(
                checkpoint_path,
                {
                    "last_since_token": next_since,
                    "latest_fetched_time": latest_in_page.isoformat() if latest_in_page is not None else None,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            if latest_in_page is None or latest_in_page >= end_time or next_since == since:
                break
            since = next_since
            time.sleep(min_spacing)

    def _append_trade_rows(self, venue: str, rows: Iterable[dict[str, object]], *, upper_bound: datetime) -> None:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            event_time = _utc_timestamp(row["event_time"]).to_pydatetime()
            if event_time >= upper_bound:
                continue
            grouped.setdefault(event_time.date().isoformat(), []).append(
                {
                    "trade_id": row.get("trade_id"),
                    "event_time": event_time.isoformat(),
                    "price": row.get("price"),
                    "size": row.get("size"),
                    "side": row.get("side"),
                }
            )
        for day_key, day_rows in grouped.items():
            path = self._raw_path(venue, day_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for row in day_rows:
                    handle.write(json.dumps(row, default=str) + "\n")

    def _read_cached_window(self, venue: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_day = start_time.date()
        end_day = end_time.date()
        for day in _daterange(start_day, end_day):
            path = self._raw_path(venue, day.isoformat())
            if not path.exists():
                continue
            rows = _read_jsonl_objects(path)
            if rows:
                frames.append(pd.DataFrame(rows))
        if not frames:
            return pd.DataFrame(columns=["trade_id", "event_time", "price", "size", "side"])
        frame = pd.concat(frames, ignore_index=True)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, format="ISO8601")
        frame = frame[(frame["event_time"] >= pd.Timestamp(start_time)) & (frame["event_time"] < pd.Timestamp(end_time))]
        if "trade_id" in frame.columns:
            frame = frame.drop_duplicates(subset=["trade_id"], keep="last")
        return frame.sort_values("event_time").reset_index(drop=True)

    def _raw_path(self, venue: str, day_key: str) -> Path:
        return Path(self.config.raw_cache_root) / venue / day_key / "trades.jsonl"

    def _checkpoint_path(self, venue: str) -> Path:
        return Path(self.config.raw_cache_root) / venue / "_checkpoint.json"

    @staticmethod
    def _read_checkpoint(path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_iso_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    malformed_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if not isinstance(payload, dict):
                malformed_count += 1
                continue
            rows.append(payload)
    if malformed_count:
        LOGGER.warning("Skipped %s malformed cached trade line(s) in %s", malformed_count, path)
    return rows


def default_markets_path() -> Path:
    for candidate in DEFAULT_MARKETS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a default KXBTC15M markets parquet under output/kalshi_series_backfill or output/kalshi_series.")


def infer_date_range_from_markets(markets_path: Path) -> tuple[date, date]:
    frame = pd.read_parquet(markets_path)
    if frame.empty:
        raise ValueError(f"Markets file is empty: {markets_path}")
    open_time = pd.to_datetime(frame["open_time"], utc=True)
    close_time = pd.to_datetime(frame["close_time"], utc=True)
    return open_time.min().date(), close_time.max().date()


def build_historical_btc_spot_backfill(
    *,
    config: HistoricalBTCSpotBackfillConfig | None = None,
    start_date: date,
    end_date: date,
    overwrite: bool = False,
) -> list[BackfillDateResult]:
    resolved_config = config or HistoricalBTCSpotBackfillConfig()
    cache = _RawTradeCache(resolved_config)
    runner = HistoricalBTCSpotBackfillRunner(resolved_config)
    warmup_start, _ = _day_bounds(start_date)
    _, range_end = _day_bounds(end_date)
    cache_start = warmup_start - timedelta(seconds=resolved_config.history_window_seconds)
    for venue in resolved_config.venues:
        cache.ensure_range(venue, cache_start, range_end)
    return runner.build_date_range(
        start_date=start_date,
        end_date=end_date,
        trade_loader=cache.load_window,
        overwrite=overwrite,
    )
