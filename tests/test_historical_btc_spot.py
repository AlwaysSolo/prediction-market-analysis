from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.live.data.btc_spot_feed import BTCSpotUpdate
from src.live.data.historical_btc_spot import (
    BackfillDateResult,
    HistoricalBTCSpotBackfillConfig,
    HistoricalBTCSpotBackfillRunner,
    _RawTradeCache,
    build_venue_proxy_frame,
    normalized_spot_column_order,
)
from src.live.kalshi.offline_training import enrich_market_feature_frame_with_external_spot


def _trade_frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp(timestamp),
                "price": float(price),
                "size": float(size),
            }
            for timestamp, price, size in rows
        ]
    )


def test_build_venue_proxy_frame_uses_trailing_two_second_median() -> None:
    trades = _trade_frame(
        [
            ("2026-01-01T00:00:00.000Z", 100.0, 1.0),
            ("2026-01-01T00:00:00.900Z", 102.0, 1.0),
            ("2026-01-01T00:00:01.700Z", 101.0, 1.0),
            ("2026-01-01T00:00:03.100Z", 110.0, 1.0),
        ]
    )

    proxy = build_venue_proxy_frame(trades, venue="coinbase")

    assert list(proxy["proxy_mid"]) == pytest.approx([100.0, 101.0, 101.0, 105.5])
    assert list(proxy["venue"]) == ["coinbase"] * 4


def test_backfill_runner_writes_spot_update_compatible_parquet_and_skips_existing_dates(tmp_path: Path) -> None:
    output_root = tmp_path / "spot"
    config = HistoricalBTCSpotBackfillConfig(
        output_root=output_root,
        environment="backfill",
        max_latency_ms=500,
        random_seed=7,
    )
    runner = HistoricalBTCSpotBackfillRunner(config)

    start_day = date(2026, 1, 1)
    end_day = date(2026, 1, 1)
    coinbase = _trade_frame(
        [
            ("2025-12-31T23:59:59.200Z", 85000.0, 0.5),
            ("2026-01-01T00:00:00.200Z", 85001.0, 0.4),
            ("2026-01-01T00:00:00.600Z", 85002.0, 0.3),
        ]
    )
    kraken = _trade_frame(
        [
            ("2025-12-31T23:59:59.300Z", 84999.0, 0.2),
            ("2026-01-01T00:00:00.350Z", 85000.0, 0.6),
            ("2026-01-01T00:00:00.900Z", 85003.0, 0.5),
        ]
    )

    results = runner.build_date_range(
        start_date=start_day,
        end_date=end_day,
        trade_loader=lambda venue, _start, _end: coinbase.copy() if venue == "coinbase" else kraken.copy(),
    )

    assert results == [BackfillDateResult(date=start_day, row_count=4, skipped=False)]
    output_path = output_root / "external_spot_normalized" / "backfill" / "2026-01-01" / "events.parquet"
    assert output_path.exists()

    frame = pd.read_parquet(output_path)
    assert tuple(frame.columns) == normalized_spot_column_order()
    assert set(frame["btc_spot_source"]) == {"trade_proxy"}
    assert frame["btc_spot_is_fresh"].all()
    assert (pd.to_datetime(frame["received_at"], utc=True) >= pd.to_datetime(frame["event_time"], utc=True)).all()
    assert float(frame["btc_spot_age_ms"].max()) <= 500.0

    second_results = runner.build_date_range(
        start_date=start_day,
        end_date=end_day,
        trade_loader=lambda venue, _start, _end: coinbase.copy() if venue == "coinbase" else kraken.copy(),
    )
    assert second_results == [BackfillDateResult(date=start_day, row_count=0, skipped=True)]


def test_enrich_market_feature_frame_uses_received_at_as_availability_time() -> None:
    target_df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-TEST",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00.150Z"),
                "open_time": pd.Timestamp("2026-01-01T11:55:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.55,
                "count": 1,
                "taker_side": "yes",
                "z_implied": 0.0,
                "tau_minutes": 10.0,
                "price_momentum": 0.0,
                "abs_price_momentum": 0.0,
                "price_direction": 0.0,
                "distance_from_mid": 0.0,
                "last_trade_count": 1.0,
                "last_trade_side_sign": 1.0,
                "last_trade_signed_count": 1.0,
                "time_since_last_trade_seconds": 1.0,
                "minutes_since_market_open": 5.0,
                "trade_count_30s": 1.0,
                "contracts_sum_30s": 1.0,
                "signed_contracts_sum_30s": 1.0,
                "yes_taker_share_30s": 1.0,
                "price_return_30s": 0.0,
                "price_volatility_30s": 0.0,
                "trade_count_120s": 1.0,
                "contracts_sum_120s": 1.0,
                "signed_contracts_sum_120s": 1.0,
                "yes_taker_share_120s": 1.0,
                "price_return_120s": 0.0,
                "price_volatility_120s": 0.0,
                "trade_count_300s": 1.0,
                "contracts_sum_300s": 1.0,
                "signed_contracts_sum_300s": 1.0,
                "yes_taker_share_300s": 1.0,
                "price_return_300s": 0.0,
                "price_volatility_300s": 0.0,
            }
        ]
    )
    market_row = pd.Series(
        {
            "ticker": "KXBTC15M-TEST",
            "event_ticker": "KXBTC15M",
            "yes_sub_title": "Price to beat: $85,000.00",
            "open_time": pd.Timestamp("2026-01-01T11:55:00Z"),
            "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
        }
    )
    external_spot_df = pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp("2026-01-01T12:00:00.000Z"),
                "received_at": pd.Timestamp("2026-01-01T12:00:00.250Z"),
                "btc_spot_price": 86000.0,
                "btc_spot_twap_60s": 85990.0,
                "btc_spot_age_ms": 250.0,
                "btc_spot_is_fresh": True,
                "btc_spot_venues_fresh": 2,
                "btc_spot_venue_divergence_bps": 1.5,
                "btc_vol_effective_sample_size": 120.0,
                "btc_spot_source": "trade_proxy",
                "btc_spot_return_30s": 0.001,
                "btc_spot_return_120s": 0.002,
                "btc_spot_return_300s": 0.003,
                "btc_spot_return_900s": 0.004,
                "btc_spot_vol_120s": 0.55,
                "btc_spot_vol_300s": 0.5,
                "btc_spot_vol_900s": 0.45,
                "btc_spot_vol_1800s": 0.4,
                "btc_spot_vol_ewma_hl300": 0.48,
            }
        ]
    )

    with pytest.raises(ValueError, match="No external spot snapshot available"):
        enrich_market_feature_frame_with_external_spot(
            target_df,
            market_row=market_row,
            external_spot_df=external_spot_df,
            external_spot_latency_ms=0,
        )


def test_raw_trade_cache_skips_malformed_jsonl_lines(tmp_path: Path) -> None:
    config = HistoricalBTCSpotBackfillConfig(raw_cache_root=tmp_path / "raw")
    cache = _RawTradeCache(config)
    path = tmp_path / "raw" / "coinbase" / "2026-03-10" / "trades.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '{"trade_id": 1, "event_time": "2026-03-10T00:00:00Z", "price": "85000.0", "size": "0.1", "side": "buy"}',
                '}\n',
                '{"trade_id": 2, "event_time": "2026-03-10T00:00:01Z", "price": "85001.0", "size": "0.2", "side": "sell"}',
                '"bad trailing fragment',
            ]
        ),
        encoding="utf-8",
    )

    frame = cache._read_cached_window(
        "coinbase",
        datetime(2026, 3, 10, 0, 0, tzinfo=UTC),
        datetime(2026, 3, 11, 0, 0, tzinfo=UTC),
    )

    assert list(frame["trade_id"]) == [1, 2]
    assert len(frame) == 2


def test_raw_trade_cache_accepts_mixed_iso8601_event_time_formats(tmp_path: Path) -> None:
    config = HistoricalBTCSpotBackfillConfig(raw_cache_root=tmp_path / "raw")
    cache = _RawTradeCache(config)
    path = tmp_path / "raw" / "coinbase" / "2026-03-03" / "trades.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '{"trade_id": 1, "event_time": "2026-03-03T18:20:21+00:00", "price": "85000.0", "size": "0.1", "side": "buy"}',
                '{"trade_id": 2, "event_time": "2026-03-03T18:20:21.123456+00:00", "price": "85001.0", "size": "0.2", "side": "sell"}',
            ]
        ),
        encoding="utf-8",
    )

    frame = cache._read_cached_window(
        "coinbase",
        datetime(2026, 3, 3, 0, 0, tzinfo=UTC),
        datetime(2026, 3, 4, 0, 0, tzinfo=UTC),
    )

    assert list(frame["trade_id"]) == [1, 2]
    assert str(frame.loc[0, "event_time"]) == "2026-03-03 18:20:21+00:00"
    assert str(frame.loc[1, "event_time"]) == "2026-03-03 18:20:21.123456+00:00"
