from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.live.kalshi import (
    FEATURE_ORDER,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureEngineConfig,
    KalshiFeatureStateEngine,
    KalshiMarketDataCollector,
)
from src.live.kalshi.offline_training import (
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_BAGGED_LASSO_C_VALUES,
    DEFAULT_ELASTIC_NET_L1_RATIOS,
    DEFAULT_LINEAR_SVM_C_VALUES,
    HOURLY_CONTEXT_SERIES,
    HourlyContextConfig,
    MINIMUM_POLICY_TRADES_PER_DAY,
    PolicyConfig,
    PolicyResult,
    SplitManifest,
    FeatureCacheStatus,
    bagged_lasso_raw_predictions,
    build_offline_feature_order,
    build_policy_diagnostics,
    build_prediction_export_frame,
    build_feature_cache,
    build_market_feature_frame,
    build_walk_forward_folds,
    compute_purge_embargo_gap_seconds,
    evaluate_policy,
    evaluate_walk_forward,
    calibrate_validation_predictions,
    enrich_market_feature_frame_with_hourly_context,
    enrich_market_feature_frame_with_external_spot,
    inspect_feature_cache,
    elastic_net_raw_predictions,
    feature_schema_metadata,
    infer_feature_names,
    lasso_raw_predictions,
    linear_svm_raw_predictions,
    build_split_manifest,
    load_bagged_lasso_artifact,
    load_elastic_net_artifact,
    load_feature_dataset,
    load_lasso_artifact,
    load_linear_svm_artifact,
    load_or_require_external_spot_frame,
    minimum_policy_trades_for_frame,
    publish_latest_artifacts,
    project_feature_frame,
    raw_predictions,
    run_bagged_lasso_search,
    resolve_inputs,
    run_elastic_net_search,
    run_lasso_search,
    run_linear_svm_search,
    save_bagged_lasso_artifacts,
    save_elastic_net_artifacts,
    save_feature_manifest,
    select_policy_candidate,
    serialize_policy_result,
    save_lasso_artifacts,
    save_linear_svm_artifacts,
    train_bagged_lasso_model,
    train_elastic_net_model,
    train_lasso_model,
    train_linear_svm_model,
    train_lightgbm_model,
    _bagged_lasso_positive_class_probabilities,
    _prepare_external_spot_frame_for_cache_build,
    _lasso_feature_matrix,
    _prepare_standard_policy_inputs,
    _simulate_policy,
    _simulate_standard_policy_prepared,
)
from src.live.kalshi.features import (
    LINEAR_V1_DERIVED_FEATURE_ORDER,
    LINEAR_V1_FEATURE_ORDER,
    LINEAR_V1_FEATURE_SCHEMA,
    SPOT_V1_FEATURE_ORDER,
    SPOT_V1_FEATURE_SCHEMA,
)
from src.live.kalshi.strike import strike_price_from_market
from src.live.kalshi.types import KalshiTickerUpdate


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


def _market_row(ticker: str) -> pd.Series:
    return pd.Series(
        {
            "ticker": ticker,
            "result": "yes",
            "open_time": pd.Timestamp("2026-01-01T11:55:00Z"),
            "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
        }
    )


def _trade_frame(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "ticker": ticker,
                "count": 2,
                "yes_price": 55,
                "no_price": 45,
                "taker_side": "yes",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
            },
            {
                "trade_id": "t2",
                "ticker": ticker,
                "count": 1,
                "yes_price": 56,
                "no_price": 44,
                "taker_side": "no",
                "created_time": pd.Timestamp("2026-01-01T12:00:10Z"),
            },
            {
                "trade_id": "t3",
                "ticker": ticker,
                "count": 3,
                "yes_price": 58,
                "no_price": 42,
                "taker_side": "yes",
                "created_time": pd.Timestamp("2026-01-01T12:01:00Z"),
            },
        ]
    )


def _external_spot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp("2026-01-01T11:59:59Z"),
                "received_at": pd.Timestamp("2026-01-01T11:59:59.080Z"),
                "btc_spot_price": 70000.0,
                "btc_spot_twap_60s": 69998.0,
                "btc_spot_age_ms": 80.0,
                "btc_spot_is_fresh": True,
                "btc_spot_venues_fresh": 2,
                "btc_spot_venue_divergence_bps": 1.0,
                "btc_vol_effective_sample_size": 120.0,
                "btc_spot_source": "trade_proxy",
                "btc_spot_return_30s": 0.001,
                "btc_spot_return_120s": 0.002,
                "btc_spot_return_300s": 0.003,
                "btc_spot_return_900s": 0.004,
                "btc_spot_vol_120s": 0.45,
                "btc_spot_vol_300s": 0.5,
                "btc_spot_vol_900s": 0.55,
                "btc_spot_vol_1800s": 0.6,
                "btc_spot_vol_ewma_hl300": 0.52,
            },
            {
                "event_time": pd.Timestamp("2026-01-01T12:00:10Z"),
                "received_at": pd.Timestamp("2026-01-01T12:00:10.080Z"),
                "btc_spot_price": 70010.0,
                "btc_spot_twap_60s": 70005.0,
                "btc_spot_age_ms": 80.0,
                "btc_spot_is_fresh": True,
                "btc_spot_venues_fresh": 2,
                "btc_spot_venue_divergence_bps": 1.0,
                "btc_vol_effective_sample_size": 121.0,
                "btc_spot_source": "trade_proxy",
                "btc_spot_return_30s": 0.0015,
                "btc_spot_return_120s": 0.0025,
                "btc_spot_return_300s": 0.0035,
                "btc_spot_return_900s": 0.0045,
                "btc_spot_vol_120s": 0.46,
                "btc_spot_vol_300s": 0.51,
                "btc_spot_vol_900s": 0.56,
                "btc_spot_vol_1800s": 0.61,
                "btc_spot_vol_ewma_hl300": 0.53,
            },
            {
                "event_time": pd.Timestamp("2026-01-01T12:01:00Z"),
                "received_at": pd.Timestamp("2026-01-01T12:01:00.080Z"),
                "btc_spot_price": 70020.0,
                "btc_spot_twap_60s": 70010.0,
                "btc_spot_age_ms": 80.0,
                "btc_spot_is_fresh": True,
                "btc_spot_venues_fresh": 2,
                "btc_spot_venue_divergence_bps": 1.0,
                "btc_vol_effective_sample_size": 122.0,
                "btc_spot_source": "trade_proxy",
                "btc_spot_return_30s": 0.002,
                "btc_spot_return_120s": 0.003,
                "btc_spot_return_300s": 0.004,
                "btc_spot_return_900s": 0.005,
                "btc_spot_vol_120s": 0.47,
                "btc_spot_vol_300s": 0.52,
                "btc_spot_vol_900s": 0.57,
                "btc_spot_vol_1800s": 0.62,
                "btc_spot_vol_ewma_hl300": 0.54,
            },
        ]
    )


def _hourly_market_row(
    ticker: str,
    *,
    open_time: str = "2026-01-01T12:00:00Z",
    close_time: str = "2026-01-01T13:00:00Z",
) -> pd.Series:
    return pd.Series(
        {
            "ticker": ticker,
            "result": "yes",
            "open_time": pd.Timestamp(open_time),
            "close_time": pd.Timestamp(close_time),
        }
    )


def _hourly_trade_frame(
    ticker: str,
    trades: list[tuple[str, int, int, str, str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": trade_id,
                "ticker": ticker,
                "count": count,
                "yes_price": yes_price,
                "no_price": 100 - yes_price,
                "taker_side": taker_side,
                "created_time": pd.Timestamp(created_time),
            }
            for trade_id, count, yes_price, taker_side, created_time in trades
        ]
    )


def test_build_split_manifest_is_reproducible_and_disjoint():
    markets = pd.DataFrame(
        {
            "ticker": [f"KXBTC15M-TEST-{i:02d}" for i in range(10)],
            "result": ["yes"] * 10,
            "close_time": pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC"),
            "open_time": pd.date_range("2025-12-31 23:45:00", periods=10, freq="15min", tz="UTC"),
        }
    )

    first = build_split_manifest(markets, series_ticker="KXBTC15M")
    second = build_split_manifest(markets, series_ticker="KXBTC15M")

    assert first.train_tickers == second.train_tickers
    assert first.validation_tickers == second.validation_tickers
    assert first.test_tickers == second.test_tickers
    assert not set(first.train_tickers) & set(first.validation_tickers)
    assert not set(first.train_tickers) & set(first.test_tickers)
    assert not set(first.validation_tickers) & set(first.test_tickers)


def test_compute_purge_embargo_gap_seconds_uses_feature_lookback_and_horizon():
    markets = pd.DataFrame(
        {
            "ticker": [f"KXBTC15M-TEST-{i:02d}" for i in range(4)],
            "result": ["yes"] * 4,
            "close_time": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"),
            "open_time": pd.date_range("2025-12-31 23:45:00", periods=4, freq="15min", tz="UTC"),
        }
    )

    default_gap = compute_purge_embargo_gap_seconds(markets, feature_schema=DEFAULT_FEATURE_SCHEMA)
    spot_gap = compute_purge_embargo_gap_seconds(markets, feature_schema=SPOT_V1_FEATURE_SCHEMA)

    assert default_gap == 300 + 900 + 60
    assert spot_gap == 1800 + 900 + 60


def test_build_split_manifest_with_purge_embargo_skips_boundary_tickers():
    markets = pd.DataFrame(
        {
            "ticker": [f"KXBTC15M-TEST-{i:02d}" for i in range(12)],
            "result": ["yes"] * 12,
            "close_time": pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC"),
            "open_time": pd.date_range("2025-12-31 23:45:00", periods=12, freq="15min", tz="UTC"),
        }
    )

    manifest = build_split_manifest(
        markets,
        series_ticker="KXBTC15M",
        train_fraction=0.5,
        validation_fraction=0.25,
        purge_embargo=True,
        purge_embargo_gap_seconds=21 * 60,
    )

    assert manifest.train_tickers == tuple(f"KXBTC15M-TEST-{i:02d}" for i in range(6))
    assert manifest.validation_tickers == tuple(f"KXBTC15M-TEST-{i:02d}" for i in range(7, 10))
    assert manifest.test_tickers == ("KXBTC15M-TEST-11",)
    assert "KXBTC15M-TEST-06" not in manifest.validation_tickers
    assert "KXBTC15M-TEST-10" not in manifest.test_tickers
    assert manifest.purge_embargo_enabled is True
    assert manifest.purge_embargo_gap_seconds == 21 * 60


def test_build_walk_forward_folds_with_purge_embargo_skips_boundary_tickers():
    markets = pd.DataFrame(
        {
            "ticker": [f"KXBTC15M-TEST-{i:02d}" for i in range(12)],
            "result": ["yes"] * 12,
            "close_time": pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC"),
            "open_time": pd.date_range("2025-12-31 23:45:00", periods=12, freq="15min", tz="UTC"),
        }
    )

    folds = build_walk_forward_folds(
        markets,
        num_blocks=6,
        purge_embargo=True,
        purge_embargo_gap_seconds=21 * 60,
    )

    first_fold = folds[0]
    assert first_fold.train_tickers == ("KXBTC15M-TEST-00",)
    assert first_fold.validation_tickers == ("KXBTC15M-TEST-02", "KXBTC15M-TEST-03")
    assert first_fold.test_tickers == ("KXBTC15M-TEST-05",)


def test_evaluate_walk_forward_threads_purge_embargo_settings_to_fold_builder(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_build_walk_forward_folds(markets_df, num_blocks=6, **kwargs):
        captured["kwargs"] = kwargs
        return [
            type(
                "Fold",
                (),
                {
                    "fold_index": 1,
                    "train_tickers": ("train",),
                    "validation_tickers": ("validation",),
                    "test_tickers": ("test",),
                },
            )()
        ]

    def fake_load_feature_dataset(*args, **kwargs):
        return pd.DataFrame({"actual_outcome": [0, 1], "ticker": ["T1", "T2"]})

    class _IdentityCalibration:
        def apply(self, probabilities):
            return probabilities

    monkeypatch.setattr("src.live.kalshi.offline_training.build_walk_forward_folds", fake_build_walk_forward_folds)
    monkeypatch.setattr("src.live.kalshi.offline_training.load_feature_dataset", fake_load_feature_dataset)
    monkeypatch.setattr(
        "src.live.kalshi.offline_training.train_lightgbm_model",
        lambda train_df, validation_df, params: (object(), {"best_iteration": 7}),
    )
    monkeypatch.setattr(
        "src.live.kalshi.offline_training.raw_predictions",
        lambda model, df: np.array([0.4, 0.6], dtype=float),
    )
    monkeypatch.setattr("src.live.kalshi.offline_training.fit_platt_scaler", lambda raw, labels: _IdentityCalibration())
    monkeypatch.setattr(
        "src.live.kalshi.offline_training.sweep_policy_grid",
        lambda df, probabilities, overall_log_loss, progress_desc=None: [
            PolicyResult(
                config=PolicyConfig(
                    edge_threshold_cents=1.0,
                    min_tau_minutes=0.0,
                    max_tau_minutes=15.0,
                    price_band_min_cents=10,
                    price_band_max_cents=90,
                    reserve_cash_pct=30.0,
                ),
                objective=1.0,
                trades=12,
                net_pnl_dollars=1.0,
                max_drawdown_dollars=0.5,
                max_drawdown_pct=0.05,
                return_pct=0.01,
                log_loss=overall_log_loss,
            )
        ],
    )
    monkeypatch.setattr("src.live.kalshi.offline_training.minimum_policy_trades_for_frame", lambda df: 12)
    monkeypatch.setattr(
        "src.live.kalshi.offline_training.select_policy_candidate",
        lambda results, minimum_trades, fallback_to_best_overall: (results[0], True),
    )
    monkeypatch.setattr(
        "src.live.kalshi.offline_training.evaluate_policy",
        lambda df, probabilities, config, overall_log_loss, progress_desc=None: {"trades": 12, "log_loss": overall_log_loss},
    )

    evaluate_walk_forward(
        Path("unused"),
        pd.DataFrame({"ticker": ["train", "validation", "test"]}),
        {"learning_rate": 0.1},
        feature_schema=DEFAULT_FEATURE_SCHEMA,
        purge_embargo=True,
        purge_embargo_gap_seconds=1234,
        purge_embargo_safety_margin_seconds=99,
    )

    assert captured["kwargs"] == {
        "feature_schema": DEFAULT_FEATURE_SCHEMA,
        "purge_embargo": True,
        "purge_embargo_gap_seconds": 1234,
        "purge_embargo_safety_margin_seconds": 99,
    }


def test_offline_market_feature_frame_matches_live_feature_engine(tmp_path: Path):
    ticker = "KXBTC15M-TEST"
    trades_df = _trade_frame(ticker)
    offline_df = build_market_feature_frame(trades_df, _market_row(ticker))

    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    queue = engine.subscribe_queue()

    async def run() -> list[tuple[float, ...]]:
        await engine.start()
        observed_rows: list[tuple[float, ...]] = []
        previous_yes_price_cents: int | None = None
        for row in trades_df.itertuples(index=False):
            await collector._publish_update(
                KalshiTickerUpdate(
                    ticker=ticker,
                    event_time=row.created_time,
                    last_yes_price_cents=int(row.yes_price),
                    previous_yes_price_cents=previous_yes_price_cents,
                    close_time=pd.Timestamp("2026-01-01T12:10:00Z"),
                    is_open=True,
                    market_prob=None,
                    previous_market_prob=None,
                    price_momentum=None,
                    tau_minutes=None,
                    open_time=pd.Timestamp("2026-01-01T11:55:00Z"),
                    trade_id=str(row.trade_id),
                    count=int(row.count),
                    taker_side=str(row.taker_side),
                    last_trade_time=row.created_time,
                    last_price_cents=int(row.yes_price),
                    source="trade",
                )
            )
            update = await asyncio.wait_for(queue.get(), timeout=0.5)
            values = update.feature_values()
            assert values is not None
            observed_rows.append(values)
            previous_yes_price_cents = int(row.yes_price)
        await engine.stop()
        return observed_rows

    live_rows = asyncio.run(run())
    assert len(offline_df) == len(live_rows)
    for index, live_row in enumerate(live_rows):
        offline_row = tuple(float(offline_df.iloc[index][feature_name]) for feature_name in FEATURE_ORDER)
        assert live_row == offline_row


def test_build_feature_cache_and_load_dataset(tmp_path: Path):
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    first_ticker = "KXBTC15M-TEST-1"
    second_ticker = "KXBTC15M-TEST-2"
    _trade_frame(first_ticker).to_parquet(trades_dir / f"{first_ticker}.parquet", index=False)
    _trade_frame(second_ticker).to_parquet(trades_dir / f"{second_ticker}.parquet", index=False)

    markets_df = pd.DataFrame(
        [
            _market_row(first_ticker).to_dict(),
            {
                **_market_row(second_ticker).to_dict(),
                "close_time": pd.Timestamp("2026-01-01T12:14:00Z"),
            },
        ]
    )

    cache_dir = tmp_path / "cache"
    build_feature_cache(markets_df, trades_dir, cache_dir)
    loaded = load_feature_dataset(cache_dir, (first_ticker, second_ticker))

    assert not loaded.empty
    assert set(loaded["ticker"]) == {first_ticker, second_ticker}


def test_prepare_external_spot_frame_for_cache_build_trims_to_market_window() -> None:
    external_spot_df = pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp("2026-01-01T11:00:00Z"),
                "received_at": pd.Timestamp("2026-01-01T11:00:00.080Z"),
                "btc_spot_price": 69900.0,
            },
            {
                "event_time": pd.Timestamp("2026-01-01T11:54:59Z"),
                "received_at": pd.Timestamp("2026-01-01T11:54:59.080Z"),
                "btc_spot_price": 69950.0,
            },
            {
                "event_time": pd.Timestamp("2026-01-01T12:00:10Z"),
                "received_at": pd.Timestamp("2026-01-01T12:00:10.080Z"),
                "btc_spot_price": 70010.0,
            },
            {
                "event_time": pd.Timestamp("2026-01-01T12:30:01Z"),
                "received_at": pd.Timestamp("2026-01-01T12:30:01.080Z"),
                "btc_spot_price": 70020.0,
            },
        ]
    )
    markets_df = pd.DataFrame(
        [
            {
                **_market_row("KXBTC15M-TEST").to_dict(),
                "open_time": pd.Timestamp("2026-01-01T11:55:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
            }
        ]
    )

    prepared = _prepare_external_spot_frame_for_cache_build(
        external_spot_df,
        markets_df,
        external_spot_latency_ms=20,
    )

    assert list(prepared.frame["event_time"]) == [
        pd.Timestamp("2026-01-01T11:54:59Z"),
        pd.Timestamp("2026-01-01T12:00:10Z"),
    ]
    assert "available_time" in prepared.frame.columns
    assert prepared.frame.iloc[0]["available_time"] == pd.Timestamp("2026-01-01T11:54:59.100Z")


def test_inspect_feature_cache_marks_partial_cache_not_ready(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "KXBTC15M-TEST-1.parquet").write_bytes(b"placeholder")

    status = inspect_feature_cache(
        cache_dir,
        ("KXBTC15M-TEST-1", "KXBTC15M-TEST-2"),
        feature_schema=SPOT_V1_FEATURE_SCHEMA,
    )

    assert status.ready is False
    assert status.manifest_exists is False
    assert status.present_requested_tickers == 1
    assert status.missing_requested_tickers == ("KXBTC15M-TEST-2",)


def test_inspect_feature_cache_reports_ready_when_manifest_and_requested_files_exist(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ticker in ("KXBTC15M-TEST-1", "KXBTC15M-TEST-2"):
        pd.DataFrame([{"ticker": ticker}]).to_parquet(cache_dir / f"{ticker}.parquet", index=False)
    save_feature_manifest(
        cache_dir / "feature_manifest.json",
        feature_order=("z_implied",),
        metadata=feature_schema_metadata(SPOT_V1_FEATURE_SCHEMA),
    )

    status = inspect_feature_cache(
        cache_dir,
        ("KXBTC15M-TEST-1", "KXBTC15M-TEST-2"),
        feature_schema=SPOT_V1_FEATURE_SCHEMA,
    )

    assert status.ready is True
    assert status.manifest_exists is True
    assert status.schema_matches is True
    assert status.present_requested_tickers == 2
    assert status.missing_requested_tickers == ()


def test_build_offline_feature_order_linear_v1_matches_agreed_schema():
    feature_order = build_offline_feature_order(
        hourly_context=HourlyContextConfig(),
        feature_schema=LINEAR_V1_FEATURE_SCHEMA,
    )

    assert feature_order == LINEAR_V1_FEATURE_ORDER
    assert len(feature_order) == 36
    assert "price_direction" not in feature_order
    assert "last_trade_count" not in feature_order
    assert "trade_count_120s" not in feature_order
    assert feature_order[-len(LINEAR_V1_DERIVED_FEATURE_ORDER) :] == LINEAR_V1_DERIVED_FEATURE_ORDER


def test_build_offline_feature_order_spot_v1_appends_spot_features():
    feature_order = build_offline_feature_order(feature_schema=SPOT_V1_FEATURE_SCHEMA)

    assert feature_order[: len(FEATURE_ORDER)] == tuple(FEATURE_ORDER)
    assert feature_order[-len(SPOT_V1_FEATURE_ORDER) :] == SPOT_V1_FEATURE_ORDER
    assert "btc_log_moneyness" in feature_order
    assert "btc_kalshi_implied_vol" in feature_order


def test_feature_schema_metadata_spot_v1_includes_feature_order_hash():
    metadata = feature_schema_metadata(SPOT_V1_FEATURE_SCHEMA)

    assert metadata["schema_name"] == SPOT_V1_FEATURE_SCHEMA
    assert metadata["feature_order_hash"]


def test_load_or_require_external_spot_frame_requires_root_for_spot_v1() -> None:
    with pytest.raises(ValueError, match="requires --external-spot-root"):
        load_or_require_external_spot_frame(SPOT_V1_FEATURE_SCHEMA, None)


def test_policy_config_signal_config_clamps_default_maintain_edge_to_threshold() -> None:
    zero_edge_config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
    )

    assert zero_edge_config.signal_config.edge_threshold_cents == pytest.approx(0.0)
    assert zero_edge_config.signal_config.maintain_edge_cents == pytest.approx(0.0)

    wider_edge_config = PolicyConfig(
        edge_threshold_cents=6.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
    )

    assert wider_edge_config.signal_config.maintain_edge_cents == pytest.approx(1.0)


def test_strike_price_from_market_supports_target_price_formats() -> None:
    assert strike_price_from_market(
        {
            "yes_sub_title": "Target price: $70,029.87",
            "no_sub_title": "Price to beat: TBD",
        }
    ) == pytest.approx(70029.87)
    assert strike_price_from_market(
        {
            "title": "BTC 15 min · $70,029.87 target",
            "yes_sub_title": "Target price: TBD",
            "no_sub_title": "Target price: TBD",
        }
    ) == pytest.approx(70029.87)


def test_enrich_market_feature_frame_with_external_spot_falls_back_to_open_spot_for_generic_direction_market() -> None:
    ticker = "KXBTC15M-TEST"
    target_df = build_market_feature_frame(_trade_frame(ticker), _market_row(ticker))
    external_spot_df = _external_spot_frame()

    enriched = enrich_market_feature_frame_with_external_spot(
        target_df,
        market_row=pd.Series(
            {
                **_market_row(ticker).to_dict(),
                "title": "BTC price up in next 15 mins?",
                "yes_sub_title": "Target price: TBD",
                "no_sub_title": "Target price: TBD",
            }
        ),
        external_spot_df=external_spot_df,
    )

    assert not enriched.empty
    assert enriched["btc_log_moneyness"].notna().all()
    assert float(enriched.iloc[0]["btc_log_moneyness"]) == pytest.approx(0.0)
    expected = np.log(70010.0 / 70000.0)
    assert float(enriched.iloc[-1]["btc_log_moneyness"]) == pytest.approx(expected, rel=1e-6)


def test_project_feature_frame_linear_v1_drops_redundant_columns_and_adds_interactions():
    raw_df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-TEST",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "open_time": pd.Timestamp("2026-01-01T11:55:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.58,
                "count": 2,
                "taker_side": "yes",
                "z_implied": 0.7,
                "tau_minutes": -0.25,
                "price_momentum": 0.03,
                "abs_price_momentum": 0.03,
                "price_direction": 1.0,
                "distance_from_mid": 0.08,
                "last_trade_count": 2.0,
                "last_trade_side_sign": 1.0,
                "last_trade_signed_count": 2.0,
                "time_since_last_trade_seconds": 5.0,
                "minutes_since_market_open": 4.5,
                "trade_count_30s": 3.0,
                "contracts_sum_30s": 6.0,
                "signed_contracts_sum_30s": 4.0,
                "yes_taker_share_30s": 0.66,
                "price_return_30s": 0.01,
                "price_volatility_30s": 0.015,
                "trade_count_120s": 5.0,
                "contracts_sum_120s": 9.0,
                "signed_contracts_sum_120s": 3.0,
                "yes_taker_share_120s": 0.55,
                "price_return_120s": 0.015,
                "price_volatility_120s": 0.02,
                "trade_count_300s": 11.0,
                "contracts_sum_300s": 18.0,
                "signed_contracts_sum_300s": 7.0,
                "yes_taker_share_300s": 0.61,
                "price_return_300s": 0.04,
                "price_volatility_300s": 0.025,
                "kxbtcd_atm_z_implied": -0.5,
                "kxbtcd_atm_price_momentum": -0.02,
                "kxbtcd_atm_abs_price_momentum": 0.02,
                "kxbtcd_atm_distance_from_mid": 0.11,
                "kxbtcd_atm_trade_count_300s": 8.0,
                "kxbtcd_atm_signed_contracts_sum_300s": -6.0,
                "kxbtcd_atm_price_return_300s": -0.03,
                "kxbtcd_atm_price_volatility_300s": 0.03,
                "kxbtcd_atm_yes_taker_share_300s": 0.42,
                "k15_minus_k1h_atm_z": 1.2,
                "k15_minus_k1h_atm_price_return_300s": 0.07,
                "k15_k1h_atm_direction_agreement": -1.0,
            }
        ]
    )

    projected = project_feature_frame(raw_df, feature_schema=LINEAR_V1_FEATURE_SCHEMA)

    assert tuple(infer_feature_names(projected)) == LINEAR_V1_FEATURE_ORDER
    assert projected.attrs["feature_schema"] == LINEAR_V1_FEATURE_SCHEMA
    assert "price_direction" not in projected.columns
    assert "last_trade_count" not in projected.columns
    assert "trade_count_120s" not in projected.columns
    assert projected.loc[0, "k15_z_tau_decay"] == pytest.approx(0.7)
    assert projected.loc[0, "k15_return_300s_tau_decay"] == pytest.approx(0.04)
    assert projected.loc[0, "k15_k1h_z_product"] == pytest.approx(-0.35)
    assert projected.loc[0, "k15_k1h_return_product_300s"] == pytest.approx(-0.0012)


def test_load_feature_dataset_projects_linear_v1_from_raw_hourly_cache(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ticker = "KXBTC15M-TEST"
    raw_df = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "open_time": pd.Timestamp("2026-01-01T11:55:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.55,
                "count": 1,
                "taker_side": "yes",
                **{
                    feature_name: float(index + 1)
                    for index, feature_name in enumerate(build_offline_feature_order(hourly_context=HourlyContextConfig()))
                },
            }
        ]
    )
    raw_df.to_parquet(cache_dir / f"{ticker}.parquet", index=False)
    save_feature_manifest(
        cache_dir / "feature_manifest.json",
        feature_order=build_offline_feature_order(hourly_context=HourlyContextConfig()),
        metadata={
            **feature_schema_metadata(DEFAULT_FEATURE_SCHEMA),
            "hourly_context_series": HOURLY_CONTEXT_SERIES,
        },
    )

    loaded = load_feature_dataset(
        cache_dir,
        (ticker,),
        feature_schema=LINEAR_V1_FEATURE_SCHEMA,
    )

    assert tuple(infer_feature_names(loaded)) == LINEAR_V1_FEATURE_ORDER
    assert loaded.attrs["feature_schema"] == LINEAR_V1_FEATURE_SCHEMA


def test_enrich_market_feature_frame_with_hourly_context_uses_nearest_expiry_liquid_atm(tmp_path: Path):
    target_ticker = "KXBTC15M-TEST-1300"
    target_df = build_market_feature_frame(
        _hourly_trade_frame(
            target_ticker,
            [
                ("k15-t1", 2, 61, "yes", "2026-01-01T12:56:00Z"),
                ("k15-t2", 1, 62, "yes", "2026-01-01T12:57:00Z"),
            ],
        ),
        pd.Series(
            {
                "ticker": target_ticker,
                "result": "yes",
                "open_time": pd.Timestamp("2026-01-01T12:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T13:00:00Z"),
            }
        ),
    )

    context_cache_dir = tmp_path / "context"
    context_cache_dir.mkdir(parents=True, exist_ok=True)
    hourly_context = HourlyContextConfig(series_ticker=HOURLY_CONTEXT_SERIES)
    hourly_feature_config = KalshiFeatureEngineConfig(tau_max_minutes=hourly_context.tau_max_minutes)

    current_atm_ticker = "KXBTCD-TEST-1300-T65500"
    current_otm_ticker = "KXBTCD-TEST-1300-T70000"
    future_atm_ticker = "KXBTCD-TEST-1400-T65500"
    context_markets_df = pd.DataFrame(
        [
            _hourly_market_row(current_atm_ticker).to_dict(),
            _hourly_market_row(current_otm_ticker).to_dict(),
            _hourly_market_row(
                future_atm_ticker,
                open_time="2026-01-01T12:30:00Z",
                close_time="2026-01-01T13:30:00Z",
            ).to_dict(),
        ]
    )

    build_market_feature_frame(
        _hourly_trade_frame(
            current_atm_ticker,
            [
                ("atm-1", 3, 49, "yes", "2026-01-01T12:55:10Z"),
                ("atm-2", 2, 51, "yes", "2026-01-01T12:56:20Z"),
            ],
        ),
        _hourly_market_row(current_atm_ticker),
        feature_config=hourly_feature_config,
    ).to_parquet(context_cache_dir / f"{current_atm_ticker}.parquet", index=False)
    build_market_feature_frame(
        _hourly_trade_frame(
            current_otm_ticker,
            [("otm-1", 1, 6, "yes", "2026-01-01T12:55:15Z")],
        ),
        _hourly_market_row(current_otm_ticker),
        feature_config=hourly_feature_config,
    ).to_parquet(context_cache_dir / f"{current_otm_ticker}.parquet", index=False)
    build_market_feature_frame(
        _hourly_trade_frame(
            future_atm_ticker,
            [("future-1", 4, 50, "yes", "2026-01-01T12:56:10Z")],
        ),
        _hourly_market_row(
            future_atm_ticker,
            open_time="2026-01-01T12:30:00Z",
            close_time="2026-01-01T13:30:00Z",
        ),
        feature_config=hourly_feature_config,
    ).to_parquet(context_cache_dir / f"{future_atm_ticker}.parquet", index=False)

    enriched = enrich_market_feature_frame_with_hourly_context(
        target_df,
        context_markets_df=context_markets_df,
        context_cache_dir=context_cache_dir,
        hourly_context=hourly_context,
    )

    expected_feature_order = build_offline_feature_order(hourly_context=hourly_context)
    assert list(infer_feature_names(enriched)) == list(expected_feature_order)

    selected_context_df = pd.read_parquet(context_cache_dir / f"{current_atm_ticker}.parquet")
    selected_row = selected_context_df.iloc[-1]
    assert enriched["kxbtcd_atm_z_implied"].iloc[-1] == pytest.approx(float(selected_row["z_implied"]))
    assert enriched["kxbtcd_atm_trade_count_300s"].iloc[-1] == pytest.approx(float(selected_row["trade_count_300s"]))
    assert enriched["kxbtcd_atm_distance_from_mid"].iloc[-1] == pytest.approx(float(selected_row["distance_from_mid"]))
    assert enriched["k15_minus_k1h_atm_z"].iloc[-1] == pytest.approx(
        float(enriched["z_implied"].iloc[-1] - selected_row["z_implied"])
    )
    assert enriched["kxbtcd_atm_z_implied"].iloc[-1] != pytest.approx(
        float(pd.read_parquet(context_cache_dir / f"{future_atm_ticker}.parquet").iloc[-1]["z_implied"])
    )


def test_train_lasso_model_uses_inferred_extended_feature_order(tmp_path: Path):
    extended_feature_order = build_offline_feature_order(hourly_context=HourlyContextConfig())
    rows: list[dict[str, object]] = []
    for index in range(80):
        feature_payload = {
            feature_name: np.float32(((index * 5 + feature_offset) % 23) / 10.0)
            for feature_offset, feature_name in enumerate(extended_feature_order)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-EXT-{index // 8}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.35 + (index % 10) * 0.02,
                "count": 1,
                "taker_side": "yes",
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:50].reset_index(drop=True)
    validation_df = frame.iloc[50:65].reset_index(drop=True)
    test_df = frame.iloc[65:].reset_index(drop=True)

    best_params, _history = run_lasso_search(
        train_df,
        validation_df,
        c_values=(0.01, 0.1),
        max_iter=500,
    )
    artifact, metrics = train_lasso_model(train_df, validation_df, best_params)
    probabilities = lasso_raw_predictions(artifact, test_df)

    assert len(probabilities) == len(test_df)
    assert metrics["feature_names"] == list(extended_feature_order)

    run_dir = tmp_path / "lasso-extended-run"
    save_lasso_artifacts(run_dir, artifact, metrics)
    reloaded_metrics = json.loads((run_dir / "lasso" / "metrics.json").read_text(encoding="utf-8"))
    assert reloaded_metrics["feature_names"] == list(extended_feature_order)


def test_load_feature_dataset_skips_empty_cached_frames_without_concat_warning(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    empty_ticker = "KXBTC15M-EMPTY"
    data_ticker = "KXBTC15M-DATA"
    build_market_feature_frame(
        _trade_frame(empty_ticker),
        pd.Series(
            {
                "ticker": empty_ticker,
                "result": "yes",
                "open_time": pd.Timestamp("2026-01-01T11:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:25:00Z"),
            }
        ),
    ).to_parquet(cache_dir / f"{empty_ticker}.parquet", index=False)
    build_market_feature_frame(_trade_frame(data_ticker), _market_row(data_ticker)).to_parquet(
        cache_dir / f"{data_ticker}.parquet",
        index=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_feature_dataset(cache_dir, (empty_ticker, data_ticker))

    assert not loaded.empty
    assert set(loaded["ticker"]) == {data_ticker}
    assert not any("DataFrame concatenation with empty or all-NA entries is deprecated" in str(item.message) for item in caught)


def test_build_market_feature_frame_with_no_in_window_rows_has_stable_schema(tmp_path: Path):
    ticker = "KXBTC15M-TOO-EARLY"
    trades_df = _trade_frame(ticker)
    market_row = pd.Series(
        {
            "ticker": ticker,
            "result": "yes",
            "open_time": pd.Timestamp("2026-01-01T11:00:00Z"),
            "close_time": pd.Timestamp("2026-01-01T12:25:00Z"),
        }
    )

    feature_df = build_market_feature_frame(trades_df, market_row)

    assert feature_df.empty
    assert feature_df.columns.is_unique
    assert "tau_minutes" in feature_df.columns

    output_path = tmp_path / f"{ticker}.parquet"
    feature_df.to_parquet(output_path, index=False)
    reloaded = pd.read_parquet(output_path)
    assert list(reloaded.columns) == list(feature_df.columns)


def test_train_lightgbm_model_and_raw_predictions_do_not_warn_about_feature_names():
    rows: list[dict[str, object]] = []
    for index in range(60):
        feature_payload = {
            feature_name: np.float32((index + feature_offset) / 100.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-TEST-{index // 10}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.4 + (index % 10) * 0.01,
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:40].reset_index(drop=True)
    validation_df = frame.iloc[40:50].reset_index(drop=True)
    test_df = frame.iloc[50:].reset_index(drop=True)

    params = {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, _metrics = train_lightgbm_model(train_df, validation_df, params)
        probabilities = raw_predictions(model, test_df)

    assert len(probabilities) == len(test_df)
    assert not any("valid feature names" in str(item.message) for item in caught)


def test_train_lasso_model_and_predictions_round_trip(tmp_path: Path):
    rows: list[dict[str, object]] = []
    for index in range(80):
        feature_payload = {
            feature_name: np.float32(((index + feature_offset) % 17) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-LASSO-{index // 8}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.35 + (index % 10) * 0.02,
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:50].reset_index(drop=True)
    validation_df = frame.iloc[50:65].reset_index(drop=True)
    test_df = frame.iloc[65:].reset_index(drop=True)

    best_params, history = run_lasso_search(
        train_df,
        validation_df,
        c_values=(0.01, 0.1),
        max_iter=500,
    )
    assert history
    artifact, metrics = train_lasso_model(train_df, validation_df, best_params)
    probabilities = lasso_raw_predictions(artifact, test_df)

    assert len(probabilities) == len(test_df)
    assert metrics["nonzero_coefficients"] >= 0

    run_dir = tmp_path / "lasso-run"
    save_lasso_artifacts(run_dir, artifact, metrics)
    reloaded = load_lasso_artifact(run_dir / "lasso" / "model.joblib")
    reloaded_probabilities = lasso_raw_predictions(reloaded, test_df)

    assert np.allclose(probabilities, reloaded_probabilities)


def test_train_bagged_lasso_model_and_predictions_round_trip(tmp_path: Path):
    rows: list[dict[str, object]] = []
    for index in range(96):
        feature_payload = {
            feature_name: np.float32(((index * 7 + feature_offset) % 19) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-BAG-{index // 8}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": (index // 2) % 2,
                "market_prob": 0.32 + (index % 12) * 0.03,
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:60].reset_index(drop=True)
    validation_df = frame.iloc[60:78].reset_index(drop=True)
    test_df = frame.iloc[78:].reset_index(drop=True)

    best_params, history = run_bagged_lasso_search(
        train_df,
        validation_df,
        c_values=DEFAULT_BAGGED_LASSO_C_VALUES[:2],
        n_estimators=5,
        max_samples=0.8,
        max_iter=500,
        n_jobs=1,
    )
    assert history

    artifact, metrics = train_bagged_lasso_model(train_df, validation_df, best_params)
    probabilities = bagged_lasso_raw_predictions(artifact, test_df)

    assert len(probabilities) == len(test_df)
    assert metrics["estimators_trained"] == 5
    assert metrics["mean_nonzero_coefficients"] >= 0.0

    run_dir = tmp_path / "bagged-lasso-run"
    save_bagged_lasso_artifacts(run_dir, artifact, metrics)
    reloaded = load_bagged_lasso_artifact(run_dir / "bagged_lasso" / "model.joblib")
    reloaded_probabilities = bagged_lasso_raw_predictions(reloaded, test_df)

    assert np.allclose(probabilities, reloaded_probabilities)


def test_bagged_lasso_manual_probability_path_matches_sklearn_predict_proba() -> None:
    rows: list[dict[str, object]] = []
    for index in range(96):
        feature_payload = {
            feature_name: np.float32(((index * 7 + feature_offset) % 19) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-BAG-CHECK-{index // 8}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": (index // 2) % 2,
                "market_prob": 0.32 + (index % 12) * 0.03,
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:60].reset_index(drop=True)
    validation_df = frame.iloc[60:78].reset_index(drop=True)

    params = {
        "C": float(DEFAULT_BAGGED_LASSO_C_VALUES[0]),
        "fit_intercept": True,
        "n_estimators": 5,
        "max_samples": 0.8,
        "max_iter": 500,
        "tol": 1e-4,
        "n_jobs": 2,
    }
    artifact, _ = train_bagged_lasso_model(train_df, validation_df, params)
    feature_names = infer_feature_names(validation_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    X_valid_scaled = artifact.scaler.transform(X_valid)

    expected = np.clip(artifact.model.predict_proba(X_valid_scaled)[:, 1], 1e-6, 1.0 - 1e-6)
    observed = _bagged_lasso_positive_class_probabilities(artifact.model, X_valid_scaled)

    assert np.allclose(expected, observed)


def test_train_elastic_net_model_and_predictions_round_trip(tmp_path: Path):
    rows: list[dict[str, object]] = []
    for index in range(96):
        feature_payload = {
            feature_name: np.float32(((index * 3 + feature_offset) % 19) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-EN-{index // 8}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": (index // 3) % 2,
                "market_prob": 0.30 + (index % 12) * 0.03,
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:60].reset_index(drop=True)
    validation_df = frame.iloc[60:78].reset_index(drop=True)
    test_df = frame.iloc[78:].reset_index(drop=True)

    best_params, history = run_elastic_net_search(
        train_df,
        validation_df,
        c_values=(0.01, 0.1),
        l1_ratios=(0.25, 0.75),
        max_iter=500,
    )
    assert history
    assert float(best_params["l1_ratio"]) in {0.25, 0.75}

    artifact, metrics = train_elastic_net_model(train_df, validation_df, best_params)
    probabilities = elastic_net_raw_predictions(artifact, test_df)

    assert len(probabilities) == len(test_df)
    assert metrics["nonzero_coefficients"] >= 0

    run_dir = tmp_path / "elastic-net-run"
    save_elastic_net_artifacts(run_dir, artifact, metrics)
    reloaded = load_elastic_net_artifact(run_dir / "elastic_net" / "model.joblib")
    reloaded_probabilities = elastic_net_raw_predictions(reloaded, test_df)

    assert np.allclose(probabilities, reloaded_probabilities)


def test_train_linear_svm_model_and_predictions_round_trip(tmp_path: Path):
    rows: list[dict[str, object]] = []
    for index in range(96):
        feature_payload = {
            feature_name: np.float32(((index * 5 + feature_offset) % 23) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-SVM-{index // 8}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": (index // 4) % 2,
                "market_prob": 0.28 + (index % 12) * 0.03,
                **feature_payload,
            }
        )

    frame = pd.DataFrame(rows)
    train_df = frame.iloc[:60].reset_index(drop=True)
    validation_df = frame.iloc[60:78].reset_index(drop=True)
    test_df = frame.iloc[78:].reset_index(drop=True)

    best_params, history = run_linear_svm_search(
        train_df,
        validation_df,
        c_values=DEFAULT_LINEAR_SVM_C_VALUES[:2],
        max_iter=500,
    )
    assert history

    artifact, metrics = train_linear_svm_model(train_df, validation_df, best_params)
    probabilities = linear_svm_raw_predictions(artifact, test_df)

    assert len(probabilities) == len(test_df)
    assert metrics["nonzero_coefficients"] >= 0

    run_dir = tmp_path / "linear-svm-run"
    save_linear_svm_artifacts(run_dir, artifact, metrics)
    reloaded = load_linear_svm_artifact(run_dir / "linear_svm" / "model.joblib")
    reloaded_probabilities = linear_svm_raw_predictions(reloaded, test_df)

    assert np.allclose(probabilities, reloaded_probabilities)


def test_publish_latest_artifacts_copies_lasso_run_outputs(tmp_path: Path):
    run_dir = tmp_path / "20260325T120000Z"
    rows: list[dict[str, object]] = []
    for index in range(60):
        feature_payload = {
            feature_name: np.float32(((index + feature_offset) % 11) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-LATEST-{index // 6}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.40 + (index % 8) * 0.02,
                **feature_payload,
            }
        )
    frame = pd.DataFrame(rows)
    artifact, metrics = train_lasso_model(frame.iloc[:40], frame.iloc[40:50], {"C": 0.1, "max_iter": 300, "tol": 1e-4})
    save_lasso_artifacts(run_dir, artifact, metrics)
    lasso_dir = run_dir / "lasso"
    (lasso_dir / "calibration.json").write_text('{"a": 1.0, "b": 0.0}', encoding="utf-8")
    (run_dir / "feature_manifest.json").write_text('{"feature_order": ["z_implied"]}', encoding="utf-8")
    (run_dir / "split_manifest.json").write_text('{"train_tickers": ["KXBTC15M-TEST-1"]}', encoding="utf-8")
    (run_dir / "policy.json").write_text('{"config": {"edge_threshold_cents": 1.0}}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"series": "KXBTC15M", "model_family": "lasso"}', encoding="utf-8")

    latest_dir = publish_latest_artifacts(run_dir, model_family="lasso")

    assert (latest_dir / "lasso" / "model.joblib").exists()
    assert json.loads((latest_dir / "run.json").read_text(encoding="utf-8"))["source_run_dir"] == str(run_dir)


def test_publish_latest_artifacts_copies_bagged_lasso_run_outputs(tmp_path: Path):
    run_dir = tmp_path / "20260326T123000Z"
    rows: list[dict[str, object]] = []
    for index in range(72):
        feature_payload = {
            feature_name: np.float32(((index + feature_offset * 4) % 17) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-BAG-LATEST-{index // 6}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.34 + (index % 8) * 0.03,
                **feature_payload,
            }
        )
    frame = pd.DataFrame(rows)
    artifact, metrics = train_bagged_lasso_model(
        frame.iloc[:48],
        frame.iloc[48:60],
        {"C": 0.1, "n_estimators": 5, "max_samples": 0.75, "max_iter": 300, "tol": 1e-4, "n_jobs": 1},
    )
    save_bagged_lasso_artifacts(run_dir, artifact, metrics)
    bagged_dir = run_dir / "bagged_lasso"
    (bagged_dir / "calibration.json").write_text('{"a": 1.0, "b": 0.0}', encoding="utf-8")
    (run_dir / "feature_manifest.json").write_text('{"feature_order": ["z_implied"]}', encoding="utf-8")
    (run_dir / "split_manifest.json").write_text('{"train_tickers": ["KXBTC15M-TEST-1"]}', encoding="utf-8")
    (run_dir / "policy.json").write_text('{"config": {"edge_threshold_cents": 1.0}}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"series": "KXBTC15M", "model_family": "bagged_lasso"}', encoding="utf-8")

    latest_dir = publish_latest_artifacts(run_dir, model_family="bagged_lasso")

    assert (latest_dir / "bagged_lasso" / "model.joblib").exists()
    assert json.loads((latest_dir / "run.json").read_text(encoding="utf-8"))["source_run_dir"] == str(run_dir)


def test_publish_latest_artifacts_copies_elastic_net_run_outputs(tmp_path: Path):
    run_dir = tmp_path / "20260326T120000Z"
    rows: list[dict[str, object]] = []
    for index in range(60):
        feature_payload = {
            feature_name: np.float32(((index + feature_offset * 2) % 13) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-EN-LATEST-{index // 6}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.38 + (index % 8) * 0.03,
                **feature_payload,
            }
        )
    frame = pd.DataFrame(rows)
    artifact, metrics = train_elastic_net_model(
        frame.iloc[:40],
        frame.iloc[40:50],
        {"C": 0.1, "l1_ratio": DEFAULT_ELASTIC_NET_L1_RATIOS[2], "max_iter": 300, "tol": 1e-4},
    )
    save_elastic_net_artifacts(run_dir, artifact, metrics)
    elastic_net_dir = run_dir / "elastic_net"
    (elastic_net_dir / "calibration.json").write_text('{"a": 1.0, "b": 0.0}', encoding="utf-8")
    (run_dir / "feature_manifest.json").write_text('{"feature_order": ["z_implied"]}', encoding="utf-8")
    (run_dir / "split_manifest.json").write_text('{"train_tickers": ["KXBTC15M-TEST-1"]}', encoding="utf-8")
    (run_dir / "policy.json").write_text('{"config": {"edge_threshold_cents": 1.0}}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"series": "KXBTC15M", "model_family": "elastic_net"}', encoding="utf-8")

    latest_dir = publish_latest_artifacts(run_dir, model_family="elastic_net")

    assert (latest_dir / "elastic_net" / "model.joblib").exists()
    assert json.loads((latest_dir / "run.json").read_text(encoding="utf-8"))["source_run_dir"] == str(run_dir)


def test_publish_latest_artifacts_copies_linear_svm_run_outputs(tmp_path: Path):
    run_dir = tmp_path / "20260326T130000Z"
    rows: list[dict[str, object]] = []
    for index in range(60):
        feature_payload = {
            feature_name: np.float32(((index + feature_offset * 3) % 17) / 10.0)
            for feature_offset, feature_name in enumerate(FEATURE_ORDER)
        }
        rows.append(
            {
                "ticker": f"KXBTC15M-SVM-LATEST-{index // 6}",
                "trade_id": f"trade-{index}",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(seconds=index),
                "open_time": pd.Timestamp("2026-01-01T11:45:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:15:00Z"),
                "actual_outcome": index % 2,
                "market_prob": 0.36 + (index % 8) * 0.03,
                **feature_payload,
            }
        )
    frame = pd.DataFrame(rows)
    artifact, metrics = train_linear_svm_model(
        frame.iloc[:40],
        frame.iloc[40:50],
        {"C": DEFAULT_LINEAR_SVM_C_VALUES[1], "max_iter": 300, "tol": 1e-4},
    )
    save_linear_svm_artifacts(run_dir, artifact, metrics)
    svm_dir = run_dir / "linear_svm"
    (svm_dir / "calibration.json").write_text('{"a": 1.0, "b": 0.0}', encoding="utf-8")
    (run_dir / "feature_manifest.json").write_text('{"feature_order": ["z_implied"]}', encoding="utf-8")
    (run_dir / "split_manifest.json").write_text('{"train_tickers": ["KXBTC15M-TEST-1"]}', encoding="utf-8")
    (run_dir / "policy.json").write_text('{"config": {"edge_threshold_cents": 1.0}}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"series": "KXBTC15M", "model_family": "linear_svm"}', encoding="utf-8")

    latest_dir = publish_latest_artifacts(run_dir, model_family="linear_svm")

    assert (latest_dir / "linear_svm" / "model.joblib").exists()
    assert json.loads((latest_dir / "run.json").read_text(encoding="utf-8"))["source_run_dir"] == str(run_dir)


def test_build_policy_diagnostics_reports_yes_and_no_breakdowns():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-A",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:05:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
            },
            {
                "ticker": "KXBTC15M-B",
                "trade_id": "t2",
                "created_time": pd.Timestamp("2026-01-01T12:01:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:06:00Z"),
                "actual_outcome": 0,
                "market_prob": 0.60,
                "tau_minutes": 5.0,
            },
            {
                "ticker": "KXBTC15M-C",
                "trade_id": "t3",
                "created_time": pd.Timestamp("2026-01-01T12:02:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:07:00Z"),
                "actual_outcome": 0,
                "market_prob": 0.35,
                "tau_minutes": 5.0,
            },
            {
                "ticker": "KXBTC15M-D",
                "trade_id": "t4",
                "created_time": pd.Timestamp("2026-01-01T12:03:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:08:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.65,
                "tau_minutes": 5.0,
            },
        ]
    )
    raw_probabilities = np.array([0.70, 0.30, 0.60, 0.20], dtype=np.float64)
    calibrated_probabilities = raw_probabilities.copy()
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
    )

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        raw_probabilities,
        calibrated_probabilities,
        config,
        overall_log_loss=0.5,
        time_blocks=2,
    )

    assert len(trade_records) == 4
    side_breakdown = {row["side"]: row for row in diagnostics["side_breakdown"]}
    assert set(side_breakdown) == {"YES", "NO"}
    assert side_breakdown["YES"]["trades"] == 2
    assert side_breakdown["NO"]["trades"] == 2
    assert diagnostics["tau_bucket_breakdown"]
    assert diagnostics["price_bucket_breakdown"]
    assert diagnostics["time_block_breakdown"]
    assert diagnostics["calibration"]["raw_probability_buckets"]
    assert diagnostics["calibration"]["calibrated_probability_buckets"]


def test_build_policy_diagnostics_adds_realized_vol_and_time_slices(monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("src.live.kalshi.offline_training")
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-A",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:05:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
            },
            {
                "ticker": "KXBTC15M-B",
                "trade_id": "t2",
                "created_time": pd.Timestamp("2026-01-01T12:01:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:06:00Z"),
                "actual_outcome": 0,
                "market_prob": 0.60,
                "tau_minutes": 5.0,
            },
            {
                "ticker": "KXBTC15M-C",
                "trade_id": "t3",
                "created_time": pd.Timestamp("2026-01-01T12:02:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:07:00Z"),
                "actual_outcome": 0,
                "market_prob": 0.35,
                "tau_minutes": 5.0,
            },
            {
                "ticker": "KXBTC15M-D",
                "trade_id": "t4",
                "created_time": pd.Timestamp("2026-01-01T12:03:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:08:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.65,
                "tau_minutes": 5.0,
            },
        ]
    )
    raw_probabilities = np.array([0.70, 0.30, 0.60, 0.20], dtype=np.float64)
    calibrated_probabilities = raw_probabilities.copy()
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
    )

    def _fake_annotate(frame: pd.DataFrame, *, external_spot_root=None):
        annotated = frame.copy()
        annotated["realized_vol_regime_source"] = ["spot", "spot", "market_prob", "market_prob"]
        annotated["realized_vol_regime_bucket"] = ["low", "high", "medium", "extreme"]
        annotated["realized_vol_regime_bucket_version"] = [
            "spot_2026q1_backfill_v1",
            "spot_2026q1_backfill_v1",
            "market_prob_2025q4_2026q1_proxy_v1",
            "market_prob_2025q4_2026q1_proxy_v1",
        ]
        annotated["realized_vol_regime_metric"] = [0.30, 0.70, 110.0, 150.0]
        annotated["hour_of_day_et"] = [7, 9, 14, 21]
        annotated["hour_of_day_et_label"] = ["07", "09", "14", "21"]
        annotated["session_block_et"] = ["us_preopen", "us_regular", "us_regular", "asia"]
        metadata = {
            "row_source_summary": [
                {"realized_vol_regime_source": "spot", "rows": 2, "row_pct": 50.0},
                {"realized_vol_regime_source": "market_prob", "rows": 2, "row_pct": 50.0},
                {"realized_vol_regime_source": "unknown", "rows": 0, "row_pct": 0.0},
            ],
            "bucket_versions": {
                "spot": {"version": "spot_2026q1_backfill_v1", "edges": [0.4, 0.6, 0.9]},
                "market_prob": {
                    "version": "market_prob_2025q4_2026q1_proxy_v1",
                    "edges": [100.0, 125.0, 140.0],
                },
            },
            "timezone_metadata": {
                "timezone": "America/New_York",
                "zoneinfo_source": "tzdata",
                "tzdata_version": "2024.2",
                "zoneinfo_paths": [],
            },
        }
        return annotated, metadata

    monkeypatch.setattr(module, "_annotate_diagnostic_context", _fake_annotate)

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        raw_probabilities,
        calibrated_probabilities,
        config,
        overall_log_loss=0.5,
        time_blocks=2,
    )

    assert diagnostics["realized_vol_regime_metadata"]["bucket_versions"]["spot"]["edges"] == [0.4, 0.6, 0.9]
    assert diagnostics["realized_vol_regime_metadata"]["timezone_metadata"]["tzdata_version"] == "2024.2"
    assert diagnostics["realized_vol_regime_breakdown"]
    assert diagnostics["hour_of_day_breakdown"]
    assert diagnostics["session_block_breakdown"]
    assert {"realized_vol_regime_source", "realized_vol_regime_bucket", "session_block_et"} <= set(trade_records.columns)
    hour_row = next(row for row in diagnostics["hour_of_day_breakdown"] if row["hour_of_day_et"] == 9)
    assert hour_row["rows"] == 1
    assert hour_row["row_pct"] == pytest.approx(25.0)


def test_build_policy_diagnostics_supports_capital_percent_position_sizing():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-SIZE",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:05:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
            }
        ]
    )
    raw_probabilities = np.array([0.70], dtype=np.float64)
    calibrated_probabilities = raw_probabilities.copy()
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
        starting_cash_dollars=100.0,
        contracts_per_order=1,
        capital_pct_per_order=2.0,
    )

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        raw_probabilities,
        calibrated_probabilities,
        config,
        overall_log_loss=0.5,
    )

    assert len(trade_records) == 1
    assert int(trade_records.iloc[0]["contracts"]) > 1
    assert diagnostics["side_breakdown"][0]["avg_contracts"] > 1.0


def test_build_prediction_export_frame_adds_realized_vol_and_time_columns(monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("src.live.kalshi.offline_training")
    frame = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-A",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:05:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
            }
        ]
    )

    def _fake_annotate(split_df: pd.DataFrame, *, external_spot_root=None):
        annotated = split_df.copy()
        annotated["realized_vol_regime_source"] = ["spot"]
        annotated["realized_vol_regime_bucket"] = ["medium"]
        annotated["realized_vol_regime_bucket_version"] = ["spot_2026q1_backfill_v1"]
        annotated["realized_vol_regime_metric"] = [0.55]
        annotated["hour_of_day_et"] = [9]
        annotated["hour_of_day_et_label"] = ["09"]
        annotated["session_block_et"] = ["us_regular"]
        return annotated, {"timezone_metadata": {"timezone": "America/New_York"}}

    monkeypatch.setattr(module, "_annotate_diagnostic_context", _fake_annotate)

    export = build_prediction_export_frame(
        frame,
        np.array([0.7], dtype=np.float64),
        np.array([0.6], dtype=np.float64),
    )

    assert export.loc[0, "realized_vol_regime_source"] == "spot"
    assert export.loc[0, "realized_vol_regime_bucket"] == "medium"
    assert export.loc[0, "hour_of_day_et"] == 9
    assert export.loc[0, "session_block_et"] == "us_regular"


def test_build_policy_diagnostics_supports_kelly_position_sizing():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-KELLY",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:05:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
            }
        ]
    )
    probabilities = np.array([0.70], dtype=np.float64)
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
        starting_cash_dollars=100.0,
        contracts_per_order=1,
        kelly_fraction_multiplier=1.0,
        kelly_fraction_cap_pct=10.0,
    )

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        probabilities,
        probabilities,
        config,
        overall_log_loss=0.5,
    )

    assert len(trade_records) == 1
    assert trade_records.iloc[0]["sizing_method"] == "kelly"
    assert float(trade_records.iloc[0]["kelly_fraction_of_equity"]) > 0.0
    assert float(trade_records.iloc[0]["target_fraction_of_equity"]) <= 0.10 + 1e-12
    assert diagnostics["side_breakdown"][0]["avg_kelly_fraction_of_equity"] is not None


def test_build_policy_diagnostics_limits_stacking_by_max_entries_per_ticker():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-STACK",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.60,
                "tau_minutes": 10.0,
            },
            {
                "ticker": "KXBTC15M-STACK",
                "trade_id": "t2",
                "created_time": pd.Timestamp("2026-01-01T12:01:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.58,
                "tau_minutes": 9.0,
            },
            {
                "ticker": "KXBTC15M-STACK",
                "trade_id": "t3",
                "created_time": pd.Timestamp("2026-01-01T12:02:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.56,
                "tau_minutes": 8.0,
            },
        ]
    )
    probabilities = np.array([0.80, 0.80, 0.80], dtype=np.float64)
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
        allow_stacking=True,
        max_entries_per_ticker=2,
    )

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        probabilities,
        probabilities,
        config,
        overall_log_loss=0.5,
    )

    assert len(trade_records) == 2
    assert list(trade_records["trade_id"]) == ["t1", "t2"]
    assert diagnostics["policy_metrics"]["skipped_due_open_ticker"] == 1


def test_build_policy_diagnostics_requires_price_improvement_for_stack():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-STACK",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.55,
                "tau_minutes": 10.0,
            },
            {
                "ticker": "KXBTC15M-STACK",
                "trade_id": "t2",
                "created_time": pd.Timestamp("2026-01-01T12:01:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:10:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.57,
                "tau_minutes": 9.0,
            },
        ]
    )
    probabilities = np.array([0.80, 0.80], dtype=np.float64)
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
        allow_stacking=True,
        max_entries_per_ticker=3,
        require_price_improvement_for_stack=True,
    )

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        probabilities,
        probabilities,
        config,
        overall_log_loss=0.5,
    )

    assert len(trade_records) == 1
    assert list(trade_records["trade_id"]) == ["t1"]
    assert diagnostics["policy_metrics"]["skipped_due_open_ticker"] == 1


def test_build_policy_diagnostics_adds_regime_fields_and_counterfactual():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-DOWN",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-01-01T12:00:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:05:00Z"),
                "actual_outcome": 0,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
                "price_momentum": -0.03,
                "signed_contracts_sum_300s": -6.0,
                "yes_taker_share_300s": 0.30,
            },
            {
                "ticker": "KXBTC15M-NEUTRAL",
                "trade_id": "t2",
                "created_time": pd.Timestamp("2026-01-01T12:01:00Z"),
                "close_time": pd.Timestamp("2026-01-01T12:06:00Z"),
                "actual_outcome": 1,
                "market_prob": 0.40,
                "tau_minutes": 5.0,
                "price_momentum": 0.00,
                "signed_contracts_sum_300s": 0.0,
                "yes_taker_share_300s": 0.50,
            },
        ]
    )
    probabilities = np.array([0.70, 0.70], dtype=np.float64)
    config = PolicyConfig(
        edge_threshold_cents=0.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=1,
        price_band_max_cents=99,
        reserve_cash_pct=0.0,
    )

    diagnostics, trade_records = build_policy_diagnostics(
        df,
        probabilities,
        probabilities,
        config,
        overall_log_loss=0.5,
    )

    assert set(trade_records["regime_label"]) == {"downtrend", "neutral"}
    assert diagnostics["regime_breakdown"]
    assert diagnostics["regime_side_breakdown"]
    assert diagnostics["regime_row_counts"]
    assert diagnostics["regime_counterfactuals"]["downtrend_blocks_yes"]["skipped_due_regime"] == 1
    assert diagnostics["regime_counterfactuals"]["downtrend_blocks_yes"]["trades"] == 1


def test_publish_latest_artifacts_copies_run_outputs(tmp_path: Path):
    run_dir = tmp_path / "20260323T120000Z"
    lightgbm_dir = run_dir / "lightgbm"
    lightgbm_dir.mkdir(parents=True, exist_ok=True)
    (lightgbm_dir / "model.txt").write_text("model", encoding="utf-8")
    (lightgbm_dir / "metrics.json").write_text('{"best_iteration": 42}', encoding="utf-8")
    (lightgbm_dir / "feature_importance.json").write_text('{"z_implied": 1.0}', encoding="utf-8")
    (lightgbm_dir / "calibration.json").write_text('{"a": 1.0, "b": 0.0}', encoding="utf-8")
    (run_dir / "feature_manifest.json").write_text('{"feature_order": ["z_implied"]}', encoding="utf-8")
    (run_dir / "split_manifest.json").write_text('{"train_tickers": ["KXBTC15M-TEST-1"]}', encoding="utf-8")
    (run_dir / "policy.json").write_text('{"config": {"edge_threshold_cents": 1.0}}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"series": "KXBTC15M"}', encoding="utf-8")

    latest_dir = publish_latest_artifacts(run_dir)

    assert (latest_dir / "lightgbm" / "model.txt").read_text(encoding="utf-8") == "model"
    assert (latest_dir / "policy.json").read_text(encoding="utf-8") == '{"config": {"edge_threshold_cents": 1.0}}'
    published = json.loads((latest_dir / "run.json").read_text(encoding="utf-8"))
    assert published["source_run_dir"] == str(run_dir)


def test_resolve_inputs_reports_missing_explicit_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.live.kalshi.offline_training._candidate_input_pairs",
        lambda _series: [
            (
                Path("output/kalshi_series/KXBTC15M_markets.parquet"),
                Path("output/kalshi_series/KXBTC15M_trades.parquet"),
            )
        ],
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_inputs("KXBTC15M", "data/markets.parquet", "data/trades")

    message = str(exc_info.value)
    assert 'Provided --markets-path does not exist: "data\\markets.parquet"' in message
    assert 'Provided --trades-path does not exist: "data\\trades"' in message
    assert "Detected local input candidates:" in message
    assert '--markets-path "output\\kalshi_series\\KXBTC15M_markets.parquet"' in message


def test_select_policy_candidate_prefers_strict_candidates():
    config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )
    weak = PolicyResult(
        config=config,
        objective=2.0,
        trades=100,
        net_pnl_dollars=10.0,
        max_drawdown_dollars=5.0,
        max_drawdown_pct=1.0,
        return_pct=0.1,
        log_loss=0.5,
        skipped_due_open_ticker=0,
        skipped_due_price_band=0,
        skipped_due_post_cost_edge=0,
    )
    strong = PolicyResult(
        config=config,
        objective=1.5,
        trades=600,
        net_pnl_dollars=8.0,
        max_drawdown_dollars=5.0,
        max_drawdown_pct=1.0,
        return_pct=0.08,
        log_loss=0.49,
        skipped_due_open_ticker=0,
        skipped_due_price_band=0,
        skipped_due_post_cost_edge=0,
    )

    selected, met_requirements = select_policy_candidate([weak, strong], minimum_trades=500)

    assert met_requirements is True
    assert selected is strong


def test_minimum_policy_trades_for_frame_scales_with_validation_span():
    frame = pd.DataFrame(
        {
            "created_time": pd.to_datetime(
                [
                    "2026-02-13T07:15:25.919130Z",
                    "2026-02-27T04:14:13.762032Z",
                ],
                utc=True,
            )
        }
    )

    minimum_trades = minimum_policy_trades_for_frame(frame, trades_per_day=MINIMUM_POLICY_TRADES_PER_DAY)

    assert minimum_trades == 139


def test_standard_policy_fast_path_matches_generic_simulator():
    config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=12.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )
    frame = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-A",
                "trade_id": "a1",
                "created_time": pd.Timestamp("2026-02-13T00:00:00Z"),
                "close_time": pd.Timestamp("2026-02-13T00:15:00Z"),
                "market_prob": 0.40,
                "tau_minutes": 11.0,
                "actual_outcome": 1,
            },
            {
                "ticker": "KXBTC15M-A",
                "trade_id": "a2",
                "created_time": pd.Timestamp("2026-02-13T00:01:00Z"),
                "close_time": pd.Timestamp("2026-02-13T00:15:00Z"),
                "market_prob": 0.38,
                "tau_minutes": 10.0,
                "actual_outcome": 1,
            },
            {
                "ticker": "KXBTC15M-B",
                "trade_id": "b1",
                "created_time": pd.Timestamp("2026-02-13T00:02:00Z"),
                "close_time": pd.Timestamp("2026-02-13T00:15:00Z"),
                "market_prob": 0.62,
                "tau_minutes": 9.0,
                "actual_outcome": 0,
            },
            {
                "ticker": "KXBTC15M-C",
                "trade_id": "c1",
                "created_time": pd.Timestamp("2026-02-13T00:16:00Z"),
                "close_time": pd.Timestamp("2026-02-13T00:30:00Z"),
                "market_prob": 0.58,
                "tau_minutes": 8.0,
                "actual_outcome": 0,
            },
        ]
    )
    probabilities = np.array([0.70, 0.72, 0.30, 0.25], dtype=np.float64)

    prepared = _prepare_standard_policy_inputs(frame, probabilities, slippage=config.signal_config.slippage)
    fast = _simulate_standard_policy_prepared(prepared, config, overall_log_loss=0.5)
    slow = _simulate_policy(frame, probabilities, config, overall_log_loss=0.5)
    evaluated = evaluate_policy(frame, probabilities, config, overall_log_loss=0.5)

    assert fast.trades == slow.trades
    assert fast.skipped_due_open_ticker == slow.skipped_due_open_ticker
    assert fast.skipped_due_price_band == slow.skipped_due_price_band
    assert fast.skipped_due_post_cost_edge == slow.skipped_due_post_cost_edge
    assert fast.net_pnl_dollars == pytest.approx(slow.net_pnl_dollars)
    assert fast.max_drawdown_dollars == pytest.approx(slow.max_drawdown_dollars)
    assert fast.max_drawdown_pct == pytest.approx(slow.max_drawdown_pct)
    assert fast.return_pct == pytest.approx(slow.return_pct)
    assert fast.objective == pytest.approx(slow.objective)
    assert evaluated["trades"] == slow.trades
    assert evaluated["net_pnl_dollars"] == pytest.approx(slow.net_pnl_dollars)
    assert evaluated["objective"] == pytest.approx(slow.objective)


def test_fast_policy_reselection_script_updates_policy_without_full_sweep(tmp_path: Path):
    run_dir = tmp_path / "lasso-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    validation_df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-V1",
                "trade_id": "v1",
                "created_time": pd.Timestamp("2026-02-13T00:00:00Z"),
                "close_time": pd.Timestamp("2026-02-13T00:15:00Z"),
                "market_prob": 0.40,
                "tau_minutes": 5.0,
                "actual_outcome": 1,
            },
            {
                "ticker": "KXBTC15M-V2",
                "trade_id": "v2",
                "created_time": pd.Timestamp("2026-02-14T00:00:00Z"),
                "close_time": pd.Timestamp("2026-02-14T00:15:00Z"),
                "market_prob": 0.60,
                "tau_minutes": 5.0,
                "actual_outcome": 0,
            },
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-T1",
                "trade_id": "t1",
                "created_time": pd.Timestamp("2026-02-15T00:00:00Z"),
                "close_time": pd.Timestamp("2026-02-15T00:15:00Z"),
                "market_prob": 0.42,
                "tau_minutes": 5.0,
                "actual_outcome": 1,
            },
            {
                "ticker": "KXBTC15M-T2",
                "trade_id": "t2",
                "created_time": pd.Timestamp("2026-02-16T00:00:00Z"),
                "close_time": pd.Timestamp("2026-02-16T00:15:00Z"),
                "market_prob": 0.58,
                "tau_minutes": 5.0,
                "actual_outcome": 0,
            },
        ]
    )

    build_prediction_export_frame(
        validation_df,
        np.array([0.70, 0.30], dtype=np.float64),
        np.array([0.70, 0.30], dtype=np.float64),
    ).to_parquet(run_dir / "validation_predictions.parquet", index=False)
    build_prediction_export_frame(
        test_df,
        np.array([0.72, 0.28], dtype=np.float64),
        np.array([0.72, 0.28], dtype=np.float64),
    ).to_parquet(run_dir / "test_predictions.parquet", index=False)

    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "series": "KXBTC15M",
                "train_tickers": ["KXBTC15M-TRAIN"],
                "validation_tickers": ["KXBTC15M-V1", "KXBTC15M-V2"],
                "test_tickers": ["KXBTC15M-T1", "KXBTC15M-T2"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"series": "KXBTC15M", "model_family": "lasso", "walk_forward_completed": False}),
        encoding="utf-8",
    )

    strict_config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )
    rejected_config = PolicyConfig(
        edge_threshold_cents=6.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )
    policy_search = [
        serialize_policy_result(
            PolicyResult(
                config=rejected_config,
                objective=9.0,
                trades=9,
                net_pnl_dollars=9.0,
                max_drawdown_dollars=1.0,
                max_drawdown_pct=1.0,
                return_pct=0.09,
                log_loss=0.4,
                skipped_due_open_ticker=0,
                skipped_due_price_band=0,
                skipped_due_post_cost_edge=0,
            )
        ),
        serialize_policy_result(
            PolicyResult(
                config=strict_config,
                objective=5.0,
                trades=11,
                net_pnl_dollars=5.0,
                max_drawdown_dollars=1.0,
                max_drawdown_pct=1.0,
                return_pct=0.05,
                log_loss=0.41,
                skipped_due_open_ticker=0,
                skipped_due_price_band=0,
                skipped_due_post_cost_edge=0,
            )
        ),
    ]
    (run_dir / "policy_search.json").write_text(json.dumps(policy_search), encoding="utf-8")

    command = [
        sys.executable,
        str(Path("scripts") / "reselect_kxbtc15m_policy.py"),
        "--run-dir",
        str(run_dir),
        "--skip-latest-publish",
        "--minimum-trades-per-day",
        "10",
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    policy = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    validation_predictions = pd.read_parquet(run_dir / "validation_predictions.parquet")

    assert policy["config"]["edge_threshold_cents"] == 1.0
    assert policy["minimum_trades_required"] == 10
    assert summary["minimum_validation_trades_required"] == 10
    assert "close_time" in validation_predictions.columns


def test_fast_policy_reselection_script_rebuilds_missing_predictions_from_saved_model(tmp_path: Path):
    run_dir = tmp_path / "lasso-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def split_frame(ticker: str, start: str, outcomes: list[int]) -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz="UTC")
        rows: list[dict[str, object]] = []
        for index, outcome in enumerate(outcomes):
            created_time = start_ts + pd.Timedelta(minutes=index)
            close_time = created_time + pd.Timedelta(minutes=15)
            row: dict[str, object] = {
                "ticker": ticker,
                "trade_id": f"{ticker}-{index}",
                "created_time": created_time,
                "open_time": created_time - pd.Timedelta(minutes=15),
                "close_time": close_time,
                "actual_outcome": int(outcome),
                "market_prob": 0.62 if outcome == 1 else 0.38,
                "tau_minutes": float(12 - index),
                "count": 1 + index,
                "taker_side": "yes" if outcome == 1 else "no",
            }
            signal = 1.0 if outcome == 1 else -1.0
            for feature_index, feature_name in enumerate(FEATURE_ORDER):
                if feature_name == "tau_minutes":
                    value = float(row["tau_minutes"])
                elif feature_name == "z_implied":
                    value = signal * 0.8
                elif feature_name == "price_direction":
                    value = signal
                elif feature_name == "distance_from_mid":
                    value = 0.12 + (0.01 * index)
                else:
                    value = signal * (0.05 + (0.01 * feature_index)) + (0.001 * index)
                row[feature_name] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    train_df = pd.concat(
        [
            split_frame("KXBTC15M-TRAIN-A", "2026-02-01T00:00:00Z", [1, 0, 1, 0, 1, 0]),
            split_frame("KXBTC15M-TRAIN-B", "2026-02-02T00:00:00Z", [0, 1, 0, 1, 0, 1]),
        ],
        ignore_index=True,
    )
    validation_df = split_frame("KXBTC15M-VAL", "2026-02-13T00:00:00Z", [1, 0, 1, 0])
    test_df = split_frame("KXBTC15M-TEST", "2026-02-15T00:00:00Z", [0, 1, 0, 1])

    for frame in (validation_df, test_df):
        frame.to_parquet(cache_dir / f"{frame['ticker'].iloc[0]}.parquet", index=False)

    params = {
        "C": 0.1,
        "penalty": "elasticnet",
        "l1_ratio": 1.0,
        "solver": "saga",
        "fit_intercept": True,
        "max_iter": 2000,
        "tol": 1e-4,
    }
    artifact, metrics = train_lasso_model(train_df, validation_df, params)
    save_lasso_artifacts(run_dir, artifact, metrics)
    validation_raw = lasso_raw_predictions(artifact, validation_df)
    calibrate_validation_predictions(run_dir, validation_df, validation_raw, model_family="lasso")

    build_prediction_export_frame(
        test_df,
        lasso_raw_predictions(artifact, test_df),
        lasso_raw_predictions(artifact, test_df),
    ).to_parquet(run_dir / "test_predictions.parquet", index=False)

    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "series": "KXBTC15M",
                "train_tickers": ["KXBTC15M-TRAIN-A", "KXBTC15M-TRAIN-B"],
                "validation_tickers": ["KXBTC15M-VAL"],
                "test_tickers": ["KXBTC15M-TEST"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"series": "KXBTC15M", "model_family": "lasso", "walk_forward_completed": False}),
        encoding="utf-8",
    )
    (run_dir / "policy_search.json").write_text(
        json.dumps(
            [
                serialize_policy_result(
                    PolicyResult(
                        config=PolicyConfig(
                            edge_threshold_cents=1.0,
                            min_tau_minutes=0.0,
                            max_tau_minutes=15.0,
                            price_band_min_cents=10,
                            price_band_max_cents=90,
                            reserve_cash_pct=30.0,
                        ),
                        objective=1.0,
                        trades=2,
                        net_pnl_dollars=2.0,
                        max_drawdown_dollars=1.0,
                        max_drawdown_pct=0.01,
                        return_pct=0.02,
                        log_loss=0.5,
                        skipped_due_open_ticker=0,
                        skipped_due_price_band=0,
                        skipped_due_post_cost_edge=0,
                    )
                )
            ]
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(Path("scripts") / "reselect_kxbtc15m_policy.py"),
        "--run-dir",
        str(run_dir),
        "--dataset-cache-dir",
        str(cache_dir),
        "--skip-latest-publish",
        "--minimum-trades",
        "1",
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    validation_predictions = pd.read_parquet(run_dir / "validation_predictions.parquet")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert "close_time" in validation_predictions.columns
    assert set(validation_predictions["ticker"]) == {"KXBTC15M-VAL"}
    assert summary["minimum_validation_trades_required"] == 1


def test_lightgbm_trainer_writes_bagged_lasso_diagnostic_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = importlib.import_module("scripts.train_kxbtc15m_lightgbm")

    artifacts_root = tmp_path / "artifacts"
    run_name = "lightgbm_diag_contract"
    run_dir = artifacts_root / run_name
    dataset_cache_dir = tmp_path / "cache"
    markets_path = tmp_path / "markets.parquet"
    trades_path = tmp_path / "trades.parquet"

    markets_df = pd.DataFrame(
        {
            "ticker": [f"KXBTC15M-TEST-{index:02d}" for index in range(6)],
            "result": ["yes", "no", "yes", "no", "yes", "no"],
            "open_time": pd.date_range("2026-03-10T00:00:00Z", periods=6, freq="15min"),
            "close_time": pd.date_range("2026-03-10T00:15:00Z", periods=6, freq="15min"),
        }
    )

    split_manifest = SplitManifest(
        series="KXBTC15M",
        generated_at="2026-04-21T00:00:00+00:00",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        train_tickers=("KXBTC15M-TEST-00", "KXBTC15M-TEST-01", "KXBTC15M-TEST-02"),
        validation_tickers=("KXBTC15M-TEST-03",),
        test_tickers=("KXBTC15M-TEST-04", "KXBTC15M-TEST-05"),
    )

    def _frame_for_tickers(tickers: tuple[str, ...]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, ticker in enumerate(tickers):
            created_time = pd.Timestamp("2026-03-10T01:00:00Z") + pd.Timedelta(minutes=index)
            close_time = created_time + pd.Timedelta(minutes=15)
            rows.append(
                {
                    "ticker": ticker,
                    "trade_id": f"{ticker}-trade",
                    "created_time": created_time,
                    "open_time": created_time - pd.Timedelta(minutes=15),
                    "close_time": close_time,
                    "actual_outcome": int(index % 2 == 0),
                    "market_prob": 0.55 if index % 2 == 0 else 0.45,
                    "tau_minutes": 5.0,
                    "count": 1,
                    "taker_side": "yes" if index % 2 == 0 else "no",
                    "feature_a": 0.1 + index,
                    "feature_b": 0.2 + index,
                }
            )
        return pd.DataFrame(rows)

    best_policy = PolicyResult(
        config=PolicyConfig(
            edge_threshold_cents=1.0,
            min_tau_minutes=0.0,
            max_tau_minutes=15.0,
            price_band_min_cents=10,
            price_band_max_cents=90,
            reserve_cash_pct=30.0,
        ),
        objective=1.0,
        trades=12,
        net_pnl_dollars=2.5,
        max_drawdown_dollars=1.0,
        max_drawdown_pct=0.1,
        return_pct=0.025,
        log_loss=0.49,
        skipped_due_open_ticker=0,
        skipped_due_price_band=0,
        skipped_due_regime=0,
        skipped_due_post_cost_edge=0,
    )

    def _prediction_export(frame: pd.DataFrame, raw_probabilities, calibrated_probabilities) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": frame["ticker"],
                "trade_id": frame["trade_id"],
                "created_time": frame["created_time"],
                "close_time": frame["close_time"],
                "actual_outcome": frame["actual_outcome"],
                "raw_probability": np.asarray(raw_probabilities, dtype=float),
                "calibrated_probability": np.asarray(calibrated_probabilities, dtype=float),
            }
        )

    def _diagnostics(frame: pd.DataFrame, raw_probabilities, calibrated_probabilities, config, *, overall_log_loss, progress_desc=None):
        trade_records = pd.DataFrame(
            {
                "ticker": frame["ticker"],
                "created_time": frame["created_time"],
                "close_time": frame["close_time"],
                "side": ["YES"] * len(frame),
                "reference_price_cents": [55] * len(frame),
                "tau_minutes": [5.0] * len(frame),
                "net_pnl_dollars": [1.0] * len(frame),
            }
        )
        diagnostics = {
            "policy_metrics": serialize_policy_result(
                PolicyResult(
                    config=config,
                    objective=1.0,
                    trades=len(frame),
                    net_pnl_dollars=float(len(frame)),
                    max_drawdown_dollars=1.0,
                    max_drawdown_pct=0.1,
                    return_pct=0.01,
                    log_loss=float(overall_log_loss),
                    skipped_due_open_ticker=0,
                    skipped_due_price_band=0,
                    skipped_due_regime=0,
                    skipped_due_post_cost_edge=0,
                )
            ),
            "side_breakdown": [],
            "calibration": {},
        }
        return diagnostics, trade_records

    monkeypatch.setattr(module, "resolve_inputs", lambda *args, **kwargs: (markets_path, trades_path))
    monkeypatch.setattr(module, "load_markets_frame", lambda path, series: markets_df.copy())
    monkeypatch.setattr(module, "build_split_manifest", lambda *args, **kwargs: split_manifest)
    monkeypatch.setattr(
        module,
        "inspect_feature_cache",
        lambda *args, **kwargs: FeatureCacheStatus(
            ready=False,
            manifest_exists=False,
            manifest_schema_name=None,
            schema_matches=True,
            requested_tickers=6,
            present_requested_tickers=0,
            missing_requested_tickers=tuple(),
        ),
    )
    monkeypatch.setattr(module, "build_feature_cache", lambda *args, **kwargs: dataset_cache_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "load_feature_dataset", lambda cache_dir, tickers, **kwargs: _frame_for_tickers(tuple(tickers)))
    monkeypatch.setattr(module, "infer_feature_names", lambda frame: ["feature_a", "feature_b"])
    monkeypatch.setattr(
        module,
        "save_feature_manifest",
        lambda path, feature_order, metadata: path.write_text(
            json.dumps({"feature_order": feature_order, "metadata": metadata}),
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        module,
        "run_optuna_search",
        lambda train_subset, validation_df, n_trials=40: ({"n_estimators": 10}, [{"trial": 0, "value": 0.5}]),
    )
    monkeypatch.setattr(module, "train_lightgbm_model", lambda train_df, validation_df, params: (object(), {"best_iteration": 7}))
    monkeypatch.setattr(
        module,
        "save_lightgbm_artifacts",
        lambda artifacts_dir, model, metrics: (artifacts_dir / "lightgbm").mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(module, "raw_predictions", lambda model, frame: np.linspace(0.4, 0.6, num=len(frame), dtype=float))
    monkeypatch.setattr(
        module,
        "calibrate_validation_predictions",
        lambda artifacts_dir, validation_df, validation_raw: np.asarray(validation_raw, dtype=float),
    )
    monkeypatch.setattr(module, "sweep_policy_grid", lambda validation_df, probabilities, overall_log_loss, progress_desc=None: [best_policy])
    monkeypatch.setattr(
        module,
        "select_policy_candidate",
        lambda policy_results, minimum_trades, fallback_to_best_overall=True: (best_policy, True),
    )
    monkeypatch.setattr(module, "build_prediction_export_frame", _prediction_export, raising=False)
    monkeypatch.setattr(module, "build_policy_diagnostics", _diagnostics, raising=False)
    monkeypatch.setattr(
        module,
        "apply_calibration_to_predictions",
        lambda artifacts_dir, raw_probabilities: np.asarray(raw_probabilities, dtype=float),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_kxbtc15m_lightgbm.py",
            "--series",
            "KXBTC15M",
            "--markets-path",
            str(markets_path),
            "--trades-path",
            str(trades_path),
            "--artifacts-root",
            str(artifacts_root),
            "--run-name",
            run_name,
            "--skip-walk-forward",
            "--skip-latest-publish",
        ],
    )

    module.main()

    assert (run_dir / "validation_metrics.json").exists()
    assert (run_dir / "validation_diagnostics.json").exists()
    assert (run_dir / "validation_predictions.parquet").exists()
    assert (run_dir / "validation_trade_records.parquet").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "test_diagnostics.json").exists()
    assert (run_dir / "test_predictions.parquet").exists()
    assert (run_dir / "test_trade_records.parquet").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_family"] == "lightgbm"
    assert summary["validation_metrics"]["trades"] == 1


def test_select_policy_candidate_can_fall_back_to_best_overall():
    config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=15.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )
    losing = PolicyResult(
        config=config,
        objective=-0.5,
        trades=800,
        net_pnl_dollars=-5.0,
        max_drawdown_dollars=10.0,
        max_drawdown_pct=1.0,
        return_pct=-0.05,
        log_loss=0.5,
        skipped_due_open_ticker=0,
        skipped_due_price_band=0,
        skipped_due_post_cost_edge=0,
    )
    thin = PolicyResult(
        config=config,
        objective=3.0,
        trades=10,
        net_pnl_dollars=20.0,
        max_drawdown_dollars=2.0,
        max_drawdown_pct=1.0,
        return_pct=0.2,
        log_loss=0.45,
        skipped_due_open_ticker=0,
        skipped_due_price_band=0,
        skipped_due_post_cost_edge=0,
    )

    selected, met_requirements = select_policy_candidate(
        [losing, thin],
        minimum_trades=500,
        fallback_to_best_overall=True,
    )

    assert met_requirements is False
    assert selected is losing
