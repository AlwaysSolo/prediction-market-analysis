from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.indexers.kalshi.models import Market
from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiExecutionMode,
    KalshiExecutionUpdate,
    KalshiFeatureStateEngine,
    KalshiLayeringDecision,
    KalshiLightGBMScoreUpdate,
    KalshiLiveArchiveConfig,
    KalshiLiveArchiveManager,
    KalshiLiveArchiveRuntime,
    KalshiMarketDataCollector,
    KalshiRawStreamEvent,
    KalshiSignalDecisionUpdate,
    KalshiTickerState,
    KalshiTradeIntent,
    KalshiTickerUpdate,
    repair_live_archive,
)
from src.live.kalshi.features import feature_state_from_ticker_update, feature_update_from_state
from src.live.kalshi.types import state_to_update


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.PRODUCTION,
        credentials=KalshiCredentials(api_key_id="prod-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
        max_stale_message_age_seconds=0.0,
    )


def _market(ticker: str) -> Market:
    return Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="open",
        yes_bid=54,
        yes_ask=56,
        no_bid=44,
        no_ask=46,
        last_price=55,
        volume=12,
        volume_24h=12,
        open_interest=5,
        result="",
        created_time=None,
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
    )


def _archive_manager(tmp_path: Path, collector: KalshiMarketDataCollector) -> KalshiLiveArchiveManager:
    feature_engine = KalshiFeatureStateEngine(collector)
    return KalshiLiveArchiveManager(
        collector,
        feature_engine,
        [],
        KalshiLiveArchiveConfig(
            archive_root=tmp_path / "archive",
            run_name="bagged_lasso_live",
            environment="production",
        ),
    )


def test_live_archive_manager_writes_model_signal_and_execution_rows(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    manager = _archive_manager(tmp_path, collector)
    runtime = KalshiLiveArchiveRuntime(
        label="bagged_lasso",
        family="bagged_lasso",
        model_file=Path("artifacts/kalshi/kxbtc15m_bagged_lasso/latest/bagged_lasso/model.joblib"),
        scorer=SimpleNamespace(),
        signal_engine=SimpleNamespace(config=SimpleNamespace(), subscribe_queue=lambda: asyncio.Queue()),
        execution_engine=SimpleNamespace(config=SimpleNamespace(mode=KalshiExecutionMode.SHADOW, subaccount=7), subscribe_queue=lambda: asyncio.Queue()),
        calibration_enabled=True,
    )
    event_time = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
    score_update = KalshiLightGBMScoreUpdate(
        ticker="KXBTC15M-TEST",
        event_time=event_time,
        market_prob=0.55,
        tau_minutes=7.0,
        predicted_yes_probability=0.76,
        model_edge=0.21,
        model_file=runtime.model_file,
        last_yes_price_cents=55,
        last_price_cents=55,
        yes_bid_cents=54,
        yes_ask_cents=56,
        no_bid_cents=44,
        no_ask_cents=46,
        ticker_update_time=event_time,
        trade_yes_prob=0.55,
        quote_mid_prob=0.55,
        quote_spread_cents=2,
        buy_yes_price_cents=56,
        buy_no_price_cents=46,
        quote_age_seconds=0.2,
        last_to_mid_gap=0.0,
        price_momentum=0.03,
        signed_contracts_sum_300s=6.0,
        yes_taker_share_300s=0.75,
        received_at=event_time + timedelta(seconds=1),
        event_id="ticker:event-2",
        raw_event_id="session1:2",
        source="ticker",
    )
    trade_intent = KalshiTradeIntent(
        decision_id="decision-1",
        ticker="KXBTC15M-TEST",
        side="YES",
        contracts=1,
        reference_price_cents=56,
        max_acceptable_entry_price_cents=70,
        predicted_yes_probability=0.76,
        predicted_no_probability=0.24,
        feature_basis_market_prob=0.55,
        raw_model_edge=0.21,
        post_cost_edge=0.18,
        yes_post_cost_edge=0.18,
        no_post_cost_edge=-0.28,
        last_yes_price_cents=55,
        yes_bid_cents=54,
        yes_ask_cents=56,
        buy_yes_price_cents=56,
        buy_no_price_cents=46,
        quote_mid_prob=0.55,
        quote_spread_cents=2,
        quote_age_seconds=0.2,
        estimated_entry_cost_dollars=0.57,
        estimated_fees_dollars=0.02,
        estimated_cash_required_dollars=0.59,
        generated_at=event_time,
        thesis_id="thesis-1",
        tranche_index=0,
        tranche_window="10m",
        tranche_reason="opened",
        lifecycle_state="probe_pending",
        total_thesis_budget_dollars=2.95,
        payout_if_yes_dollars=0.43,
        payout_if_no_dollars=-0.57,
        expected_value_dollars=0.19,
        worst_case_loss_dollars=0.57,
    )
    signal_update = KalshiSignalDecisionUpdate(
        ticker="KXBTC15M-TEST",
        event_time=event_time,
        approved=True,
        side="YES",
        predicted_yes_probability=0.76,
        predicted_no_probability=0.24,
        feature_basis_market_prob=0.55,
        raw_model_edge=0.21,
        post_cost_edge=0.18,
        yes_post_cost_edge=0.18,
        no_post_cost_edge=-0.28,
        tau_minutes=7.0,
        reference_price_cents=56,
        max_acceptable_entry_price_cents=70,
        last_yes_price_cents=55,
        yes_bid_cents=54,
        yes_ask_cents=56,
        buy_yes_price_cents=56,
        buy_no_price_cents=46,
        quote_mid_prob=0.55,
        quote_spread_cents=2,
        quote_age_seconds=0.2,
        regime_label="uptrend",
        bearish_vote_count=0,
        bullish_vote_count=3,
        regime_price_momentum_bearish=False,
        regime_signed_flow_bearish=False,
        regime_yes_share_bearish=False,
        regime_price_momentum_bullish=True,
        regime_signed_flow_bullish=True,
        regime_yes_share_bullish=True,
        tau_bucket="6-8",
        price_bucket="50-60",
        chosen_side_probability_bucket="70-80",
        chosen_side_edge_bucket="10-20",
        bucket_policy_dimension=None,
        bucket_policy_bucket=None,
        bucket_policy_side=None,
        block_reason=None,
        trade_intent=trade_intent,
    )
    execution_update = KalshiExecutionUpdate(
        decision_id="decision-1",
        ticker="KXBTC15M-TEST",
        side="YES",
        contracts=1,
        mode=KalshiExecutionMode.LIVE,
        status="filled",
        event_time=event_time + timedelta(seconds=2),
        reference_price_cents=56,
        limit_price_cents=70,
        client_order_id="decision-1",
        order_id="order-1",
        filled_contracts=1,
        remaining_contracts=0,
        fill_price_cents=56,
        entry_cost_dollars=0.56,
        fees_dollars=0.02,
        cash_required_dollars=0.58,
        available_cash_dollars=99.42,
        realized_pnl_dollars=None,
        cumulative_realized_pnl_dollars=None,
        settlement_result=None,
        message="order_terminal_fill",
        live_order=None,
        thesis_id="thesis-1",
        tranche_index=0,
        tranche_window="10m",
        tranche_reason="opened",
        lifecycle_state="probe_open",
        total_thesis_budget_dollars=2.95,
        payout_if_yes_dollars=0.43,
        payout_if_no_dollars=-0.57,
        expected_value_dollars=0.19,
        worst_case_loss_dollars=0.57,
    )
    layering_update = KalshiLayeringDecision(
        event_time=event_time + timedelta(seconds=1),
        ticker="KXBTC15M-TEST",
        thesis_id="thesis-1",
        action="opened",
        status="emitted",
        decision_window="10m",
        side="YES",
        tranche_index=0,
        contracts=1,
        entry_price_cents=56,
        payout_if_yes_dollars=0.43,
        payout_if_no_dollars=-0.57,
        expected_value_dollars=0.19,
        worst_case_loss_dollars=0.57,
        total_thesis_budget_dollars=2.95,
        lifecycle_state="probe_pending",
        message=None,
    )

    async def run() -> None:
        await manager._write_model_output(runtime, score_update)
        await manager._write_signal_decision_event(runtime, signal_update)
        await manager._write_layering_event(runtime, layering_update)
        await manager._write_execution_event(runtime, execution_update)
        await manager.compact_all_staging()

    asyncio.run(run())

    model_files = list((tmp_path / "archive" / "model_outputs").rglob("*.parquet"))
    strategy_files = list((tmp_path / "archive" / "strategy_events").rglob("*.parquet"))
    assert model_files
    assert strategy_files

    model_df = pd.read_parquet(model_files[0])
    strategy_df = pd.read_parquet(strategy_files[0])
    assert model_df.iloc[0]["model_label"] == "bagged_lasso"
    assert abs(float(model_df.iloc[0]["predicted_yes_probability"]) - 0.76) < 1e-9
    assert {"signal_approved", "layering_opened", "execution_filled"} <= set(strategy_df["event_kind"])
    execution_rows = strategy_df.loc[strategy_df["event_kind"] == "execution_filled"]
    assert int(execution_rows.iloc[0]["subaccount"]) == 7
    assert execution_rows.iloc[0]["thesis_id"] == "thesis-1"
    layering_rows = strategy_df.loc[strategy_df["event_kind"] == "layering_opened"]
    assert layering_rows.iloc[0]["tranche_window"] == "10m"


def test_repair_live_archive_compacts_leftover_stage(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = _market(ticker)
    manager = _archive_manager(tmp_path, collector)
    event_time = datetime(2026, 1, 1, 12, 3, tzinfo=UTC)
    state = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=event_time,
        previous_yes_price_cents=54,
        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
        is_open=True,
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        last_price_cents=55,
        yes_bid_cents=54,
        yes_ask_cents=56,
        no_bid_cents=44,
        no_ask_cents=46,
        volume=123,
        open_interest=77,
        ticker_update_time=event_time,
        received_at=event_time + timedelta(seconds=1),
        event_id="ticker:event-3",
        raw_event_id="session1:3",
    )
    ticker_update = state_to_update(state, event_time, now=event_time, source="ticker")
    feature_update = feature_update_from_state(feature_state_from_ticker_update(ticker_update))

    async def run() -> None:
        await manager._write_market_event(ticker_update)
        await manager._write_feature_row(feature_update)
        await repair_live_archive(tmp_path / "archive", run_name="bagged_lasso_live", environment="production")

    asyncio.run(run())

    market_files = list((tmp_path / "archive" / "market_events").rglob("*.parquet"))
    feature_files = list((tmp_path / "archive" / "feature_rows").rglob("*.parquet"))
    assert market_files
    assert feature_files


def test_live_archive_feature_rows_write_v2_schema_when_spot_diagnostics_present(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    manager = _archive_manager(tmp_path, collector)
    event_time = datetime(2026, 1, 1, 12, 3, tzinfo=UTC)
    ticker_update = state_to_update(
        KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=55,
            last_trade_time=event_time,
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
            is_open=True,
            open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            last_price_cents=55,
            yes_bid_cents=54,
            yes_ask_cents=56,
            no_bid_cents=44,
            no_ask_cents=46,
            ticker_update_time=event_time,
            received_at=event_time,
            event_id="ticker:event-spot",
            raw_event_id="session1:spot",
        ),
        event_time,
        now=event_time,
        source="ticker",
    )
    feature_update = feature_update_from_state(feature_state_from_ticker_update(ticker_update))
    feature_update = feature_update.__class__(
        **{
            **feature_update.__dict__,
            "btc_spot_price": 85000.0,
            "btc_spot_twap_60s": 84999.5,
            "btc_spot_age_ms": 120.0,
            "btc_spot_is_fresh": True,
            "btc_spot_venues_fresh": 2.0,
            "btc_spot_venue_divergence_bps": 1.5,
            "btc_vol_effective_sample_size": 120.0,
            "btc_spot_source": "quote",
        }
    )

    async def run() -> None:
        await manager._write_feature_row(feature_update)
        await manager.compact_all_staging()

    asyncio.run(run())

    feature_file = next((tmp_path / "archive" / "feature_rows").rglob("*.parquet"))
    df = pd.read_parquet(feature_file)

    assert df.iloc[0]["schema_version"] == "kalshi_feature_row_v2"
    assert float(df.iloc[0]["btc_spot_price"]) == pytest.approx(85000.0)


def test_live_archive_feature_rows_write_v2_schema_without_spot_fields(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    manager = _archive_manager(tmp_path, collector)
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    ticker_update = KalshiTickerUpdate(
        ticker="KXBTC15M-TEST",
        event_time=event_time,
        last_yes_price_cents=55,
        previous_yes_price_cents=54,
        close_time=event_time + timedelta(minutes=5),
        is_open=True,
        market_prob=None,
        previous_market_prob=None,
        price_momentum=None,
        tau_minutes=None,
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        last_price_cents=55,
        yes_bid_cents=54,
        yes_ask_cents=56,
        no_bid_cents=44,
        no_ask_cents=46,
        ticker_update_time=event_time,
        received_at=event_time,
        event_id="ticker:event-nospot",
        raw_event_id="session1:nospot",
    )
    feature_update = feature_update_from_state(feature_state_from_ticker_update(ticker_update))

    async def run() -> None:
        await manager._write_feature_row(feature_update)
        await manager.compact_all_staging()

    asyncio.run(run())

    feature_file = next((tmp_path / "archive" / "feature_rows").rglob("*.parquet"))
    df = pd.read_parquet(feature_file)

    assert df.iloc[0]["schema_version"] == "kalshi_feature_row_v2"
