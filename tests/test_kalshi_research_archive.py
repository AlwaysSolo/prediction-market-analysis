from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.indexers.kalshi.models import Market
from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureStateEngine,
    KalshiLightGBMScoreUpdate,
    KalshiMarketDataCollector,
    KalshiResearchArchiveConfig,
    KalshiResearchArchiveManager,
    KalshiResearchArchiveRuntime,
    KalshiResearchSample,
    KalshiResearchSampleUpdate,
    KalshiResearchSettlementUpdate,
    KalshiResearchSummaryUpdate,
    KalshiRawStreamEvent,
    KalshiTickerState,
)
from src.live.kalshi.features import feature_state_from_ticker_update, feature_update_from_state
from src.live.kalshi.research_archive import repair_research_archive
from src.live.kalshi.types import state_to_update


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
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


def _archive_manager(tmp_path: Path, collector: KalshiMarketDataCollector) -> KalshiResearchArchiveManager:
    feature_engine = KalshiFeatureStateEngine(collector)
    return KalshiResearchArchiveManager(
        collector,
        feature_engine,
        [],
        KalshiResearchArchiveConfig(
            archive_root=tmp_path / "archive",
            run_name="bucket_discovery",
            environment="demo",
        ),
    )


def test_collector_raw_stream_and_no_quote_fields_propagate(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = _market(ticker)
    collector._stream_session_id = "session1"
    raw_queue = collector.subscribe_raw_stream_queue()
    update_queue = collector.subscribe_queue()

    async def run() -> None:
        await collector._handle_message(
            {
                "type": "ticker",
                "msg": {
                    "market_ticker": ticker,
                    "price_dollars": "0.550",
                    "yes_bid_dollars": "0.540",
                    "yes_ask_dollars": "0.560",
                    "no_bid_dollars": "0.440",
                    "no_ask_dollars": "0.460",
                    "volume_fp": "123.00",
                    "open_interest_fp": "77.00",
                    "time": "2026-01-01T12:01:00Z",
                },
            }
        )
        raw_event = await asyncio.wait_for(raw_queue.get(), timeout=0.5)
        update = await asyncio.wait_for(update_queue.get(), timeout=0.5)

        assert raw_event.channel == "ticker"
        assert raw_event.raw_event_id == "session1:1"
        assert update.no_bid_cents == 44
        assert update.no_ask_cents == 46
        assert update.raw_event_id == raw_event.raw_event_id
        assert update.received_at is not None

    asyncio.run(run())


def test_archive_manager_writes_raw_market_and_feature_layers(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = _market(ticker)
    manager = _archive_manager(tmp_path, collector)
    event_time = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
    raw_event = KalshiRawStreamEvent(
        channel="ticker",
        market_ticker=ticker,
        exchange_event_time=event_time,
        received_at=event_time + timedelta(seconds=1),
        session_id="session1",
        message_index=1,
        payload={"type": "ticker", "msg": {"market_ticker": ticker}},
        raw_event_id="session1:1",
    )
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
        dollar_volume=456,
        dollar_open_interest=210,
        ticker_update_time=event_time,
        received_at=event_time + timedelta(seconds=1),
        event_id="ticker:event-1",
        raw_event_id="session1:1",
    )
    ticker_update = state_to_update(state, event_time, now=event_time, source="ticker")
    feature_update = feature_update_from_state(feature_state_from_ticker_update(ticker_update))

    async def run() -> None:
        await manager._write_raw_stream_event(raw_event)
        await manager._write_market_event(ticker_update)
        await manager._write_feature_row(feature_update)
        await manager.compact_all_staging()

    asyncio.run(run())

    raw_files = list((tmp_path / "archive" / "raw_ws").rglob("*.jsonl.zst"))
    market_files = list((tmp_path / "archive" / "market_events").rglob("*.parquet"))
    feature_files = list((tmp_path / "archive" / "feature_rows").rglob("*.parquet"))
    assert raw_files
    assert market_files
    assert feature_files

    market_df = pd.read_parquet(market_files[0])
    feature_df = pd.read_parquet(feature_files[0])
    assert int(market_df.iloc[0]["no_bid_cents"]) == 44
    assert int(market_df.iloc[0]["no_ask_cents"]) == 46
    assert feature_df.iloc[0]["feature_row_id"] == "feature:ticker:event-1"
    assert int(feature_df.iloc[0]["no_bid_cents"]) == 44


def test_archive_manager_writes_model_outputs_and_strategy_events(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    manager = _archive_manager(tmp_path, collector)
    runtime = KalshiResearchArchiveRuntime(
        label="lasso",
        family="lasso",
        model_file=Path("artifacts/kalshi/kxbtc15m_lasso/latest/lasso/model.joblib"),
        scorer=SimpleNamespace(),
        sampler=SimpleNamespace(),
        ledger=SimpleNamespace(),
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
    sample = KalshiResearchSample(
        sample_id="sample-1",
        ticker="KXBTC15M-TEST",
        side="YES",
        recorded_at=event_time,
        contracts=1,
        reference_price_cents=56,
        max_acceptable_entry_price_cents=70,
        predicted_yes_probability=0.76,
        predicted_no_probability=0.24,
        chosen_side_probability=0.76,
        feature_basis_market_prob=0.55,
        raw_model_edge=0.21,
        yes_post_cost_edge=0.18,
        no_post_cost_edge=-0.28,
        chosen_post_cost_edge=0.18,
        last_yes_price_cents=55,
        last_price_cents=55,
        yes_bid_cents=54,
        yes_ask_cents=56,
        no_bid_cents=44,
        no_ask_cents=46,
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
        tau_minutes=7.0,
        tau_bucket="6-8",
        price_bucket="50-60",
        chosen_side_probability_bucket="70-80",
        chosen_side_edge_bucket="10-20",
        estimated_entry_cost_dollars=0.57,
        estimated_fees_dollars=0.02,
        estimated_cash_required_dollars=0.59,
        market_event_id="ticker:event-2",
        feature_row_id="feature:ticker:event-2",
        raw_event_id="session1:2",
        received_at=event_time + timedelta(seconds=1),
        source="ticker",
    )
    sample_update = KalshiResearchSampleUpdate(
        ticker=sample.ticker,
        event_time=event_time,
        status="recorded",
        reason=None,
        side=sample.side,
        predicted_yes_probability=sample.predicted_yes_probability,
        predicted_no_probability=sample.predicted_no_probability,
        feature_basis_market_prob=sample.feature_basis_market_prob,
        raw_model_edge=sample.raw_model_edge,
        yes_post_cost_edge=sample.yes_post_cost_edge,
        no_post_cost_edge=sample.no_post_cost_edge,
        chosen_post_cost_edge=sample.chosen_post_cost_edge,
        tau_minutes=sample.tau_minutes,
        reference_price_cents=sample.reference_price_cents,
        yes_bid_cents=sample.yes_bid_cents,
        yes_ask_cents=sample.yes_ask_cents,
        no_bid_cents=sample.no_bid_cents,
        no_ask_cents=sample.no_ask_cents,
        buy_yes_price_cents=sample.buy_yes_price_cents,
        buy_no_price_cents=sample.buy_no_price_cents,
        quote_mid_prob=sample.quote_mid_prob,
        quote_spread_cents=sample.quote_spread_cents,
        quote_age_seconds=sample.quote_age_seconds,
        regime_label=sample.regime_label,
        bearish_vote_count=sample.bearish_vote_count,
        bullish_vote_count=sample.bullish_vote_count,
        regime_price_momentum_bearish=sample.regime_price_momentum_bearish,
        regime_signed_flow_bearish=sample.regime_signed_flow_bearish,
        regime_yes_share_bearish=sample.regime_yes_share_bearish,
        regime_price_momentum_bullish=sample.regime_price_momentum_bullish,
        regime_signed_flow_bullish=sample.regime_signed_flow_bullish,
        regime_yes_share_bullish=sample.regime_yes_share_bullish,
        tau_bucket=sample.tau_bucket,
        price_bucket=sample.price_bucket,
        chosen_side_probability_bucket=sample.chosen_side_probability_bucket,
        chosen_side_edge_bucket=sample.chosen_side_edge_bucket,
        bucket_policy_dimension=None,
        bucket_policy_bucket=None,
        bucket_policy_side=None,
        sample=sample,
        market_event_id=sample.market_event_id,
        feature_row_id=sample.feature_row_id,
        raw_event_id=sample.raw_event_id,
        received_at=sample.received_at,
        source=sample.source,
    )
    settlement_update = KalshiResearchSettlementUpdate(
        sample_id=sample.sample_id,
        ticker=sample.ticker,
        event_time=event_time + timedelta(minutes=10),
        side=sample.side,
        settlement_result="YES",
        is_win=True,
        realized_pnl_dollars=0.41,
        cumulative_realized_pnl_dollars=0.41,
        sample=sample,
    )
    summary_update = KalshiResearchSummaryUpdate(
        event_time=event_time + timedelta(minutes=10),
        open_sample_count=0,
        settled_sample_count=1,
        win_count=1,
        loss_count=0,
        cumulative_realized_pnl_dollars=0.41,
    )

    async def run() -> None:
        await manager._write_model_output(runtime, score_update)
        await manager._write_strategy_sample_event(runtime, sample_update)
        await manager._write_strategy_settlement_event(runtime, settlement_update)
        await manager._write_strategy_summary_event(runtime, summary_update)
        await manager.compact_all_staging()

    asyncio.run(run())

    model_files = list((tmp_path / "archive" / "model_outputs").rglob("*.parquet"))
    strategy_files = list((tmp_path / "archive" / "strategy_events").rglob("*.parquet"))
    assert model_files
    assert strategy_files

    model_df = pd.read_parquet(model_files[0])
    strategy_df = pd.read_parquet(strategy_files[0])
    assert model_df.iloc[0]["model_label"] == "lasso"
    assert abs(float(model_df.iloc[0]["predicted_yes_probability"]) - 0.76) < 1e-9
    assert {"recorded", "settled", "summary_snapshot"} <= set(strategy_df["event_kind"])


def test_repair_research_archive_compacts_leftover_stage(tmp_path: Path) -> None:
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

    async def run() -> None:
        await manager._write_market_event(ticker_update)
        await repair_research_archive(tmp_path / "archive", run_name="bucket_discovery", environment="demo")

    asyncio.run(run())

    market_files = list((tmp_path / "archive" / "market_events").rglob("*.parquet"))
    assert market_files
