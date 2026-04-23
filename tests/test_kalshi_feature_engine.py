from __future__ import annotations

import asyncio
import importlib.util
from datetime import UTC, datetime, timedelta
import math
from pathlib import Path

import numpy as np
import pytest

from src.live.data.btc_spot_feed import BTCSpotUpdate
from src.indexers.kalshi.models import Market
from src.live.kalshi import (
    FEATURE_ORDER,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureEngineConfig,
    KalshiFeatureStateEngine,
    KalshiMarketDataCollector,
)
from src.live.kalshi.features import (
    SPOT_V1_FEATURE_SCHEMA,
    KalshiTradeFeatureAccumulator,
    build_feature_row,
    feature_state_from_ticker_state,
    feature_state_from_ticker_update,
)
from src.live.kalshi.types import KalshiTickerState, KalshiTickerUpdate


class _FakeSpotFeed:
    def __init__(self, lookup=None):
        self._lookup = lookup

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue:
        return asyncio.Queue(maxsize=maxsize)

    def lookup_at_or_before(self, event_time: datetime, *, max_age_ms: int | None = None):
        if callable(self._lookup):
            return self._lookup(event_time, max_age_ms=max_age_ms)
        return self._lookup

    def snapshot_state(self):
        return self._lookup


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


def _load_score_helper():
    helper_path = Path("Gemini scripts") / "load_lightgbm_and_score.py"
    spec = importlib.util.spec_from_file_location("load_lightgbm_and_score", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ticker_update(
    *,
    ticker: str = "KXBTC15M-TEST",
    event_time: datetime | None = None,
    last_yes_price_cents: int = 55,
    previous_yes_price_cents: int | None = 54,
    yes_bid_cents: int | None = None,
    yes_ask_cents: int | None = None,
    ticker_update_time: datetime | None = None,
    close_time: datetime | None = None,
    is_open: bool = True,
) -> KalshiTickerUpdate:
    event_time = event_time or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    if close_time is None:
        close_time = event_time + timedelta(minutes=7)
    if yes_bid_cents is None:
        yes_bid_cents = min(98, max(1, last_yes_price_cents - 1))
    if yes_ask_cents is None:
        yes_ask_cents = max(yes_bid_cents + 1, min(99, last_yes_price_cents + 1))
    if ticker_update_time is None:
        ticker_update_time = event_time
    return KalshiTickerUpdate(
        ticker=ticker,
        event_time=event_time,
        last_yes_price_cents=last_yes_price_cents,
        previous_yes_price_cents=previous_yes_price_cents,
        close_time=close_time,
        is_open=is_open,
        market_prob=None,
        previous_market_prob=None,
        price_momentum=None,
        tau_minutes=None,
        yes_bid_cents=yes_bid_cents,
        yes_ask_cents=yes_ask_cents,
        ticker_update_time=ticker_update_time,
    )


def test_feature_row_matches_standalone_scorer():
    helper = _load_score_helper()
    row = build_feature_row(market_prob=0.55, tau_minutes=7.0, previous_market_prob=0.54)
    helper_row = helper.build_feature_row(market_prob=0.55, tau_minutes=7.0, previous_market_prob=0.54)

    assert FEATURE_ORDER == helper.FEATURE_ORDER
    assert np.allclose(row, helper_row)


def test_feature_state_clips_probability_and_uses_zero_momentum_without_previous():
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    state = feature_state_from_ticker_update(
        _ticker_update(
            event_time=event_time,
            last_yes_price_cents=100,
            previous_yes_price_cents=None,
            close_time=event_time + timedelta(minutes=5),
        )
    )

    assert state.market_prob == 0.99
    assert state.previous_market_prob == 0.99
    assert state.price_momentum == 0.0
    assert state.abs_price_momentum == 0.0
    assert state.price_direction == 0.0
    assert state.distance_from_mid == 0.49
    assert state.in_training_window is True
    assert state.is_scoreable is True


def test_feature_state_scoreability_rules():
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    closed_state = feature_state_from_ticker_update(_ticker_update(event_time=event_time, is_open=False))
    missing_close_update = KalshiTickerUpdate(
        ticker="KXBTC15M-TEST",
        event_time=event_time,
        last_yes_price_cents=55,
        previous_yes_price_cents=54,
        close_time=None,
        is_open=True,
        market_prob=None,
        previous_market_prob=None,
        price_momentum=None,
        tau_minutes=None,
    )
    missing_close_state = feature_state_from_ticker_update(missing_close_update)
    outside_window_state = feature_state_from_ticker_update(
        _ticker_update(event_time=event_time, close_time=event_time + timedelta(minutes=16))
    )

    assert closed_state.is_scoreable is False
    assert missing_close_state.tau_minutes is None
    assert missing_close_state.is_scoreable is False
    assert outside_window_state.in_training_window is False
    assert outside_window_state.is_scoreable is False


def test_feature_state_exposes_quote_diagnostics_without_changing_feature_row():
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    state = feature_state_from_ticker_update(
        _ticker_update(
            event_time=event_time,
            last_yes_price_cents=55,
            previous_yes_price_cents=54,
            yes_bid_cents=54,
            yes_ask_cents=57,
            ticker_update_time=event_time - timedelta(seconds=2),
            close_time=event_time + timedelta(minutes=5),
        )
    )

    assert state.trade_yes_prob == pytest.approx(0.55)
    assert state.quote_mid_prob == pytest.approx(0.555)
    assert state.quote_spread_cents == 3
    assert state.buy_yes_price_cents == 57
    assert state.buy_no_price_cents == 46
    assert state.quote_age_seconds == pytest.approx(2.0)
    assert state.last_to_mid_gap == pytest.approx(-0.005)
    assert state.feature_row() is not None
    assert state.feature_row().shape == (1, len(FEATURE_ORDER))


def test_feature_state_engine_requires_fresh_external_spot_for_spot_v1(tmp_path: Path):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="BTC price up in next 15 mins?",
        yes_sub_title="Price to beat: $85,000.00",
        no_sub_title="Price to beat: TBD",
        status="open",
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
        open_time=datetime(2026, 1, 1, 11, 55, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
    )
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    stale_lookup = None
    engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            feature_schema=SPOT_V1_FEATURE_SCHEMA,
            external_spot_required_for_schema=True,
        ),
        spot_feed=_FakeSpotFeed(lookup=stale_lookup),
    )

    async def run() -> None:
        queue = engine.subscribe_queue()
        await engine.start()
        assert queue.empty()
        await collector._publish_update(_ticker_update(ticker=ticker, event_time=event_time))
        await asyncio.sleep(0)
        assert queue.empty()
        state = engine.get_state(ticker)
        assert state is not None
        assert state.is_scoreable is False

    asyncio.run(run())


def test_feature_state_engine_enriches_spot_v1_from_fresh_external_spot(tmp_path: Path):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="BTC price up in next 15 mins?",
        yes_sub_title="Price to beat: $85,000.00",
        no_sub_title="Price to beat: TBD",
        status="open",
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
        open_time=datetime(2026, 1, 1, 11, 55, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
    )
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    spot_update = BTCSpotUpdate(
        event_time=event_time - timedelta(milliseconds=100),
        received_at=event_time - timedelta(milliseconds=50),
        btc_spot_price=86000.0,
        btc_spot_twap_60s=85980.0,
        btc_spot_age_ms=100.0,
        btc_spot_is_fresh=True,
        btc_spot_venues_fresh=2,
        btc_spot_venue_divergence_bps=3.0,
        btc_vol_effective_sample_size=120.0,
        btc_spot_source="quote",
        btc_spot_return_30s=0.001,
        btc_spot_return_120s=0.002,
        btc_spot_return_300s=0.003,
        btc_spot_return_900s=0.004,
        btc_spot_vol_120s=0.55,
        btc_spot_vol_300s=0.5,
        btc_spot_vol_900s=0.45,
        btc_spot_vol_1800s=0.4,
        btc_spot_vol_ewma_hl300=0.48,
    )
    engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            feature_schema=SPOT_V1_FEATURE_SCHEMA,
            external_spot_required_for_schema=True,
        ),
        spot_feed=_FakeSpotFeed(lookup=spot_update),
    )

    async def run() -> None:
        queue = engine.subscribe_queue()
        await engine.start()
        await collector._publish_update(_ticker_update(ticker=ticker, event_time=event_time))
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == ticker
        assert update.is_scoreable is True
        assert update.btc_spot_is_fresh == pytest.approx(1.0)
        assert update.btc_spot_price == pytest.approx(86000.0)
        assert update.btc_log_moneyness == pytest.approx(math.log(86000.0 / 85000.0))
        assert update.btc_log_moneyness_twap60 == pytest.approx(math.log(85980.0 / 85000.0))
        await engine.stop()

    asyncio.run(run())


def test_feature_state_from_ticker_state_uses_snapshot_time():
    snapshot_time = datetime(2026, 1, 1, 12, 5, tzinfo=UTC)
    ticker_state = KalshiTickerState(
        ticker="KXBTC15M-TEST",
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
        is_open=True,
    )

    state = feature_state_from_ticker_state(ticker_state, event_time=snapshot_time)

    assert state.event_time == snapshot_time
    assert state.tau_minutes == 5.0


def test_trade_feature_accumulator_computes_trade_windows_without_leakage():
    accumulator = KalshiTradeFeatureAccumulator()
    ticker = "KXBTC15M-TEST"
    open_time = datetime(2026, 1, 1, 11, 55, tzinfo=UTC)
    close_time = datetime(2026, 1, 1, 12, 10, tzinfo=UTC)
    event_times = (
        datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC),
        datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
    )
    updates = (
        KalshiTickerUpdate(
            ticker=ticker,
            event_time=event_times[0],
            last_yes_price_cents=55,
            previous_yes_price_cents=None,
            close_time=close_time,
            is_open=True,
            market_prob=None,
            previous_market_prob=None,
            price_momentum=None,
            tau_minutes=None,
            open_time=open_time,
            trade_id="t1",
            count=2,
            taker_side="yes",
            last_trade_time=event_times[0],
            last_price_cents=55,
            source="trade",
        ),
        KalshiTickerUpdate(
            ticker=ticker,
            event_time=event_times[1],
            last_yes_price_cents=56,
            previous_yes_price_cents=55,
            close_time=close_time,
            is_open=True,
            market_prob=None,
            previous_market_prob=None,
            price_momentum=None,
            tau_minutes=None,
            open_time=open_time,
            trade_id="t2",
            count=1,
            taker_side="no",
            last_trade_time=event_times[1],
            last_price_cents=56,
            source="trade",
        ),
        KalshiTickerUpdate(
            ticker=ticker,
            event_time=event_times[2],
            last_yes_price_cents=58,
            previous_yes_price_cents=56,
            close_time=close_time,
            is_open=True,
            market_prob=None,
            previous_market_prob=None,
            price_momentum=None,
            tau_minutes=None,
            open_time=open_time,
            trade_id="t3",
            count=3,
            taker_side="yes",
            last_trade_time=event_times[2],
            last_price_cents=58,
            source="trade",
        ),
    )

    first_state = accumulator.build_feature_state_from_update(updates[0])
    second_state = accumulator.build_feature_state_from_update(updates[1])
    third_state = accumulator.build_feature_state_from_update(updates[2])

    assert first_state.time_since_last_trade_seconds == 0.0
    assert first_state.trade_count_30s == 1.0
    assert first_state.contracts_sum_30s == 2.0

    assert second_state.time_since_last_trade_seconds == 10.0
    assert second_state.last_trade_side_sign == -1.0
    assert second_state.last_trade_signed_count == -1.0
    assert second_state.trade_count_30s == 2.0
    assert second_state.contracts_sum_30s == 3.0
    assert second_state.signed_contracts_sum_30s == 1.0
    assert second_state.yes_taker_share_30s == pytest.approx(2.0 / 3.0)
    assert second_state.price_return_30s == pytest.approx(0.01)
    assert second_state.price_volatility_30s == 0.0

    assert third_state.time_since_last_trade_seconds == 50.0
    assert third_state.minutes_since_market_open == 6.0
    assert third_state.last_trade_count == 3.0
    assert third_state.last_trade_side_sign == 1.0
    assert third_state.last_trade_signed_count == 3.0
    assert third_state.trade_count_30s == 1.0
    assert third_state.contracts_sum_30s == 3.0
    assert third_state.signed_contracts_sum_30s == 3.0
    assert third_state.yes_taker_share_30s == 1.0
    assert third_state.price_return_30s == 0.0
    assert third_state.price_volatility_30s == 0.0
    assert third_state.trade_count_120s == 3.0
    assert third_state.contracts_sum_120s == 6.0
    assert third_state.signed_contracts_sum_120s == 4.0
    assert third_state.yes_taker_share_120s == pytest.approx(5.0 / 6.0)
    assert third_state.price_return_120s == pytest.approx(0.03)
    assert third_state.price_volatility_120s == pytest.approx(np.std(np.array([0.01, 0.02], dtype=np.float64)))


def test_feature_engine_emits_scoreable_updates_from_collector(tmp_path: Path):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    queue = engine.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await collector._publish_update(_ticker_update())
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == "KXBTC15M-TEST"
        assert update.is_scoreable is True
        assert update.feature_values() is not None
        await engine.stop()

    asyncio.run(run())


def test_feature_engine_metadata_update_refreshes_tau_and_close_time(tmp_path: Path):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    queue = engine.subscribe_queue()

    first_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    second_time = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)

    async def run() -> None:
        await engine.start()
        await collector._publish_update(
            _ticker_update(event_time=first_time, close_time=first_time + timedelta(minutes=7))
        )
        first_update = await asyncio.wait_for(queue.get(), timeout=0.5)

        await collector._publish_update(
            _ticker_update(event_time=second_time, close_time=second_time + timedelta(minutes=9))
        )
        second_update = await asyncio.wait_for(queue.get(), timeout=0.5)

        assert first_update.close_time != second_update.close_time
        assert round(second_update.tau_minutes or 0.0, 2) == 9.00
        await engine.stop()

    asyncio.run(run())


def test_feature_engine_suppresses_duplicate_scoreable_updates(tmp_path: Path):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    queue = engine.subscribe_queue()
    update = _ticker_update()

    async def run() -> None:
        await engine.start()
        await collector._publish_update(update)
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.ticker == update.ticker

        await collector._publish_update(update)
        try:
            await asyncio.wait_for(queue.get(), timeout=0.1)
            raise AssertionError("Expected duplicate scoreable update to be suppressed.")
        except asyncio.TimeoutError:
            pass
        await engine.stop()

    asyncio.run(run())


def test_feature_engine_bootstraps_existing_collector_state(tmp_path: Path):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    future_close_time = datetime.now(UTC) + timedelta(minutes=10)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 11, 58, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=future_close_time,
        is_open=True,
    )
    engine = KalshiFeatureStateEngine(collector)
    queue = engine.subscribe_queue()

    async def run() -> None:
        await engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == ticker
        assert update.is_scoreable is True
        await engine.stop()

    asyncio.run(run())


def test_feature_engine_attaches_hourly_context_and_republishes_on_better_atm_candidate(tmp_path: Path):
    collector = KalshiMarketDataCollector(
        KalshiCollectorConfig(
            environment=KalshiEnvironment.DEMO,
            credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
            log_dir=tmp_path / "logs",
            series_tickers=("KXBTC15M", "KXBTCD"),
            metadata_refresh_interval_seconds=3600.0,
        )
    )
    engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            publish_series_tickers=("KXBTC15M",),
            hourly_context_target_series_ticker="KXBTC15M",
            hourly_context_series_ticker="KXBTCD",
        ),
    )
    queue = engine.subscribe_queue()

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    target_time = base_time + timedelta(seconds=20)
    nearest_expiry = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)
    later_expiry = datetime(2026, 1, 1, 14, 0, tzinfo=UTC)

    async def run() -> None:
        await engine.start()

        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-NEAR-T95000.00",
                event_time=base_time,
                last_yes_price_cents=48,
                previous_yes_price_cents=47,
                close_time=nearest_expiry,
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=base_time,
                trade_id="ctx-near-1",
                count=3,
                taker_side="yes",
                last_trade_time=base_time,
                last_price_cents=48,
                source="trade",
            )
        )
        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-ATM-T95050.00",
                event_time=base_time + timedelta(seconds=1),
                last_yes_price_cents=51,
                previous_yes_price_cents=50,
                close_time=nearest_expiry,
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=base_time,
                trade_id="ctx-atm-1",
                count=4,
                taker_side="yes",
                last_trade_time=base_time + timedelta(seconds=1),
                last_price_cents=51,
                source="trade",
            )
        )
        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-LATER-T95025.00",
                event_time=base_time + timedelta(seconds=2),
                last_yes_price_cents=50,
                previous_yes_price_cents=49,
                close_time=later_expiry,
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=base_time,
                trade_id="ctx-later-1",
                count=10,
                taker_side="yes",
                last_trade_time=base_time + timedelta(seconds=2),
                last_price_cents=50,
                source="trade",
            )
        )

        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTC15M-TEST",
                event_time=target_time,
                last_yes_price_cents=55,
                previous_yes_price_cents=54,
                close_time=target_time + timedelta(minutes=7),
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=base_time,
                trade_id="target-1",
                count=2,
                taker_side="yes",
                last_trade_time=target_time,
                last_price_cents=55,
                source="trade",
            )
        )

        first_update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first_update.ticker == "KXBTC15M-TEST"
        assert first_update.kxbtcd_atm_ticker == "KXBTCD-CTX-ATM-T95050.00"
        assert first_update.kxbtcd_atm_z_implied == pytest.approx(feature_state_from_ticker_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-ATM-T95050.00",
                event_time=base_time + timedelta(seconds=1),
                last_yes_price_cents=51,
                previous_yes_price_cents=50,
                close_time=nearest_expiry,
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=base_time,
                trade_id="ctx-atm-1",
                count=4,
                taker_side="yes",
                last_trade_time=base_time + timedelta(seconds=1),
                last_price_cents=51,
                source="trade",
            )
        ).z_implied)

        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-NEAR-T95000.00",
                event_time=target_time + timedelta(seconds=5),
                last_yes_price_cents=50,
                previous_yes_price_cents=48,
                close_time=nearest_expiry,
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=base_time,
                trade_id="ctx-near-2",
                count=6,
                taker_side="yes",
                last_trade_time=target_time + timedelta(seconds=5),
                last_price_cents=50,
                source="trade",
            )
        )

        second_update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second_update.ticker == "KXBTC15M-TEST"
        assert second_update.kxbtcd_atm_ticker == "KXBTCD-CTX-NEAR-T95000.00"
        assert second_update.kxbtcd_atm_ticker != first_update.kxbtcd_atm_ticker

        await engine.stop()

    asyncio.run(run())
