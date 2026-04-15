from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiExecutionFeedback,
    KalshiFeatureStateEngine,
    KalshiLightGBMScorer,
    KalshiMarketDataCollector,
    KalshiPortfolioPosition,
    KalshiPortfolioSnapshot,
    KalshiSignalRiskConfig,
    KalshiSignalRiskEngine,
)
from src.live.kalshi.scorer import KalshiLightGBMScoreState, LoadedLightGBMModel
from src.live.kalshi.signal_risk import calculate_kelly_sizing_metrics, find_max_acceptable_entry_price_cents
from src.live.kalshi.types import KalshiTickerState, KalshiTickerUpdate


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


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


class _PredictingBooster:
    def __init__(self, prediction: float):
        self.prediction = prediction

    def predict(self, _feature_row: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        return np.array([self.prediction], dtype=np.float64)


def _fake_model(prediction: float) -> LoadedLightGBMModel:
    return LoadedLightGBMModel(
        booster=_PredictingBooster(prediction),  # type: ignore[arg-type]
        model_file=Path("fake-model.txt"),
        best_iteration=None,
    )


class _FakeScorer:
    def __init__(self, collector: KalshiMarketDataCollector, snapshot_states: dict[str, KalshiLightGBMScoreState]):
        self.feature_engine = SimpleNamespace(collector=collector)
        self._states = dict(snapshot_states)
        self._queues: list[asyncio.Queue[KalshiLightGBMScoreState]] = []

    def snapshot_states(self) -> dict[str, KalshiLightGBMScoreState]:
        return dict(self._states)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLightGBMScoreState]:
        queue: asyncio.Queue[KalshiLightGBMScoreState] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue


def test_signal_risk_config_from_env_uses_demo_override_and_shared_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KALSHI_SIGNAL_STARTING_CASH_DOLLARS", "10")
    monkeypatch.setenv("KALSHI_SIGNAL_ALLOW_STACKING", "true")
    monkeypatch.setenv("KALSHI_SIGNAL_TRADE_COOLDOWN_SECONDS", "1.0")
    monkeypatch.setenv("KALSHI_SIGNAL_INVERT_MODEL_SIGNAL", "false")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_STARTING_CASH_DOLLARS", "25")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_ALLOW_STACKING", "false")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_TRADE_COOLDOWN_SECONDS", "2.5")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_INVERT_MODEL_SIGNAL", "true")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_KELLY_FRACTION_MULTIPLIER", "0.25")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_KELLY_FRACTION_CAP_PCT", "2.0")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_QUOTE_MAX_AGE_SECONDS", "4.5")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_QUOTE_CONSISTENCY_TOLERANCE_CENTS", "1")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_BANNED_YES_TAU_BUCKETS", "2-4,4-6")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_ENABLE_BUCKET_BAN_POLICY", "true")
    monkeypatch.setenv("KALSHI_SIGNAL_BANNED_NO_PRICE_BUCKETS", "20-30,30-40")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_ENABLE_LOW_LIQUIDITY_CHOP_GATE", "true")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_LOW_LIQUIDITY_MIN_TRADE_COUNT_300S", "4")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_STRUCTURAL_REGIME_REFRESH_SECONDS", "45")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_STRUCTURAL_REGIME_FLIP_CONFIRMATIONS", "3")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_MAX_TICKER_SIDE_EXPOSURE_DOLLARS", "1.25")

    demo_config = KalshiSignalRiskConfig.from_env(KalshiEnvironment.DEMO)
    prod_config = KalshiSignalRiskConfig.from_env(KalshiEnvironment.PRODUCTION)

    assert demo_config.starting_cash_dollars == 25.0
    assert demo_config.allow_stacking is False
    assert demo_config.trade_cooldown_seconds == 2.5
    assert demo_config.invert_model_signal is True
    assert demo_config.kelly_fraction_multiplier == pytest.approx(0.25)
    assert demo_config.kelly_fraction_cap_pct == pytest.approx(2.0)
    assert demo_config.quote_max_age_seconds == pytest.approx(4.5)
    assert demo_config.quote_consistency_tolerance_cents == 1
    assert demo_config.enable_bucket_ban_policy is True
    assert demo_config.enable_low_liquidity_chop_gate is True
    assert demo_config.low_liquidity_min_trade_count_300s == pytest.approx(4.0)
    assert demo_config.structural_regime_refresh_seconds == pytest.approx(45.0)
    assert demo_config.structural_regime_flip_confirmations == 3
    assert demo_config.max_ticker_side_exposure_dollars == pytest.approx(1.25)
    assert demo_config.banned_yes_tau_buckets == frozenset({"2-4", "4-6"})
    assert prod_config.starting_cash_dollars == 10.0
    assert prod_config.allow_stacking is True
    assert prod_config.trade_cooldown_seconds == 1.0
    assert prod_config.invert_model_signal is False
    assert prod_config.kelly_fraction_multiplier is None
    assert prod_config.kelly_fraction_cap_pct is None
    assert prod_config.banned_no_price_buckets == frozenset({"20-30", "30-40"})


def test_signal_risk_config_defaults():
    config = KalshiSignalRiskConfig()

    assert config.edge_threshold_cents == 4.0
    assert config.min_tau_minutes == 2.0
    assert config.max_tau_minutes == 14.0
    assert config.invert_model_signal is False
    assert config.allow_stacking is False
    assert config.starting_cash_dollars == 10.0
    assert config.contracts_per_order == 1
    assert config.capital_pct_per_order is None
    assert config.kelly_fraction_multiplier is None
    assert config.kelly_fraction_cap_pct is None
    assert config.max_ticker_side_exposure_dollars is None
    assert config.price_band_min_cents == 20
    assert config.price_band_max_cents == 80
    assert config.quote_max_age_seconds == 3.0
    assert config.quote_consistency_tolerance_cents == 1
    assert config.enable_low_liquidity_chop_gate is False
    assert config.low_liquidity_min_trade_count_300s == pytest.approx(3.0)
    assert config.low_liquidity_min_contracts_sum_300s == pytest.approx(10.0)
    assert config.low_liquidity_min_abs_signed_contracts_sum_300s == pytest.approx(3.0)
    assert config.low_liquidity_price_volatility_300s_threshold == pytest.approx(0.01)
    assert config.structural_regime_refresh_seconds == pytest.approx(30.0)
    assert config.structural_regime_flip_confirmations == 2
    assert config.trade_cooldown_seconds == 0.0
    assert config.enable_bucket_ban_policy is True
    assert config.banned_yes_tau_buckets == frozenset({"2-4"})
    assert config.banned_no_price_buckets == frozenset({"20-30"})


def test_signal_risk_config_allows_full_price_band_for_broad_research_runs():
    config = KalshiSignalRiskConfig(price_band_min_cents=0, price_band_max_cents=100)

    assert config.price_band_min_cents == 0
    assert config.price_band_max_cents == 100


def test_find_max_acceptable_entry_price_stays_inside_price_band():
    config = KalshiSignalRiskConfig()
    max_price = find_max_acceptable_entry_price_cents(
        side="YES",
        predicted_yes_probability=0.82,
        config=config,
    )

    assert max_price is not None
    assert config.price_band_min_cents <= max_price <= config.price_band_max_cents


def test_calculate_kelly_sizing_metrics_returns_positive_fraction_for_positive_edge():
    metrics = calculate_kelly_sizing_metrics(
        side="YES",
        predicted_yes_probability=0.70,
        displayed_entry_price_cents=40,
        slippage=0.01,
    )

    assert metrics.raw_fraction_of_equity > 0.0
    assert metrics.capped_fraction_of_equity == metrics.scaled_fraction_of_equity
    assert metrics.per_contract_cash_required_dollars > 0.0
    assert metrics.per_contract_win_profit_dollars > 0.0


def test_calculate_kelly_sizing_metrics_respects_multiplier_and_cap():
    metrics = calculate_kelly_sizing_metrics(
        side="YES",
        predicted_yes_probability=0.70,
        displayed_entry_price_cents=40,
        slippage=0.01,
        fraction_multiplier=0.5,
        fraction_cap=0.05,
    )

    assert metrics.scaled_fraction_of_equity <= metrics.raw_fraction_of_equity
    assert metrics.capped_fraction_of_equity <= 0.05 + 1e-12


def test_signal_risk_bootstraps_from_scorer_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 11, 58, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime.now(UTC) + timedelta(minutes=10),
        is_open=True,
        yes_bid_cents=54,
        yes_ask_cents=56,
        ticker_update_time=datetime.now(UTC),
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.side == "YES"
        assert update.trade_intent is not None
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_approves_valid_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update())
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.side == "YES"
        assert update.trade_intent is not None
        assert update.trade_intent.contracts == 1
        assert update.trade_intent.reference_price_cents == 56
        assert 20 <= update.trade_intent.max_acceptable_entry_price_cents <= 80
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_uses_kelly_sizing_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            starting_cash_dollars=10000.0,
            reserve_cash_pct=30.0,
            kelly_fraction_multiplier=0.25,
            kelly_fraction_cap_pct=2.0,
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.trade_intent is not None
        assert update.trade_intent.contracts > 1
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_inverts_candidate_side_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(invert_model_signal=True),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.side is None
        assert update.trade_intent is None
        assert update.block_reason == "insufficient_post_cost_edge"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_trade_intent_queue_only_receives_approved_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.95))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    decision_queue = signal_engine.subscribe_queue()
    intent_queue = signal_engine.subscribe_trade_intent_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()

        await collector._publish_update(_ticker_update(last_yes_price_cents=85, previous_yes_price_cents=84))
        blocked = await asyncio.wait_for(decision_queue.get(), timeout=0.5)
        assert blocked.approved is False
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(intent_queue.get(), timeout=0.05)

        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        approved = await asyncio.wait_for(decision_queue.get(), timeout=0.5)
        intent = await asyncio.wait_for(intent_queue.get(), timeout=0.5)
        assert approved.approved is True
        assert approved.trade_intent is not None
        assert intent.decision_id == approved.trade_intent.decision_id

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_outside_tau_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        await collector._publish_update(
            _ticker_update(
                event_time=event_time,
                close_time=event_time + timedelta(minutes=15),
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "outside_tau_window"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_yes_in_downtrend_regime(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.55,
                tau_minutes=7.0,
                predicted_yes_probability=0.78,
                model_edge=0.23,
                model_file=Path("fake-model.txt"),
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
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
                price_momentum=-0.03,
                signed_contracts_sum_300s=-6.0,
                yes_taker_share_300s=0.30,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(apply_regime_hard_gate=True),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.side == "YES"
        assert update.block_reason == "blocked_by_regime_downtrend"
        assert update.regime_label == "downtrend"
        assert update.trade_intent is None

    asyncio.run(run())


def test_signal_risk_structural_regime_uses_hourly_context_hysteresis(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    score_state = KalshiLightGBMScoreState(
        ticker="KXBTC15M-TEST",
        event_time=event_time,
        market_prob=0.55,
        tau_minutes=7.0,
        predicted_yes_probability=0.78,
        model_edge=0.23,
        model_file=Path("fake-model.txt"),
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
        quote_age_seconds=0.0,
        last_to_mid_gap=0.0,
    )
    scorer = _FakeScorer(collector, {score_state.ticker: score_state})

    structural_feature_state = SimpleNamespace(
        price_return_300s=0.02,
        kxbtcd_atm_price_return_300s=0.03,
        kxbtcd_atm_signed_contracts_sum_300s=10.0,
        k15_k1h_atm_direction_agreement=1.0,
    )
    scorer.feature_engine.get_state = lambda _ticker: structural_feature_state

    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            structural_regime_refresh_seconds=30.0,
            structural_regime_flip_confirmations=2,
        ),
    )

    first = signal_engine._resolve_regime(score_state)
    assert first.regime_label == "uptrend"

    structural_feature_state.price_return_300s = -0.02
    structural_feature_state.kxbtcd_atm_price_return_300s = -0.03
    structural_feature_state.kxbtcd_atm_signed_contracts_sum_300s = -10.0
    second = signal_engine._resolve_regime(
        replace(score_state, event_time=event_time + timedelta(seconds=31))
    )
    assert second.regime_label == "uptrend"

    third = signal_engine._resolve_regime(
        replace(score_state, event_time=event_time + timedelta(seconds=62))
    )
    assert third.regime_label == "downtrend"


def test_signal_risk_blocks_low_liquidity_chop_when_enabled(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    score_state = KalshiLightGBMScoreState(
        ticker="KXBTC15M-TEST",
        event_time=event_time,
        market_prob=0.55,
        tau_minutes=8.0,
        predicted_yes_probability=0.78,
        model_edge=0.23,
        model_file=Path("fake-model.txt"),
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
        quote_age_seconds=0.0,
        last_to_mid_gap=0.0,
    )
    scorer = _FakeScorer(collector, {score_state.ticker: score_state})
    scorer.feature_engine.get_state = lambda _ticker: SimpleNamespace(
        trade_count_300s=1.0,
        contracts_sum_300s=2.0,
        signed_contracts_sum_300s=0.5,
        price_volatility_300s=0.03,
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            enable_low_liquidity_chop_gate=True,
            enable_bucket_ban_policy=False,
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.block_reason == "low_liquidity_chop"

    asyncio.run(run())


def test_signal_risk_blocks_bucket_ban_policy_for_yes_tau_bucket(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.55,
                tau_minutes=3.0,
                predicted_yes_probability=0.78,
                model_edge=0.23,
                model_file=Path("fake-model.txt"),
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
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(scorer)  # type: ignore[arg-type]
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.side == "YES"
        assert update.block_reason == "blocked_by_bucket_policy"
        assert update.tau_bucket == "2-4"
        assert update.bucket_policy_dimension == "tau"
        assert update.bucket_policy_bucket == "2-4"
        assert update.bucket_policy_side == "YES"

    asyncio.run(run())


def test_signal_risk_blocks_bucket_ban_policy_for_no_price_bucket(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.55,
                tau_minutes=8.0,
                predicted_yes_probability=0.20,
                model_edge=-0.35,
                model_file=Path("fake-model.txt"),
                last_yes_price_cents=55,
                last_price_cents=55,
                yes_bid_cents=72,
                yes_ask_cents=74,
                no_bid_cents=26,
                no_ask_cents=28,
                ticker_update_time=event_time,
                trade_yes_prob=0.55,
                quote_mid_prob=0.73,
                quote_spread_cents=2,
                buy_yes_price_cents=74,
                buy_no_price_cents=28,
                quote_age_seconds=0.0,
                last_to_mid_gap=-0.18,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(scorer)  # type: ignore[arg-type]
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.side == "NO"
        assert update.block_reason == "blocked_by_bucket_policy"
        assert update.price_bucket == "20-30"
        assert update.bucket_policy_dimension == "price"
        assert update.bucket_policy_bucket == "20-30"
        assert update.bucket_policy_side == "NO"

    asyncio.run(run())


def test_signal_risk_blocks_combo_ban_policy(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.45,
                tau_minutes=10.5,
                predicted_yes_probability=0.56,
                model_edge=0.11,
                model_file=Path("fake-model.txt"),
                last_yes_price_cents=45,
                last_price_cents=45,
                yes_bid_cents=44,
                yes_ask_cents=45,
                no_bid_cents=55,
                no_ask_cents=56,
                ticker_update_time=event_time,
                trade_yes_prob=0.45,
                quote_mid_prob=0.445,
                quote_spread_cents=1,
                buy_yes_price_cents=45,
                buy_no_price_cents=56,
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            enable_bucket_ban_policy=False,
            enable_combo_ban_policy=True,
            banned_combo_buckets=frozenset({"10-12|40-50|50-60|5-10"}),
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.block_reason == "blocked_by_combo_policy"
        assert update.bucket_policy_dimension == "combo"
        assert update.bucket_policy_bucket == "10-12|40-50|50-60|5-10"
        assert update.side == "YES"

    asyncio.run(run())


def test_signal_risk_blocks_neutral_regime_when_configured(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.55,
                tau_minutes=8.0,
                predicted_yes_probability=0.72,
                model_edge=0.17,
                model_file=Path("fake-model.txt"),
                last_yes_price_cents=55,
                last_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=55,
                no_bid_cents=45,
                no_ask_cents=46,
                ticker_update_time=event_time,
                trade_yes_prob=0.55,
                quote_mid_prob=0.545,
                quote_spread_cents=1,
                buy_yes_price_cents=55,
                buy_no_price_cents=46,
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
                price_momentum=0.0,
                signed_contracts_sum_300s=0.0,
                yes_taker_share_300s=0.5,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            enable_bucket_ban_policy=False,
            blocked_regime_labels=frozenset({"neutral"}),
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.regime_label == "neutral"
        assert update.block_reason == "blocked_by_regime_neutral"

    asyncio.run(run())


def test_signal_risk_blocks_outside_price_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.95))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update(last_yes_price_cents=85, previous_yes_price_cents=84))
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "outside_price_band"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_uses_yes_ask_for_yes_entries_and_logs_quote_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(
            _ticker_update(
                last_yes_price_cents=55,
                previous_yes_price_cents=54,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.side == "YES"
        assert update.reference_price_cents == 56
        assert update.trade_intent is not None
        assert update.trade_intent.reference_price_cents == 56
        assert update.yes_bid_cents == 54
        assert update.yes_ask_cents == 56
        assert update.quote_spread_cents == 2
        assert update.yes_post_cost_edge is not None
        assert update.no_post_cost_edge is not None
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_uses_no_ask_from_yes_bid_for_no_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            edge_threshold_cents=4.0,
            price_band_min_cents=20,
            price_band_max_cents=90,
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(
            _ticker_update(
                last_yes_price_cents=66,
                previous_yes_price_cents=65,
                yes_bid_cents=60,
                yes_ask_cents=66,
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.side == "NO"
        assert update.reference_price_cents == 40
        assert update.trade_intent is not None
        assert update.trade_intent.reference_price_cents == 40
        assert update.no_post_cost_edge is not None
        assert update.no_post_cost_edge > update.yes_post_cost_edge
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_missing_quote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                last_yes_price_cents=55,
                previous_yes_price_cents=54,
                close_time=event_time + timedelta(minutes=7),
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                yes_bid_cents=None,
                yes_ask_cents=None,
                ticker_update_time=None,
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "missing_quote"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_stale_quote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(quote_max_age_seconds=3.0),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        await collector._publish_update(
            _ticker_update(
                event_time=event_time,
                ticker_update_time=event_time - timedelta(seconds=5),
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "stale_quote"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_crossed_quote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.75))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(
            _ticker_update(
                yes_bid_cents=60,
                yes_ask_cents=60,
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "crossed_quote"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_inconsistent_quote(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.09,
                tau_minutes=8.0,
                predicted_yes_probability=0.20,
                model_edge=0.11,
                model_file=Path("fake-model.txt"),
                last_yes_price_cents=9,
                last_price_cents=9,
                yes_bid_cents=8,
                yes_ask_cents=10,
                no_bid_cents=63,
                no_ask_cents=64,
                ticker_update_time=event_time,
                trade_yes_prob=0.09,
                quote_mid_prob=0.09,
                quote_spread_cents=2,
                buy_yes_price_cents=10,
                buy_no_price_cents=64,
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(enable_bucket_ban_policy=False),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is False
        assert update.block_reason == "inconsistent_quote"
        assert update.side is None

    asyncio.run(run())


def test_signal_risk_allows_valid_no_when_yes_side_is_locked_but_no_quote_is_usable(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.60,
                tau_minutes=8.0,
                predicted_yes_probability=0.20,
                model_edge=0.40,
                model_file=Path("fake-model.txt"),
                last_yes_price_cents=60,
                last_price_cents=60,
                yes_bid_cents=60,
                yes_ask_cents=60,
                no_bid_cents=39,
                no_ask_cents=40,
                ticker_update_time=event_time,
                trade_yes_prob=0.60,
                quote_mid_prob=0.60,
                quote_spread_cents=0,
                buy_yes_price_cents=60,
                buy_no_price_cents=40,
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            enable_bucket_ban_policy=False,
            price_band_min_cents=0,
            price_band_max_cents=100,
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await signal_engine.stop()

        assert update.approved is True
        assert update.side == "NO"
        assert update.block_reason is None

    asyncio.run(run())


def test_signal_risk_rejects_ltp_trap_when_spread_destroys_no_edge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.58))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(
            _ticker_update(
                last_yes_price_cents=66,
                previous_yes_price_cents=65,
                yes_bid_cents=60,
                yes_ask_cents=66,
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "spread_destroyed_edge"
        assert update.raw_model_edge is not None
        assert update.raw_model_edge < 0
        assert update.no_post_cost_edge is not None
        assert update.no_post_cost_edge < signal_engine.config.edge_threshold
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_no_stacking_blocks_second_same_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer, KalshiSignalRiskConfig())
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True

        await collector._publish_update(_ticker_update(last_yes_price_cents=56, previous_yes_price_cents=55))
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is False
        assert second.block_reason == "ticker_locked"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_stacking_enabled_allows_second_same_ticker_when_signature_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(allow_stacking=True),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True

        await collector._publish_update(_ticker_update(last_yes_price_cents=65, previous_yes_price_cents=55))
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is True
        assert second.trade_intent is not None
        assert second.price_bucket != first.price_bucket
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_stacking_blocks_duplicate_signature_after_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(allow_stacking=True),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()

        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        assert first.trade_intent is not None

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="accepted",
                event_time=datetime.now(UTC),
            )
        )
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is False
        assert second.block_reason == "duplicate_signature"
        assert second.tau_bucket == first.tau_bucket
        assert second.price_bucket == first.price_bucket
        assert second.chosen_side_probability_bucket == first.chosen_side_probability_bucket
        assert second.chosen_side_edge_bucket == first.chosen_side_edge_bucket

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_stacking_blocks_duplicate_signature_after_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(allow_stacking=True),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()

        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        assert first.trade_intent is not None

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="filled",
                event_time=datetime.now(UTC),
                filled_contracts=first.trade_intent.contracts,
            )
        )
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is False
        assert second.block_reason == "duplicate_signature"

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_trade_cooldown_blocks_rapid_reentry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(allow_stacking=True, trade_cooldown_seconds=0.2),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()

        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True

        await collector._publish_update(_ticker_update(last_yes_price_cents=65, previous_yes_price_cents=55))
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is False
        assert second.block_reason == "cooldown_active"

        await asyncio.sleep(0.25)
        await collector._publish_update(_ticker_update(last_yes_price_cents=65, previous_yes_price_cents=64))
        third = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert third.approved is True

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_stacking_retry_after_reservation_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(allow_stacking=True, reservation_ttl_seconds=0.1),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()

        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True

        await asyncio.sleep(0.25)
        await collector._publish_update(_ticker_update(last_yes_price_cents=56, previous_yes_price_cents=55))
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is True
        assert second.trade_intent is not None
        assert first.trade_intent is not None
        assert second.trade_intent.decision_id != first.trade_intent.decision_id

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_blocks_insufficient_cash_and_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    cash_block_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(starting_cash_dollars=0.40),
    )
    reserve_block_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(starting_cash_dollars=10.0, reserve_cash_pct=95.0),
    )
    cash_queue = cash_block_engine.subscribe_queue()
    reserve_queue = reserve_block_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await cash_block_engine.start()
        await reserve_block_engine.start()
        await collector._publish_update(_ticker_update())
        cash_update = await asyncio.wait_for(cash_queue.get(), timeout=0.5)
        reserve_update = await asyncio.wait_for(reserve_queue.get(), timeout=0.5)
        assert cash_update.approved is False
        assert cash_update.block_reason == "insufficient_cash"
        assert reserve_update.approved is False
        assert reserve_update.block_reason == "reserve_violation"
        await cash_block_engine.stop()
        await reserve_block_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_reservation_expiry_unlocks_ticker_and_cash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(reservation_ttl_seconds=0.1),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update(last_yes_price_cents=55, previous_yes_price_cents=54))
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        await asyncio.sleep(0.25)
        portfolio_state = signal_engine.get_portfolio_state()
        assert portfolio_state.pending_reservations == ()
        assert portfolio_state.available_cash_dollars == pytest.approx(signal_engine.config.starting_cash_dollars)

        await collector._publish_update(_ticker_update(last_yes_price_cents=56, previous_yes_price_cents=55))
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is True
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_external_portfolio_snapshot_overrides_cash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await signal_engine.apply_portfolio_snapshot(
            KalshiPortfolioSnapshot(
                event_time=datetime.now(UTC),
                available_cash_dollars=0.10,
                deployed_capital_dollars=0.0,
                open_positions=(),
            )
        )
        await collector._publish_update(_ticker_update())
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is False
        assert update.block_reason == "insufficient_cash"
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_execution_feedback_updates_and_releases_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update())
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        assert first.trade_intent is not None

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="accepted",
                event_time=datetime.now(UTC),
            )
        )
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is False
        assert second.block_reason == "ticker_locked"

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="released",
                event_time=datetime.now(UTC),
            )
        )
        third = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert third.approved is True
        assert third.trade_intent is not None
        assert third.trade_intent.decision_id != first.trade_intent.decision_id
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_late_accepted_feedback_recovers_recently_expired_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(reservation_ttl_seconds=0.1),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update())
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        assert first.trade_intent is not None

        await asyncio.sleep(0.25)
        assert signal_engine.get_portfolio_state().pending_reservations == ()

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="accepted",
                event_time=datetime.now(UTC),
            )
        )

        portfolio_state = signal_engine.get_portfolio_state()
        assert len(portfolio_state.pending_reservations) == 1
        assert portfolio_state.pending_reservations[0].decision_id == first.trade_intent.decision_id

        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.approved is False
        assert second.block_reason == "ticker_locked"

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_manual_trade_reservation_builds_layering_metadata(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.55,
                tau_minutes=9.0,
                predicted_yes_probability=0.78,
                model_edge=0.23,
                model_file=Path("fake-model.txt"),
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
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(auto_reserve_trade_intents=False, starting_cash_dollars=100.0, contracts_per_order=5),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.trade_intent is None

        manual = signal_engine.reserve_manual_trade_intent(
            decision_state=update,
            side="YES",
            entry_price_cents=56,
            contracts=2,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
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

        assert manual is not None
        assert manual.thesis_id == "thesis-1"
        assert manual.tranche_window == "10m"
        assert manual.tranche_reason == "opened"
        assert manual.lifecycle_state == "probe_pending"
        assert manual.contracts == 2
        assert manual.expected_value_dollars == pytest.approx(0.19)
        assert manual.worst_case_loss_dollars == pytest.approx(0.57)

        await signal_engine.stop()

    asyncio.run(run())


def test_signal_risk_manual_trade_reservation_respects_ticker_side_exposure_cap(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.20,
                tau_minutes=9.0,
                predicted_yes_probability=0.20,
                model_edge=-0.60,
                model_file=Path("fake-model.txt"),
                last_yes_price_cents=20,
                last_price_cents=20,
                yes_bid_cents=72,
                yes_ask_cents=74,
                no_bid_cents=26,
                no_ask_cents=28,
                ticker_update_time=event_time,
                trade_yes_prob=0.20,
                quote_mid_prob=0.73,
                quote_spread_cents=2,
                buy_yes_price_cents=74,
                buy_no_price_cents=28,
                quote_age_seconds=0.0,
                last_to_mid_gap=-0.53,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            allow_stacking=True,
            max_ticker_side_exposure_dollars=1.0,
            enable_bucket_ban_policy=False,
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.side == "NO"
        assert update.trade_intent is None

        await signal_engine.apply_portfolio_snapshot(
            KalshiPortfolioSnapshot(
                event_time=event_time,
                available_cash_dollars=9.44,
                deployed_capital_dollars=0.55,
                open_positions=(
                    KalshiPortfolioPosition(
                        ticker="KXBTC15M-TEST",
                        side="NO",
                        contracts=3,
                        entry_cost_dollars=0.84,
                        fees_dollars=0.01,
                        cash_required_dollars=0.85,
                    ),
                ),
            )
        )

        resolved_contracts = signal_engine.resolve_manual_trade_contracts(
            decision_state=update,
            side="NO",
            entry_price_cents=28,
        )
        assert resolved_contracts == 0

        manual = signal_engine.reserve_manual_trade_intent(
            decision_state=update,
            side="NO",
            entry_price_cents=28,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        await signal_engine.stop()

        assert manual is None

    asyncio.run(run())


def test_signal_risk_manual_trade_contracts_are_clamped_by_ticker_side_exposure_cap(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    event_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                ticker="KXBTC15M-TEST",
                event_time=event_time,
                market_prob=0.55,
                tau_minutes=9.0,
                predicted_yes_probability=0.78,
                model_edge=0.23,
                model_file=Path("fake-model.txt"),
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
                quote_age_seconds=0.0,
                last_to_mid_gap=0.0,
            )
        },
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            allow_stacking=True,
            contracts_per_order=5,
            starting_cash_dollars=100.0,
            max_ticker_side_exposure_dollars=1.0,
            enable_bucket_ban_policy=False,
        ),
    )
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.approved is True
        assert update.trade_intent is None

        await signal_engine.apply_portfolio_snapshot(
            KalshiPortfolioSnapshot(
                event_time=event_time,
                available_cash_dollars=99.44,
                deployed_capital_dollars=0.55,
                open_positions=(
                    KalshiPortfolioPosition(
                        ticker="KXBTC15M-TEST",
                        side="YES",
                        contracts=1,
                        entry_cost_dollars=0.55,
                        fees_dollars=0.01,
                        cash_required_dollars=0.56,
                    ),
                ),
            )
        )

        resolved_contracts = signal_engine.resolve_manual_trade_contracts(
            decision_state=update,
            side="YES",
            entry_price_cents=30,
        )
        assert resolved_contracts == 1

        manual = signal_engine.reserve_manual_trade_intent(
            decision_state=update,
            side="YES",
            entry_price_cents=30,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual is not None
        assert manual.contracts == 1
        await signal_engine.stop()

    asyncio.run(run())


def test_signal_risk_portfolio_snapshot_replaces_local_open_positions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update())
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        assert first.trade_intent is not None

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="accepted",
                event_time=datetime.now(UTC),
            )
        )
        await asyncio.wait_for(queue.get(), timeout=0.5)

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="filled",
                event_time=datetime.now(UTC),
                filled_contracts=1,
                filled_price_cents=55,
            )
        )
        await asyncio.wait_for(queue.get(), timeout=0.5)
        assert len(signal_engine.get_portfolio_state().open_positions) == 1

        await signal_engine.apply_portfolio_snapshot(
            KalshiPortfolioSnapshot(
                event_time=datetime.now(UTC),
                available_cash_dollars=9.44,
                deployed_capital_dollars=0.55,
                open_positions=(
                    KalshiPortfolioPosition(
                        ticker="KXBTC15M-TEST",
                        side="YES",
                        contracts=1,
                        entry_cost_dollars=0.55,
                        fees_dollars=0.01,
                        cash_required_dollars=0.56,
                    ),
                ),
            )
        )

        portfolio_state = signal_engine.get_portfolio_state()
        assert len(portfolio_state.open_positions) == 1
        assert portfolio_state.available_cash_dollars == pytest.approx(9.44)

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_signal_risk_released_feedback_applies_cash_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    queue = signal_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await collector._publish_update(_ticker_update())
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.approved is True
        assert first.trade_intent is not None

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="accepted",
                event_time=datetime.now(UTC),
            )
        )
        await asyncio.wait_for(queue.get(), timeout=0.5)

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="filled",
                event_time=datetime.now(UTC),
                filled_contracts=1,
                filled_price_cents=55,
            )
        )
        await asyncio.wait_for(queue.get(), timeout=0.5)
        before_release = signal_engine.get_portfolio_state()
        assert before_release.available_cash_dollars < 10.0

        await signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=first.trade_intent.decision_id,
                status="released",
                event_time=datetime.now(UTC),
                cash_delta_dollars=0.44,
            )
        )

        portfolio_state = signal_engine.get_portfolio_state()
        assert portfolio_state.available_cash_dollars == pytest.approx(10.44)
        assert len(portfolio_state.open_positions) == 0

        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())
