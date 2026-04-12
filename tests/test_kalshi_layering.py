from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiExecutionConfig,
    KalshiExecutionEngine,
    KalshiExecutionMode,
    KalshiLightGBMScoreState,
    KalshiMarketDataCollector,
    KalshiPathDependentBinaryLayeringConfig,
    KalshiPathDependentBinaryLayeringEngine,
    KalshiSignalRiskConfig,
    KalshiSignalRiskEngine,
    KalshiTickerState,
)


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


class _LayeringFakeScorer:
    def __init__(self, collector: KalshiMarketDataCollector):
        self.feature_engine = SimpleNamespace(collector=collector)
        self._states: dict[str, KalshiLightGBMScoreState] = {}
        self._queues: list[asyncio.Queue[KalshiLightGBMScoreState]] = []

    def snapshot_states(self) -> dict[str, KalshiLightGBMScoreState]:
        return dict(self._states)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLightGBMScoreState]:
        queue: asyncio.Queue[KalshiLightGBMScoreState] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    async def publish(self, state: KalshiLightGBMScoreState) -> None:
        self._states[state.ticker] = state
        for queue in self._queues:
            await queue.put(state)


def _score_state(
    *,
    ticker: str,
    event_time: datetime,
    tau_minutes: float,
    predicted_yes_probability: float,
    last_yes_price_cents: int,
    yes_bid_cents: int,
    yes_ask_cents: int,
) -> KalshiLightGBMScoreState:
    return KalshiLightGBMScoreState(
        ticker=ticker,
        event_time=event_time,
        market_prob=0.55,
        tau_minutes=tau_minutes,
        predicted_yes_probability=predicted_yes_probability,
        model_edge=predicted_yes_probability - 0.55,
        model_file=Path("fake-model.txt"),
        last_yes_price_cents=last_yes_price_cents,
        last_price_cents=last_yes_price_cents,
        yes_bid_cents=yes_bid_cents,
        yes_ask_cents=yes_ask_cents,
        no_bid_cents=100 - yes_ask_cents,
        no_ask_cents=100 - yes_bid_cents,
        ticker_update_time=event_time,
        trade_yes_prob=0.55,
        quote_mid_prob=(yes_bid_cents + yes_ask_cents) / 200.0,
        quote_spread_cents=yes_ask_cents - yes_bid_cents,
        buy_yes_price_cents=yes_ask_cents,
        buy_no_price_cents=100 - yes_bid_cents,
        quote_age_seconds=0.0,
        last_to_mid_gap=0.0,
    )


async def _wait_for_execution_fill(queue: asyncio.Queue, tranche_index: int, *, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for filled tranche {tranche_index}")
        update = await asyncio.wait_for(queue.get(), timeout=remaining)
        if update.status == "filled" and update.tranche_index == tranche_index:
            return update


async def _wait_for_layering_action(queue: asyncio.Queue, action: str, *, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for layering action {action}")
        update = await asyncio.wait_for(queue.get(), timeout=remaining)
        if update.action == action:
            return update


def test_layering_engine_requires_10m_window_for_thesis_open(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 8, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
        is_open=True,
        yes_bid_cents=54,
        yes_ask_cents=56,
        ticker_update_time=datetime(2026, 1, 1, 12, 8, tzinfo=UTC),
    )
    scorer = _LayeringFakeScorer(collector)
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=10,
            min_tau_minutes=0.0,
            enable_bucket_ban_policy=False,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(signal_engine, log_dir=tmp_path / "layering")
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
        trade_intent_source=layering_engine,
    )
    layering_engine.bind_execution_engine(execution_engine)
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        await execution_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 10, 30, tzinfo=UTC),
                tau_minutes=4.5,
                predicted_yes_probability=0.78,
                last_yes_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )

        await asyncio.sleep(0.1)
        assert layering_engine.get_active_ledger(ticker) is None
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(execution_queue.get(), timeout=0.1)

        await layering_engine.stop()
        await execution_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_layering_engine_same_side_add_flow(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=30,
        last_trade_time=datetime(2026, 1, 1, 12, 3, tzinfo=UTC),
        previous_yes_price_cents=29,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=29,
        yes_ask_cents=31,
        ticker_update_time=datetime(2026, 1, 1, 12, 3, tzinfo=UTC),
    )
    scorer = _LayeringFakeScorer(collector)
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=10,
            allow_stacking=True,
            min_tau_minutes=0.0,
            enable_bucket_ban_policy=False,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(signal_engine, log_dir=tmp_path / "layering")
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
        trade_intent_source=layering_engine,
    )
    layering_engine.bind_execution_engine(execution_engine)
    layering_queue = layering_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        await execution_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 3, tzinfo=UTC),
                tau_minutes=12.0,
                predicted_yes_probability=0.76,
                last_yes_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )
        await asyncio.sleep(0.1)
        assert layering_engine.get_active_ledger(ticker) is None

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.78,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        opened = await _wait_for_layering_action(layering_queue, "opened")
        first_fill = await _wait_for_execution_fill(execution_queue, 0)
        assert opened.decision_window == "10m"
        assert first_fill.side == "YES"

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 8, tzinfo=UTC),
                tau_minutes=7.0,
                predicted_yes_probability=0.80,
                last_yes_price_cents=33,
                yes_bid_cents=32,
                yes_ask_cents=34,
            )
        )
        await asyncio.sleep(0.1)
        assert len(layering_engine.get_active_ledger(ticker).tranches) == 1

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 10, 30, tzinfo=UTC),
                tau_minutes=4.5,
                predicted_yes_probability=0.79,
                last_yes_price_cents=35,
                yes_bid_cents=34,
                yes_ask_cents=36,
            )
        )
        add_yes = await _wait_for_layering_action(layering_queue, "same_side_add")
        second_fill = await _wait_for_execution_fill(execution_queue, 1)
        assert add_yes.decision_window == "5m"
        assert second_fill.side == "YES"

        ledger = layering_engine.get_active_ledger(ticker)
        assert ledger is not None
        assert ledger.decision_windows_hit == ("10m", "5m")
        assert ledger.lifecycle_state == "scaling_active"
        assert [tranche.side for tranche in ledger.tranches] == ["YES", "YES"]
        assert [tranche.decision_window for tranche in ledger.tranches] == ["10m", "5m"]

        await layering_engine.stop()
        await execution_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_layering_engine_flip_and_reflip_flow(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=30,
        last_trade_time=datetime(2026, 1, 1, 12, 3, tzinfo=UTC),
        previous_yes_price_cents=29,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=29,
        yes_ask_cents=31,
        ticker_update_time=datetime(2026, 1, 1, 12, 3, tzinfo=UTC),
    )
    scorer = _LayeringFakeScorer(collector)
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=10,
            allow_stacking=True,
            min_tau_minutes=0.0,
            enable_bucket_ban_policy=False,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(signal_engine, log_dir=tmp_path / "layering-flip")
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
        trade_intent_source=layering_engine,
    )
    layering_engine.bind_execution_engine(execution_engine)
    layering_queue = layering_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        await execution_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.78,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        opened = await _wait_for_layering_action(layering_queue, "opened")
        first_fill = await _wait_for_execution_fill(execution_queue, 0)
        assert opened.decision_window == "10m"
        assert first_fill.side == "YES"

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 12, 30, tzinfo=UTC),
                tau_minutes=3.5,
                predicted_yes_probability=0.05,
                last_yes_price_cents=41,
                yes_bid_cents=40,
                yes_ask_cents=42,
            )
        )
        add_no = await _wait_for_layering_action(layering_queue, "flip_add")
        third_fill = await _wait_for_execution_fill(execution_queue, 1)
        assert add_no.decision_window == "4m"
        assert third_fill.side == "NO"

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 14, 30, tzinfo=UTC),
                tau_minutes=2.5,
                predicted_yes_probability=0.90,
                last_yes_price_cents=50,
                yes_bid_cents=49,
                yes_ask_cents=51,
            )
        )
        add_yes_again = await _wait_for_layering_action(layering_queue, "flip_add")
        fourth_fill = await _wait_for_execution_fill(execution_queue, 2)
        assert add_yes_again.decision_window == "3m"
        assert fourth_fill.side == "YES"

        ledger = layering_engine.get_active_ledger(ticker)
        assert ledger is not None
        assert ledger.decision_windows_hit == ("10m", "4m", "3m")
        assert ledger.lifecycle_state == "max_size_reached"
        assert [tranche.side for tranche in ledger.tranches] == ["YES", "NO", "YES"]
        assert [tranche.decision_window for tranche in ledger.tranches] == ["10m", "4m", "3m"]

        await layering_engine.stop()
        await execution_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_layering_engine_opens_probe_even_when_base_signal_size_is_one_contract(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=50,
        last_trade_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        previous_yes_price_cents=49,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=49,
        yes_ask_cents=50,
        ticker_update_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
    )
    scorer = _LayeringFakeScorer(collector)
    signal_engine = KalshiSignalRiskEngine(
        scorer,  # type: ignore[arg-type]
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
            allow_stacking=True,
            min_tau_minutes=0.0,
            enable_bucket_ban_policy=False,
            reserve_cash_pct=0.0,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(
        signal_engine,
        KalshiPathDependentBinaryLayeringConfig(minimum_contracts_per_thesis=5),
        log_dir=tmp_path / "layering-min-contracts",
    )
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
        trade_intent_source=layering_engine,
    )
    layering_engine.bind_execution_engine(execution_engine)
    layering_queue = layering_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await signal_engine.start()
        await execution_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.75,
                last_yes_price_cents=50,
                yes_bid_cents=49,
                yes_ask_cents=50,
            )
        )

        opened = await _wait_for_layering_action(layering_queue, "opened")
        first_fill = await _wait_for_execution_fill(execution_queue, 0)

        assert opened.decision_window == "10m"
        assert first_fill.contracts >= 1
        ledger = layering_engine.get_active_ledger(ticker)
        assert ledger is not None
        assert ledger.total_thesis_budget_dollars >= first_fill.cash_required_dollars

        await layering_engine.stop()
        await execution_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())
