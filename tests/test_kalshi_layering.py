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
    KalshiExecutionUpdate,
    KalshiLightGBMScoreState,
    KalshiMarketDataCollector,
    KalshiPathDependentBinaryLayeringConfig,
    KalshiPathDependentBinaryLayeringEngine,
    KalshiSignalRiskConfig,
    KalshiSignalRiskEngine,
    KalshiTickerState,
)
from src.live.kalshi.signal_risk import find_max_acceptable_entry_price_cents


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


async def _wait_for_condition(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if predicate():
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for condition")
        await asyncio.sleep(min(0.01, remaining))


async def _wait_for_trade_intent(queue: asyncio.Queue, predicate, *, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for matching trade intent")
        intent = await asyncio.wait_for(queue.get(), timeout=remaining)
        if predicate(intent):
            return intent


async def _noop_private_ws(self) -> None:
    await self._stop_event.wait()


class _LiveRetryRestClient:
    def __init__(self, *, create_order_results: list[dict] | None = None):
        self.create_order_results = list(create_order_results or [])
        self.create_order_calls: list[dict] = []
        self.orders: dict[str, dict] = {}
        self.positions: list[dict] = []

    def close(self) -> None:
        return None

    def create_order(self, payload: dict) -> dict:
        self.create_order_calls.append(payload)
        result = self.create_order_results.pop(0)
        order = result.get("order")
        if isinstance(order, dict):
            order["client_order_id"] = payload["client_order_id"]
            order["ticker"] = payload["ticker"]
            order["side"] = payload["side"]
            order["action"] = payload["action"]
            order["type"] = payload["type"]
            order["initial_count"] = payload["count"]
            if payload.get("yes_price") is not None:
                order["yes_price"] = payload["yes_price"]
                order["no_price"] = 100 - int(payload["yes_price"])
            if payload.get("no_price") is not None:
                order["no_price"] = payload["no_price"]
                order["yes_price"] = 100 - int(payload["no_price"])
            self.orders[order["order_id"]] = order
        return result

    def get_balance(self, *, subaccount: int = 0) -> dict:
        del subaccount
        return {"balance": 10000, "portfolio_value": 10000, "updated_ts": 1767225600}

    def get_positions(self, *, subaccount: int = 0) -> list[dict]:
        del subaccount
        return list(self.positions)

    def get_orders(self, *_, **__) -> list[dict]:
        return list(self.orders.values())

    def get_order(self, order_id: str, *, subaccount: int | None = None) -> dict:
        del subaccount
        return {"order": self.orders[order_id]}

    def cancel_order(self, order_id: str) -> dict:
        order = self.orders[order_id]
        order["status"] = "canceled"
        return {"order": order, "reduced_by": 0}

    @staticmethod
    def build_order(
        payload: dict,
        *,
        order_id: str,
        status: str,
        fill_count: int,
        remaining_count: int = 0,
    ) -> dict:
        yes_price = payload.get("yes_price")
        no_price = payload.get("no_price")
        if yes_price is None and no_price is not None:
            yes_price = 100 - int(no_price)
        if no_price is None and yes_price is not None:
            no_price = 100 - int(yes_price)
        return {
            "order_id": order_id,
            "user_id": "user-1",
            "client_order_id": payload["client_order_id"],
            "ticker": payload["ticker"],
            "side": payload["side"],
            "action": payload["action"],
            "type": payload["type"],
            "status": status,
            "yes_price": yes_price,
            "no_price": no_price,
            "fill_count": fill_count,
            "remaining_count": remaining_count,
            "initial_count": payload["count"],
            "taker_fees_dollars": "0.0100",
            "maker_fees_dollars": "0.0000",
            "taker_fill_cost_dollars": "0.5500",
            "maker_fill_cost_dollars": "0.0000",
            "created_time": "2026-01-01T12:00:00Z",
            "last_update_time": "2026-01-01T12:00:01Z",
        }


class _DummyLiveExecutionEngine:
    def __init__(self) -> None:
        self.config = SimpleNamespace(mode=KalshiExecutionMode.LIVE, simulate_immediate_fills=False)
        self.queue: asyncio.Queue = asyncio.Queue()

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue:
        del maxsize
        return self.queue


def _execution_update_from_intent(
    intent,
    *,
    status: str,
    event_time: datetime,
    order_id: str,
    filled_contracts: int = 0,
    fill_price_cents: int | None = None,
    message: str | None = None,
) -> KalshiExecutionUpdate:
    remaining_contracts = max(0, intent.contracts - filled_contracts)
    return KalshiExecutionUpdate(
        decision_id=intent.decision_id,
        ticker=intent.ticker,
        side=intent.side,
        contracts=intent.contracts,
        mode=KalshiExecutionMode.LIVE,
        status=status,
        event_time=event_time,
        reference_price_cents=intent.reference_price_cents,
        limit_price_cents=intent.max_acceptable_entry_price_cents,
        client_order_id=intent.decision_id,
        order_id=order_id,
        filled_contracts=filled_contracts,
        remaining_contracts=remaining_contracts,
        fill_price_cents=fill_price_cents,
        entry_cost_dollars=0.0 if fill_price_cents is None else intent.estimated_entry_cost_dollars,
        fees_dollars=0.0 if fill_price_cents is None else intent.estimated_fees_dollars,
        cash_required_dollars=intent.estimated_cash_required_dollars,
        available_cash_dollars=0.0,
        realized_pnl_dollars=None,
        cumulative_realized_pnl_dollars=None,
        settlement_result=None,
        message=message,
        live_order=None,
        thesis_id=intent.thesis_id,
        tranche_index=intent.tranche_index,
        tranche_window=intent.tranche_window,
        tranche_reason=intent.tranche_reason,
        lifecycle_state=intent.lifecycle_state,
        total_thesis_budget_dollars=intent.total_thesis_budget_dollars,
        payout_if_yes_dollars=intent.payout_if_yes_dollars,
        payout_if_no_dollars=intent.payout_if_no_dollars,
        expected_value_dollars=intent.expected_value_dollars,
        worst_case_loss_dollars=intent.worst_case_loss_dollars,
        target_id=intent.target_id,
        attempt_index=intent.attempt_index,
        desired_contracts=intent.desired_contracts,
        remaining_contracts_before_submit=intent.remaining_contracts_before_submit,
        hard_max_price_cents=intent.hard_max_price_cents,
        retry_reason=intent.retry_reason,
        was_first_attempt=intent.was_first_attempt,
    )


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


def test_live_layering_retries_zero_fill_target_until_filled(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=54,
        yes_ask_cents=56,
        ticker_update_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        received_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
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
            price_band_min_cents=0,
            price_band_max_cents=100,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(
        signal_engine,
        KalshiPathDependentBinaryLayeringConfig(live_retry_cooldown_seconds=0.0),
        log_dir=tmp_path / "layering-live-retry",
    )
    execution_engine = _DummyLiveExecutionEngine()
    layering_engine.bind_execution_engine(execution_engine)
    layering_queue = layering_engine.subscribe_queue()
    trade_intent_queue = layering_engine.subscribe_trade_intent_queue()

    async def run() -> None:
        await signal_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.80,
                last_yes_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )

        target_created = await _wait_for_layering_action(layering_queue, "target_created")
        opened = await _wait_for_layering_action(layering_queue, "opened")
        first_intent = await asyncio.wait_for(trade_intent_queue.get(), timeout=1.0)
        target_attempt = await _wait_for_layering_action(layering_queue, "target_attempt_submitted")

        cancelled_update = KalshiExecutionUpdate(
            decision_id=first_intent.decision_id,
            ticker=ticker,
            side=first_intent.side,
            contracts=first_intent.contracts,
            mode=KalshiExecutionMode.LIVE,
            status="cancelled",
            event_time=datetime(2026, 1, 1, 12, 5, 31, tzinfo=UTC),
            reference_price_cents=first_intent.reference_price_cents,
            limit_price_cents=first_intent.max_acceptable_entry_price_cents,
            client_order_id=first_intent.decision_id,
            order_id="order-1",
            filled_contracts=0,
            remaining_contracts=0,
            fill_price_cents=None,
            entry_cost_dollars=0.0,
            fees_dollars=0.0,
            cash_required_dollars=first_intent.estimated_cash_required_dollars,
            available_cash_dollars=signal_engine.get_portfolio_state().available_cash_dollars,
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            settlement_result=None,
            message="cancelled_zero_fill",
            live_order=None,
            thesis_id=first_intent.thesis_id,
            tranche_index=first_intent.tranche_index,
            tranche_window=first_intent.tranche_window,
            tranche_reason=first_intent.tranche_reason,
            lifecycle_state=first_intent.lifecycle_state,
            total_thesis_budget_dollars=first_intent.total_thesis_budget_dollars,
            payout_if_yes_dollars=first_intent.payout_if_yes_dollars,
            payout_if_no_dollars=first_intent.payout_if_no_dollars,
            expected_value_dollars=first_intent.expected_value_dollars,
            worst_case_loss_dollars=first_intent.worst_case_loss_dollars,
            target_id=first_intent.target_id,
            attempt_index=first_intent.attempt_index,
            desired_contracts=first_intent.desired_contracts,
            remaining_contracts_before_submit=first_intent.remaining_contracts_before_submit,
            hard_max_price_cents=first_intent.hard_max_price_cents,
            retry_reason=first_intent.retry_reason,
            was_first_attempt=first_intent.was_first_attempt,
        )
        await layering_engine._handle_execution_update(cancelled_update)

        second_intent = await asyncio.wait_for(trade_intent_queue.get(), timeout=1.0)

        filled_update = KalshiExecutionUpdate(
            decision_id=second_intent.decision_id,
            ticker=ticker,
            side=second_intent.side,
            contracts=second_intent.contracts,
            mode=KalshiExecutionMode.LIVE,
            status="filled",
            event_time=datetime(2026, 1, 1, 12, 5, 32, tzinfo=UTC),
            reference_price_cents=second_intent.reference_price_cents,
            limit_price_cents=second_intent.max_acceptable_entry_price_cents,
            client_order_id=second_intent.decision_id,
            order_id="order-2",
            filled_contracts=1,
            remaining_contracts=0,
            fill_price_cents=second_intent.max_acceptable_entry_price_cents,
            entry_cost_dollars=second_intent.estimated_entry_cost_dollars,
            fees_dollars=second_intent.estimated_fees_dollars,
            cash_required_dollars=second_intent.estimated_cash_required_dollars,
            available_cash_dollars=signal_engine.get_portfolio_state().available_cash_dollars,
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            settlement_result=None,
            message="order_terminal_fill",
            live_order=None,
            thesis_id=second_intent.thesis_id,
            tranche_index=second_intent.tranche_index,
            tranche_window=second_intent.tranche_window,
            tranche_reason=second_intent.tranche_reason,
            lifecycle_state=second_intent.lifecycle_state,
            total_thesis_budget_dollars=second_intent.total_thesis_budget_dollars,
            payout_if_yes_dollars=second_intent.payout_if_yes_dollars,
            payout_if_no_dollars=second_intent.payout_if_no_dollars,
            expected_value_dollars=second_intent.expected_value_dollars,
            worst_case_loss_dollars=second_intent.worst_case_loss_dollars,
            target_id=second_intent.target_id,
            attempt_index=second_intent.attempt_index,
            desired_contracts=second_intent.desired_contracts,
            remaining_contracts_before_submit=second_intent.remaining_contracts_before_submit,
            hard_max_price_cents=second_intent.hard_max_price_cents,
            retry_reason=second_intent.retry_reason,
            was_first_attempt=second_intent.was_first_attempt,
        )
        await layering_engine._handle_execution_update(filled_update)

        ledger = layering_engine.get_active_ledger(ticker)
        assert ledger is not None
        tranche = ledger.tranches[0]

        assert opened.decision_window == "10m"
        assert target_created.target_id is not None
        assert first_intent.target_id == target_created.target_id
        assert target_attempt.target_id == target_created.target_id
        assert first_intent.attempt_index == 1
        assert first_intent.was_first_attempt is True
        assert second_intent.target_id == first_intent.target_id
        assert second_intent.attempt_index == 2
        assert second_intent.was_first_attempt is False
        assert second_intent.retry_reason == "execution_cancelled"
        assert tranche.attempt_count == 2
        assert tranche.filled_contracts == 1
        assert tranche.status in {"filled", "settled"}

        await layering_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_live_layering_same_price_retry_waits_for_cooldown(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=54,
        yes_ask_cents=56,
        ticker_update_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        received_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
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
            price_band_min_cents=0,
            price_band_max_cents=100,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(
        signal_engine,
        KalshiPathDependentBinaryLayeringConfig(live_retry_cooldown_seconds=1.0),
        log_dir=tmp_path / "layering-live-retry-cooldown",
    )
    execution_engine = _DummyLiveExecutionEngine()
    layering_engine.bind_execution_engine(execution_engine)
    trade_intent_queue = layering_engine.subscribe_trade_intent_queue()

    async def run() -> None:
        await signal_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.80,
                last_yes_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )
        first_intent = await asyncio.wait_for(trade_intent_queue.get(), timeout=1.0)

        cancelled_update = KalshiExecutionUpdate(
            decision_id=first_intent.decision_id,
            ticker=ticker,
            side=first_intent.side,
            contracts=first_intent.contracts,
            mode=KalshiExecutionMode.LIVE,
            status="cancelled",
            event_time=datetime(2026, 1, 1, 12, 5, 30, 100000, tzinfo=UTC),
            reference_price_cents=first_intent.reference_price_cents,
            limit_price_cents=first_intent.max_acceptable_entry_price_cents,
            client_order_id=first_intent.decision_id,
            order_id="order-1",
            filled_contracts=0,
            remaining_contracts=0,
            fill_price_cents=None,
            entry_cost_dollars=0.0,
            fees_dollars=0.0,
            cash_required_dollars=first_intent.estimated_cash_required_dollars,
            available_cash_dollars=signal_engine.get_portfolio_state().available_cash_dollars,
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            settlement_result=None,
            message="cancelled_zero_fill",
            live_order=None,
            thesis_id=first_intent.thesis_id,
            tranche_index=first_intent.tranche_index,
            tranche_window=first_intent.tranche_window,
            tranche_reason=first_intent.tranche_reason,
            lifecycle_state=first_intent.lifecycle_state,
            total_thesis_budget_dollars=first_intent.total_thesis_budget_dollars,
            payout_if_yes_dollars=first_intent.payout_if_yes_dollars,
            payout_if_no_dollars=first_intent.payout_if_no_dollars,
            expected_value_dollars=first_intent.expected_value_dollars,
            worst_case_loss_dollars=first_intent.worst_case_loss_dollars,
            target_id=first_intent.target_id,
            attempt_index=first_intent.attempt_index,
            desired_contracts=first_intent.desired_contracts,
            remaining_contracts_before_submit=first_intent.remaining_contracts_before_submit,
            hard_max_price_cents=first_intent.hard_max_price_cents,
            retry_reason=first_intent.retry_reason,
            was_first_attempt=first_intent.was_first_attempt,
        )
        await layering_engine._handle_execution_update(cancelled_update)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(trade_intent_queue.get(), timeout=0.05)

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, 500000, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.80,
                last_yes_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(trade_intent_queue.get(), timeout=0.05)

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 31, 500000, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.80,
                last_yes_price_cents=55,
                yes_bid_cents=54,
                yes_ask_cents=56,
            )
        )
        second_intent = await asyncio.wait_for(trade_intent_queue.get(), timeout=1.0)

        assert second_intent.target_id == first_intent.target_id
        assert second_intent.attempt_index == 2

        await layering_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_live_layering_refreshes_target_cap_on_signal_cadence(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    signal_config = KalshiSignalRiskConfig(
        auto_reserve_trade_intents=False,
        starting_cash_dollars=100.0,
        contracts_per_order=1,
        allow_stacking=True,
        min_tau_minutes=0.0,
        enable_bucket_ban_policy=False,
        price_band_min_cents=0,
        price_band_max_cents=100,
        edge_threshold_cents=4.0,
        maintain_edge_cents=1.0,
    )
    predicted_yes_probability = None
    enter_cap = None
    maintain_cap = None
    for candidate_probability in (0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76):
        candidate_enter = find_max_acceptable_entry_price_cents(
            side="YES",
            predicted_yes_probability=candidate_probability,
            config=signal_config,
            contracts=1,
            edge_threshold_cents=signal_config.edge_threshold_cents,
        )
        candidate_maintain = find_max_acceptable_entry_price_cents(
            side="YES",
            predicted_yes_probability=candidate_probability,
            config=signal_config,
            contracts=1,
            edge_threshold_cents=signal_config.maintain_edge_cents,
        )
        if (
            candidate_enter is not None
            and candidate_maintain is not None
            and candidate_maintain > candidate_enter
            and candidate_enter >= 2
        ):
            predicted_yes_probability = candidate_probability
            enter_cap = candidate_enter
            maintain_cap = candidate_maintain
            break
    assert predicted_yes_probability is not None
    assert enter_cap is not None
    assert maintain_cap is not None

    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=enter_cap,
        last_trade_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        previous_yes_price_cents=max(1, enter_cap - 1),
        close_time=close_time,
        is_open=True,
        yes_bid_cents=max(1, enter_cap - 1),
        yes_ask_cents=enter_cap,
        ticker_update_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        received_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
    )
    scorer = _LayeringFakeScorer(collector)
    signal_engine = KalshiSignalRiskEngine(scorer, signal_config)  # type: ignore[arg-type]
    layering_engine = KalshiPathDependentBinaryLayeringEngine(
        signal_engine,
        KalshiPathDependentBinaryLayeringConfig(
            live_retry_cooldown_seconds=1.0,
            live_signal_refresh_min_seconds=1.0,
        ),
        log_dir=tmp_path / "layering-live-cap-refresh",
    )
    execution_engine = _DummyLiveExecutionEngine()
    layering_engine.bind_execution_engine(execution_engine)
    trade_intent_queue = layering_engine.subscribe_trade_intent_queue()

    async def run() -> None:
        await signal_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=predicted_yes_probability,
                last_yes_price_cents=enter_cap,
                yes_bid_cents=max(1, enter_cap - 1),
                yes_ask_cents=enter_cap,
            )
        )
        first_intent = await asyncio.wait_for(trade_intent_queue.get(), timeout=1.0)
        assert first_intent.hard_max_price_cents == enter_cap

        cancelled_update = KalshiExecutionUpdate(
            decision_id=first_intent.decision_id,
            ticker=ticker,
            side=first_intent.side,
            contracts=first_intent.contracts,
            mode=KalshiExecutionMode.LIVE,
            status="cancelled",
            event_time=datetime(2026, 1, 1, 12, 5, 30, 100000, tzinfo=UTC),
            reference_price_cents=first_intent.reference_price_cents,
            limit_price_cents=first_intent.max_acceptable_entry_price_cents,
            client_order_id=first_intent.decision_id,
            order_id="order-1",
            filled_contracts=0,
            remaining_contracts=0,
            fill_price_cents=None,
            entry_cost_dollars=0.0,
            fees_dollars=0.0,
            cash_required_dollars=first_intent.estimated_cash_required_dollars,
            available_cash_dollars=signal_engine.get_portfolio_state().available_cash_dollars,
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            settlement_result=None,
            message="cancelled_zero_fill",
            live_order=None,
            thesis_id=first_intent.thesis_id,
            tranche_index=first_intent.tranche_index,
            tranche_window=first_intent.tranche_window,
            tranche_reason=first_intent.tranche_reason,
            lifecycle_state=first_intent.lifecycle_state,
            total_thesis_budget_dollars=first_intent.total_thesis_budget_dollars,
            payout_if_yes_dollars=first_intent.payout_if_yes_dollars,
            payout_if_no_dollars=first_intent.payout_if_no_dollars,
            expected_value_dollars=first_intent.expected_value_dollars,
            worst_case_loss_dollars=first_intent.worst_case_loss_dollars,
            target_id=first_intent.target_id,
            attempt_index=first_intent.attempt_index,
            desired_contracts=first_intent.desired_contracts,
            remaining_contracts_before_submit=first_intent.remaining_contracts_before_submit,
            hard_max_price_cents=first_intent.hard_max_price_cents,
            retry_reason=first_intent.retry_reason,
            was_first_attempt=first_intent.was_first_attempt,
        )
        await layering_engine._handle_execution_update(cancelled_update)

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 31, 500000, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=predicted_yes_probability,
                last_yes_price_cents=enter_cap,
                yes_bid_cents=max(1, enter_cap - 1),
                yes_ask_cents=enter_cap,
            )
        )
        second_intent = await asyncio.wait_for(trade_intent_queue.get(), timeout=1.0)
        ledger = layering_engine.get_active_ledger(ticker)
        assert ledger is not None

        assert ledger.tranches[0].hard_max_price_cents == maintain_cap
        assert second_intent.hard_max_price_cents == maintain_cap
        assert second_intent.attempt_index == 2

        await layering_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_live_layering_reduces_5m_to_recovery_probe_without_prior_fill(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=30,
        last_trade_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        previous_yes_price_cents=29,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=29,
        yes_ask_cents=31,
        ticker_update_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        received_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
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
            price_band_min_cents=0,
            price_band_max_cents=100,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(signal_engine, log_dir=tmp_path / "layering-live-recovery")
    execution_engine = _DummyLiveExecutionEngine()
    layering_engine.bind_execution_engine(execution_engine)
    trade_intent_queue = layering_engine.subscribe_trade_intent_queue()

    async def run() -> None:
        await signal_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.76,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        first_intent = await _wait_for_trade_intent(
            trade_intent_queue,
            lambda intent: intent.tranche_window == "10m",
            timeout=1.0,
        )

        cancelled_update = _execution_update_from_intent(
            first_intent,
            status="cancelled",
            event_time=datetime(2026, 1, 1, 12, 5, 30, 100000, tzinfo=UTC),
            order_id="order-10m",
            message="cancelled_zero_fill",
        )
        await layering_engine._handle_execution_update(cancelled_update)

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 10, 30, tzinfo=UTC),
                tau_minutes=4.5,
                predicted_yes_probability=0.76,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        recovery_intent = await _wait_for_trade_intent(
            trade_intent_queue,
            lambda intent: intent.tranche_window == "5m",
            timeout=1.0,
        )
        ledger = layering_engine.get_active_ledger(ticker)
        assert ledger is not None

        assert recovery_intent.contracts == 1
        assert ledger.tranches[1].requested_contracts == 1

        await layering_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())


def test_live_layering_blocks_4m_until_recovery_probe_fills(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=30,
        last_trade_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        previous_yes_price_cents=29,
        close_time=close_time,
        is_open=True,
        yes_bid_cents=29,
        yes_ask_cents=31,
        ticker_update_time=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        received_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
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
            price_band_min_cents=0,
            price_band_max_cents=100,
        ),
    )
    layering_engine = KalshiPathDependentBinaryLayeringEngine(signal_engine, log_dir=tmp_path / "layering-live-initial-fill")
    execution_engine = _DummyLiveExecutionEngine()
    layering_engine.bind_execution_engine(execution_engine)
    layering_queue = layering_engine.subscribe_queue()
    trade_intent_queue = layering_engine.subscribe_trade_intent_queue()

    async def run() -> None:
        await signal_engine.start()
        await layering_engine.start()

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 5, 30, tzinfo=UTC),
                tau_minutes=9.5,
                predicted_yes_probability=0.76,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        first_intent = await _wait_for_trade_intent(
            trade_intent_queue,
            lambda intent: intent.tranche_window == "10m",
            timeout=1.0,
        )

        await layering_engine._handle_execution_update(
            _execution_update_from_intent(
                first_intent,
                status="cancelled",
                event_time=datetime(2026, 1, 1, 12, 5, 30, 100000, tzinfo=UTC),
                order_id="order-10m",
                message="cancelled_zero_fill",
            )
        )

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 10, 30, tzinfo=UTC),
                tau_minutes=4.5,
                predicted_yes_probability=0.76,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        recovery_intent = await _wait_for_trade_intent(
            trade_intent_queue,
            lambda intent: intent.tranche_window == "5m",
            timeout=1.0,
        )
        assert recovery_intent.contracts == 1

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 11, 10, tzinfo=UTC),
                tau_minutes=3.8,
                predicted_yes_probability=0.76,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        awaiting_fill = await _wait_for_layering_action(layering_queue, "awaiting_initial_fill", timeout=1.0)
        assert awaiting_fill.decision_window == "4m"

        with pytest.raises(TimeoutError):
            await _wait_for_trade_intent(
                trade_intent_queue,
                lambda intent: intent.tranche_window == "4m",
                timeout=0.05,
            )

        await layering_engine._handle_execution_update(
            _execution_update_from_intent(
                recovery_intent,
                status="filled",
                event_time=datetime(2026, 1, 1, 12, 11, 15, tzinfo=UTC),
                order_id="order-5m",
                filled_contracts=1,
                fill_price_cents=recovery_intent.max_acceptable_entry_price_cents,
                message="order_terminal_fill",
            )
        )

        await scorer.publish(
            _score_state(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 11, 20, tzinfo=UTC),
                tau_minutes=3.67,
                predicted_yes_probability=0.76,
                last_yes_price_cents=30,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        )
        fourth_minute_intent = await _wait_for_trade_intent(
            trade_intent_queue,
            lambda intent: intent.tranche_window == "4m",
            timeout=1.0,
        )

        assert fourth_minute_intent.contracts > recovery_intent.contracts

        await layering_engine.stop()
        await signal_engine.stop()

    asyncio.run(run())
