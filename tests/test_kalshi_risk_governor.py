from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.indexers.kalshi.models import Market
from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiExecutionConfig,
    KalshiExecutionEngine,
    KalshiExecutionMode,
    KalshiFeatureStateEngine,
    KalshiLightGBMScorer,
    KalshiMarketDataCollector,
    KalshiSignalRiskEngine,
    KalshiTickerState,
    SessionRiskGovernor,
    SessionRiskGovernorConfig,
)
from src.live.kalshi.scorer import LoadedLightGBMModel
from src.live.kalshi.types import KalshiTickerUpdate


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
    ticker: str,
    event_time: datetime,
    close_time: datetime,
    last_yes_price_cents: int = 55,
    previous_yes_price_cents: int | None = 54,
) -> KalshiTickerUpdate:
    return KalshiTickerUpdate(
        ticker=ticker,
        event_time=event_time,
        last_yes_price_cents=last_yes_price_cents,
        previous_yes_price_cents=previous_yes_price_cents,
        close_time=close_time,
        is_open=True,
        market_prob=None,
        previous_market_prob=None,
        price_momentum=None,
        tau_minutes=None,
        yes_bid_cents=max(1, last_yes_price_cents - 1),
        yes_ask_cents=min(99, last_yes_price_cents + 1),
        no_bid_cents=max(1, 99 - last_yes_price_cents),
        no_ask_cents=min(99, 101 - last_yes_price_cents),
        ticker_update_time=event_time,
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


class _FakeRestClient:
    def __init__(self) -> None:
        self.market_results: dict[str, Market] = {}

    def close(self) -> None:
        return None

    def get_market(self, ticker: str) -> Market:
        return self.market_results[ticker]


async def _wait_for_execution_status(queue: asyncio.Queue, *, ticker: str, status: str, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for execution status={status} ticker={ticker}")
        update = await asyncio.wait_for(queue.get(), timeout=remaining)
        if update.ticker == ticker and update.status == status:
            return update


async def _wait_for_signal_update(queue: asyncio.Queue, *, ticker: str, approved: bool, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for signal update ticker={ticker} approved={approved}")
        update = await asyncio.wait_for(queue.get(), timeout=remaining)
        if update.ticker == ticker and update.approved is approved:
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


def _loss_market(ticker: str, *, close_time: datetime) -> Market:
    return Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title=ticker,
        yes_sub_title="yes",
        no_sub_title="no",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=55,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="no",
        created_time=close_time - timedelta(minutes=15),
        open_time=close_time - timedelta(minutes=14),
        close_time=close_time,
    )


def test_session_risk_governor_trips_on_consecutive_paper_losses_and_requires_manual_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.PAPER,
            reconcile_interval_seconds=0.05,
        ),
    )
    execution_engine._rest_client = _FakeRestClient()
    governor = SessionRiskGovernor(
        SessionRiskGovernorConfig(
            max_session_drawdown_dollars=15.0,
            max_session_loss_dollars=25.0,
            max_consecutive_losses=2,
            max_positions_opened_per_hour=99,
        ),
        environment=KalshiEnvironment.DEMO.value,
    )
    governor.attach(signal_engine=signal_engine, execution_engine=execution_engine)

    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()
    alerts: list[dict[str, object]] = []
    governor.subscribe_alerts(lambda payload: alerts.append(payload))

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()

        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        tickers = [
            "KXBTC15M-LOSS-1",
            "KXBTC15M-LOSS-2",
            "KXBTC15M-BLOCKED-3",
            "KXBTC15M-RESUMED-4",
        ]
        for index, ticker in enumerate(tickers):
            close_time = base_time + timedelta(minutes=(index * 2) + 7)
            execution_engine._rest_client.market_results[ticker] = _loss_market(ticker, close_time=close_time)  # type: ignore[attr-defined]

        for index, ticker in enumerate(tickers[:2]):
            event_time = base_time + timedelta(minutes=index * 2)
            close_time = event_time + timedelta(minutes=7)
            await collector._publish_update(_ticker_update(ticker=ticker, event_time=event_time, close_time=close_time))
            approved = await _wait_for_signal_update(signal_queue, ticker=ticker, approved=True)
            assert approved.trade_intent is not None
            await _wait_for_execution_status(execution_queue, ticker=ticker, status="filled")
            collector._states[ticker] = KalshiTickerState(
                ticker=ticker,
                last_yes_price_cents=55,
                last_trade_time=event_time,
                previous_yes_price_cents=54,
                close_time=close_time,
                is_open=False,
                yes_bid_cents=54,
                yes_ask_cents=56,
                no_bid_cents=44,
                no_ask_cents=46,
                ticker_update_time=event_time,
            )
            settled = await _wait_for_execution_status(execution_queue, ticker=ticker, status="settled")
            assert settled.realized_pnl_dollars is not None and settled.realized_pnl_dollars < 0

        await _wait_for_condition(lambda: governor.get_state().halted, timeout=2.0)
        halted_state = governor.get_state()
        assert halted_state.halt_reason == "max_consecutive_losses"
        assert halted_state.consecutive_losses == 2
        halt_record = governor.get_last_halt_record()
        assert halt_record is not None
        assert halt_record.reason == "max_consecutive_losses"
        assert alerts and alerts[-1]["reason"] == "max_consecutive_losses"

        blocked_ticker = tickers[2]
        blocked_time = base_time + timedelta(minutes=10)
        blocked_close = blocked_time + timedelta(minutes=7)
        await collector._publish_update(
            _ticker_update(ticker=blocked_ticker, event_time=blocked_time, close_time=blocked_close)
        )
        blocked = await _wait_for_signal_update(signal_queue, ticker=blocked_ticker, approved=False)
        assert blocked.block_reason == SessionRiskGovernor.BLOCK_REASON
        manual = signal_engine.reserve_manual_trade_intent(
            decision_state=blocked,
            side="YES",
            entry_price_cents=55,
            allow_unapproved_retry=True,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            ignore_post_cost_edge_threshold=True,
        )
        assert manual is None

        await governor.manual_resume(reason="operator_override")
        await _wait_for_condition(lambda: not governor.get_state().halted, timeout=1.0)

        resumed_ticker = tickers[3]
        resumed_time = base_time + timedelta(minutes=12)
        resumed_close = resumed_time + timedelta(minutes=7)
        await collector._publish_update(
            _ticker_update(ticker=resumed_ticker, event_time=resumed_time, close_time=resumed_close)
        )
        resumed = await _wait_for_signal_update(signal_queue, ticker=resumed_ticker, approved=True)
        assert resumed.trade_intent is not None

        await governor.close()
        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())
