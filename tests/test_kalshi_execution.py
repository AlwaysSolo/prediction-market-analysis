from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
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
    build_create_order_payload,
    parse_fill_event,
    parse_market_position_event,
)
from src.live.kalshi.scorer import LoadedLightGBMModel
from src.live.kalshi.signal_risk import KalshiSignalRiskConfig
from src.live.kalshi.types import KalshiTickerUpdate


class _QueueTradeIntentSource:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.callbacks = []
        self.subscribe_queue_calls = 0

    def subscribe_trade_intent_queue(self, maxsize: int = 0) -> asyncio.Queue:
        del maxsize
        self.subscribe_queue_calls += 1
        return self.queue

    def subscribe_trade_intent_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def unsubscribe_trade_intent_callback(self, callback) -> None:
        try:
            self.callbacks.remove(callback)
        except ValueError:
            pass

    async def publish(self, intent) -> None:
        for callback in list(self.callbacks):
            result = callback(intent)
            if inspect.isawaitable(result):
                await result


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
    yes_bid_size: int | None = None,
    yes_ask_size: int | None = None,
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
        yes_ask_cents = max(yes_bid_cents + 1, last_yes_price_cents)
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
        yes_bid_size=yes_bid_size,
        yes_ask_size=yes_ask_size,
        no_bid_size=yes_ask_size,
        no_ask_size=yes_bid_size,
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


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.content = b"{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _RecordingHttpClient:
    def __init__(self):
        self.calls: list[dict] = []

    def request(self, method: str, path: str, params=None, json=None, headers=None):
        self.calls.append(
            {"method": method, "path": path, "params": params, "json": json, "headers": headers}
        )
        return _FakeResponse({"ok": True})

    def close(self) -> None:
        return None


class _FakeRestClient:
    def __init__(self, *, create_order_results: list | None = None):
        self.create_order_results = list(create_order_results or [])
        self.create_order_calls: list[dict] = []
        self.orders: dict[str, dict] = {}
        self.orderbook_results: dict[str, dict] = {}
        self.balance_cents = 1000
        self.positions: list[dict] = []
        self.market_results: dict[str, Market] = {}
        self.balance_calls = 0
        self.balance_subaccounts: list[int] = []
        self.positions_calls = 0
        self.positions_subaccounts: list[int] = []
        self.get_orders_calls = 0
        self.get_orders_subaccounts: list[int | None] = []
        self.get_order_calls = 0
        self.get_order_subaccounts: list[int | None] = []
        self.get_market_calls = 0
        self.get_market_orderbook_calls: list[dict[str, object]] = []

    def close(self) -> None:
        return None

    def create_order(self, payload: dict) -> dict:
        self.create_order_calls.append(payload)
        if self.create_order_results:
            result = self.create_order_results.pop(0)
        else:
            result = {"order": self._build_order(payload, order_id="server-order", status="executed", fill_count=1)}
        if isinstance(result, Exception):
            raise result
        order = result.get("order")
        if isinstance(order, dict):
            self.orders[order["order_id"]] = order
            if order.get("status") in {"executed", "canceled"} and int(order.get("fill_count", 0)) > 0:
                self.positions = [self._build_position(order)]
        return result

    def get_balance(self, *, subaccount: int = 0) -> dict:
        self.balance_calls += 1
        self.balance_subaccounts.append(subaccount)
        return {"balance": self.balance_cents, "portfolio_value": self.balance_cents, "updated_ts": 1767225600}

    def get_positions(self, *, subaccount: int = 0) -> list[dict]:
        self.positions_calls += 1
        self.positions_subaccounts.append(subaccount)
        return list(self.positions)

    def get_orders(
        self,
        *,
        ticker: str | None = None,
        min_ts: int | None = None,
        subaccount: int | None = None,
        **_: object,
    ) -> list[dict]:
        self.get_orders_calls += 1
        self.get_orders_subaccounts.append(subaccount)
        orders = list(self.orders.values())
        if ticker is not None:
            orders = [order for order in orders if order["ticker"] == ticker]
        return orders

    def get_order(self, order_id: str, *, subaccount: int | None = None) -> dict:
        self.get_order_calls += 1
        self.get_order_subaccounts.append(subaccount)
        return {"order": self.orders[order_id]}

    def get_market(self, ticker: str) -> Market:
        self.get_market_calls += 1
        return self.market_results[ticker]

    def get_market_orderbook(self, ticker: str, *, depth: int = 1) -> dict:
        self.get_market_orderbook_calls.append({"ticker": ticker, "depth": depth})
        if ticker in self.orderbook_results:
            return self.orderbook_results[ticker]
        return {
            "orderbook_fp": {
                "yes_dollars": [["0.4500", "10.00"]],
                "no_dollars": [["0.4500", "10.00"]],
            }
        }

    def cancel_order(self, order_id: str) -> dict:
        order = self.orders[order_id]
        order["status"] = "canceled"
        order["remaining_count"] = 0
        return {"order": order, "reduced_by": 0}

    def _build_order(
        self,
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

    def _build_position(self, order: dict) -> dict:
        sign = "" if order["side"] == "yes" else "-"
        price_dollars = f"{order['yes_price'] / 100:.4f}"
        return {
            "ticker": order["ticker"],
            "position_fp": f"{sign}{order['fill_count']}.00",
            "market_exposure_dollars": price_dollars,
            "fees_paid_dollars": "0.0100",
        }


async def _noop_private_ws(self) -> None:
    await self._stop_event.wait()


async def _wait_for_status(queue: asyncio.Queue, status: str, *, timeout: float = 1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for status={status}")
        update = await asyncio.wait_for(queue.get(), timeout=remaining)
        if update.status == status:
            return update


async def _wait_for_condition(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if predicate():
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for condition")
        await asyncio.sleep(min(0.01, remaining))


class _CapturingAsyncLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def write(self, event_type: str, payload: dict[str, object], event_time=None) -> None:
        self.events.append({"event_type": event_type, "payload": payload, "event_time": event_time})


def test_execution_config_from_env_uses_demo_override_and_shared_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KALSHI_EXECUTION_MODE", "paper")
    monkeypatch.setenv("KALSHI_EXECUTION_RECONCILE_INTERVAL_SECONDS", "15")
    monkeypatch.setenv("KALSHI_DEMO_EXECUTION_MODE", "live")
    monkeypatch.setenv("KALSHI_DEMO_EXECUTION_ENABLE_LIVE_TRADING", "true")

    demo_config = KalshiExecutionConfig.from_env(KalshiEnvironment.DEMO)
    prod_config = KalshiExecutionConfig.from_env(KalshiEnvironment.PRODUCTION)

    assert demo_config.mode is KalshiExecutionMode.LIVE
    assert demo_config.enable_live_trading is True
    assert prod_config.mode is KalshiExecutionMode.PAPER
    assert prod_config.reconcile_interval_seconds == 15.0


def test_build_create_order_payload_uses_side_specific_price_field():
    yes_payload = build_create_order_payload(
        type(
            "Intent",
            (),
            {
                "ticker": "TEST",
                "decision_id": "id-1",
                "side": "YES",
                "contracts": 1,
                "max_acceptable_entry_price_cents": 58,
            },
        )(),
        subaccount=7,
    )
    no_payload = build_create_order_payload(
        type(
            "Intent",
            (),
            {
                "ticker": "TEST",
                "decision_id": "id-2",
                "side": "NO",
                "contracts": 1,
                "max_acceptable_entry_price_cents": 41,
            },
        )(),
        subaccount=7,
    )

    assert yes_payload["yes_price"] == 58
    assert "no_price" not in yes_payload
    assert yes_payload["subaccount"] == 7
    assert yes_payload["time_in_force"] == "immediate_or_cancel"
    assert no_payload["no_price"] == 41
    assert "yes_price" not in no_payload
    assert no_payload["subaccount"] == 7

    probe_payload = build_create_order_payload(
        type(
            "Intent",
            (),
            {
                "ticker": "TEST",
                "decision_id": "id-3",
                "side": "YES",
                "contracts": 1,
                "max_acceptable_entry_price_cents": 58,
            },
        )(),
        subaccount=7,
        time_in_force="good_till_canceled",
        expiration_ts=1767225602,
    )
    assert probe_payload["time_in_force"] == "good_till_canceled"
    assert probe_payload["expiration_ts"] == 1767225602


def test_live_rest_client_create_order_uses_post_and_auth_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.live.kalshi.client import KalshiLiveRestClient

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.live.kalshi.client.build_auth_headers",
        lambda credentials, method, path, timestamp_ms: captured.update(
            {"credentials": credentials, "method": method, "path": path, "timestamp_ms": timestamp_ms}
        )
        or {"KALSHI-ACCESS-KEY": "demo-key", "KALSHI-ACCESS-SIGNATURE": "sig"},
    )
    client = KalshiLiveRestClient(_collector_config(tmp_path))
    recording_client = _RecordingHttpClient()
    client.client = recording_client  # type: ignore[assignment]

    client.create_order(
        {"ticker": "TEST", "client_order_id": "abc", "side": "yes", "action": "buy", "count": 1},
        subaccount=7,
    )

    call = recording_client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/portfolio/orders"
    assert call["json"]["client_order_id"] == "abc"
    assert call["json"]["subaccount"] == 7
    assert call["headers"]["KALSHI-ACCESS-KEY"] == "demo-key"
    assert call["headers"]["Content-Type"] == "application/json"
    assert captured["method"] == "POST"
    assert captured["path"] == "/trade-api/v2/portfolio/orders"


def test_live_rest_client_get_market_orderbook_uses_get_and_auth_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.live.kalshi.client import KalshiLiveRestClient

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.live.kalshi.client.build_auth_headers",
        lambda credentials, method, path, timestamp_ms: captured.update(
            {"credentials": credentials, "method": method, "path": path, "timestamp_ms": timestamp_ms}
        )
        or {"KALSHI-ACCESS-KEY": "demo-key", "KALSHI-ACCESS-SIGNATURE": "sig"},
    )
    client = KalshiLiveRestClient(_collector_config(tmp_path))
    recording_client = _RecordingHttpClient()
    client.client = recording_client  # type: ignore[assignment]

    client.get_market_orderbook("TEST", depth=3)

    call = recording_client.calls[0]
    assert call["method"] == "GET"
    assert call["path"] == "/markets/TEST/orderbook"
    assert call["params"] == {"depth": 3}
    assert call["headers"]["KALSHI-ACCESS-KEY"] == "demo-key"
    assert captured["method"] == "GET"
    assert captured["path"] == "/trade-api/v2/markets/TEST/orderbook"


def test_execution_private_subscription_messages():
    collector = KalshiMarketDataCollector(_collector_config(Path(".")))
    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    engine = KalshiExecutionEngine(signal_engine)

    assert engine._private_subscription_messages() == [
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["user_orders", "fill", "market_positions"]},
        }
    ]


def test_parse_fill_and_market_position_events():
    fill = parse_fill_event(
        {
            "order_id": "order-1",
            "client_order_id": "decision-1",
            "market_ticker": "KXBTC15M-TEST",
            "side": "yes",
            "yes_price_dollars": "0.5500",
            "count_fp": "1.00",
            "fee_cost": "0.0100",
            "ts": 1767225600,
        }
    )
    position = parse_market_position_event(
        {
            "market_ticker": "KXBTC15M-TEST",
            "position_fp": "-2.00",
            "position_cost_dollars": "0.8000",
            "fees_paid_dollars": "0.0200",
        }
    )

    assert fill.price_cents == 55
    assert fill.count == 1
    assert position.side == "NO"
    assert position.contracts == 2


def test_simulated_live_fill_settles_to_market_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=False,
            simulate_immediate_fills=True,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    ticker = "KXBTC15M-TEST"
    fake_rest.market_results[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=55,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="yes",
        created_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
    )
    execution_engine._rest_client = fake_rest
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(
            _ticker_update(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            )
        )

        filled = await _wait_for_status(queue, "filled", timeout=2.5)
        assert filled.fill_price_cents == 55

        collector._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=False,
        )

        settled = await _wait_for_status(queue, "settled", timeout=1.5)
        assert settled.settlement_result == "YES"
        assert settled.realized_pnl_dollars == pytest.approx(0.4245)
        assert settled.cumulative_realized_pnl_dollars == pytest.approx(0.4245)
        assert execution_engine.get_portfolio_snapshot() is not None
        assert execution_engine.get_portfolio_snapshot().available_cash_dollars == pytest.approx(10.4245)

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_simulated_live_fill_settles_when_close_time_passed_even_if_collector_state_is_stale_open(
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
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=False,
            simulate_immediate_fills=True,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    ticker = "KXBTC15M-TEST"
    close_time = datetime(2026, 1, 1, 12, 7, tzinfo=UTC)
    fake_rest.market_results[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=55,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="yes",
        created_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=close_time,
    )
    execution_engine._rest_client = fake_rest
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(
            _ticker_update(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                close_time=close_time,
            )
        )

        await _wait_for_status(queue, "filled")

        collector._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 11, 59, tzinfo=UTC),
            is_open=True,
        )

        settled = await _wait_for_status(queue, "settled", timeout=1.5)
        assert settled.settlement_result == "YES"
        assert fake_rest.get_market_calls >= 1

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_paper_mode_full_fill_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer, KalshiSignalRiskConfig())
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
    )
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        filled = await _wait_for_status(queue, "filled", timeout=5.0)
        assert filled.side == "YES"
        assert filled.filled_contracts == 1
        assert filled.fill_price_cents == 55
        assert signal_engine.get_portfolio_state().open_positions[0].ticker == "KXBTC15M-TEST"
        snapshot = execution_engine.get_portfolio_snapshot()
        assert snapshot is not None
        assert snapshot.available_cash_dollars == pytest.approx(filled.available_cash_dollars)
        assert snapshot.open_positions[0].ticker == "KXBTC15M-TEST"

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_submit_success_updates_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        await asyncio.sleep(2.0)
        updates: list = []
        while not queue.empty():
            updates.append(await queue.get())
        filled = next((update for update in updates if update.status == "filled"), None)
        assert filled is not None
        assert filled.order_id == "server-order"
        await asyncio.sleep(0.1)
        snapshot = execution_engine.get_portfolio_snapshot()
        assert snapshot is not None
        assert snapshot.open_positions[0].ticker == "KXBTC15M-TEST"
        assert fake_rest.positions_calls == 1
        assert fake_rest.get_order_calls == 0
        assert fake_rest.get_orders_calls == 1
        assert fake_rest.create_order_calls[0]["subaccount"] == 7
        assert fake_rest.balance_subaccounts
        assert set(fake_rest.balance_subaccounts) == {7}
        assert fake_rest.positions_subaccounts == [7]
        assert fake_rest.get_orders_subaccounts == [7]

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_cancels_when_orderbook_moves_beyond_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    ticker = "KXBTC15M-TEST"
    fake_rest.orderbook_results[ticker] = {
        "orderbook_fp": {
            "yes_dollars": [["0.0800", "5.00"]],
            "no_dollars": [["0.9000", "5.00"]],
        }
    }
    execution_engine._rest_client = fake_rest
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker=ticker))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=36,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=35,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=36,
            yes_ask_cents=37,
            no_bid_cents=63,
            no_ask_cents=64,
            yes_bid_size=3,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=90)
        await trade_intent_source.queue.put(manual_intent)

        cancelled = await _wait_for_status(execution_queue, "cancelled", timeout=2.0)
        assert cancelled.message == "live_orderbook_limit_moved_away"
        assert not fake_rest.create_order_calls
        assert fake_rest.get_market_orderbook_calls == [{"ticker": ticker, "depth": 1}]

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_cancels_when_top_of_book_size_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=2,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    ticker = "KXBTC15M-TEST"
    fake_rest.orderbook_results[ticker] = {
        "orderbook_fp": {
            "yes_dollars": [["0.3600", "1.00"]],
            "no_dollars": [["0.6300", "8.00"]],
        }
    }
    execution_engine._rest_client = fake_rest
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker=ticker))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=36,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=35,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=36,
            yes_ask_cents=37,
            no_bid_cents=63,
            no_ask_cents=64,
            yes_bid_size=3,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=2,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=90)
        await trade_intent_source.queue.put(manual_intent)

        cancelled = await _wait_for_status(execution_queue, "cancelled", timeout=2.0)
        assert cancelled.message == "live_orderbook_insufficient_size"
        assert not fake_rest.create_order_calls
        assert fake_rest.get_market_orderbook_calls == [{"ticker": ticker, "depth": 1}]

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_submit_logs_pre_submit_timing_and_top_of_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    ticker = "KXBTC15M-TEST"
    fake_rest.orderbook_results[ticker] = {
        "orderbook_fp": {
            "yes_dollars": [["0.3600", "4.00"]],
            "no_dollars": [["0.6300", "8.00"]],
        }
    }
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker=ticker))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=36,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=35,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=36,
            yes_ask_cents=37,
            no_bid_cents=63,
            no_ask_cents=64,
            yes_bid_size=3,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=90)
        await trade_intent_source.queue.put(manual_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"

        pre_submit_event = next(event for event in logger.events if event["event_type"] == "pre_submit_orderbook_check")
        pre_submit_payload = pre_submit_event["payload"]
        assert pre_submit_payload["top_book_side"] == "yes_bid"
        assert pre_submit_payload["current_executable_ask_cents"] == 64
        assert pre_submit_payload["top_book_contracts"] == 4
        assert pre_submit_payload["feed_top_book_side"] == "yes_bid"
        assert pre_submit_payload["feed_top_book_price_cents"] == 36
        assert pre_submit_payload["feed_top_book_contracts"] == 3
        assert pre_submit_payload["feed_executable_ask_cents"] == 64
        assert pre_submit_payload["ticker_update_to_check_ms"] is not None
        assert pre_submit_payload["received_at_to_check_ms"] is not None

        submit_event = next(event for event in logger.events if event["event_type"] == "submit_requested")
        submit_payload = submit_event["payload"]
        assert submit_payload["current_executable_ask_cents"] == 64
        assert submit_payload["top_book_contracts"] == 4
        assert submit_payload["feed_top_book_contracts"] == 3
        assert submit_payload["time_in_force"] == "immediate_or_cancel"
        assert submit_payload["expiration_ts"] is None
        assert submit_payload["ticker_update_to_submit_requested_ms"] is not None
        assert submit_payload["received_at_to_submit_requested_ms"] is not None
        assert submit_payload["pre_submit_orderbook_roundtrip_ms"] is not None
        assert fake_rest.get_market_orderbook_calls == [{"ticker": ticker, "depth": 1}]

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_immediate_feed_check_bypasses_rest_orderbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            skip_rest_orderbook_check_for_immediate_orders=True,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST"))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=36,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=35,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=36,
            yes_ask_cents=37,
            no_bid_cents=63,
            no_ask_cents=64,
            yes_bid_size=3,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=90)
        await trade_intent_source.queue.put(manual_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"

        pre_submit_event = next(event for event in logger.events if event["event_type"] == "pre_submit_orderbook_check")
        pre_submit_payload = pre_submit_event["payload"]
        assert pre_submit_payload["check_source"] == "feed_quote"
        assert pre_submit_payload["top_book_side"] == "yes_bid"
        assert pre_submit_payload["top_book_contracts"] == 3
        assert pre_submit_payload["current_executable_ask_cents"] == 64

        submit_event = next(event for event in logger.events if event["event_type"] == "submit_requested")
        submit_payload = submit_event["payload"]
        assert submit_payload["pre_submit_check_source"] == "feed_quote"
        assert submit_payload["top_book_contracts"] == 3
        assert submit_payload["current_executable_ask_cents"] == 64
        assert fake_rest.get_market_orderbook_calls == []

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_immediate_feed_check_cancels_before_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            skip_rest_orderbook_check_for_immediate_orders=True,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST"))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=9,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=10,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=8,
            yes_ask_cents=10,
            no_bid_cents=90,
            no_ask_cents=92,
            yes_bid_size=3,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=90)
        await trade_intent_source.queue.put(manual_intent)

        cancelled = await _wait_for_status(execution_queue, "cancelled", timeout=2.0)
        assert cancelled.message == "feed_limit_moved_away"
        assert not fake_rest.create_order_calls
        assert fake_rest.get_market_orderbook_calls == []

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_no_probe_ioc_cushion_increases_submitted_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            skip_rest_orderbook_check_for_immediate_orders=True,
            probe_order_time_in_force="immediate_or_cancel",
            no_probe_immediate_limit_cushion_cents=3,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST"))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=34,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=35,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=33,
            yes_ask_cents=35,
            no_bid_cents=65,
            no_ask_cents=67,
            yes_bid_size=4,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=4,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        probe_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-no-cushion",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert probe_intent is not None
        probe_intent = replace(probe_intent, max_acceptable_entry_price_cents=64)
        await trade_intent_source.queue.put(probe_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        assert filled.limit_price_cents == 67
        assert filled.cash_required_dollars == pytest.approx(0.56)
        assert fake_rest.get_market_orderbook_calls == []
        assert fake_rest.create_order_calls[0]["no_price"] == 67
        assert signal_engine.get_portfolio_state().open_positions[0].cash_required_dollars == pytest.approx(0.56)

        pre_submit_event = next(event for event in logger.events if event["event_type"] == "pre_submit_orderbook_check")
        pre_submit_payload = pre_submit_event["payload"]
        assert pre_submit_payload["limit_price_cents"] == 67
        assert pre_submit_payload["model_limit_price_cents"] == 64
        assert pre_submit_payload["execution_limit_adjustment_cents"] == 3

        submit_event = next(event for event in logger.events if event["event_type"] == "submit_requested")
        submit_payload = submit_event["payload"]
        assert submit_payload["limit_price_cents"] == 67
        assert submit_payload["model_limit_price_cents"] == 64
        assert submit_payload["execution_limit_adjustment_cents"] == 3

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_yes_probe_ioc_cushion_defaults_to_no_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            skip_rest_orderbook_check_for_immediate_orders=True,
            probe_order_time_in_force="immediate_or_cancel",
            no_probe_immediate_limit_cushion_cents=3,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST"))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=54,
            yes_ask_cents=56,
            no_bid_cents=44,
            no_ask_cents=46,
            yes_bid_size=3,
            yes_ask_size=5,
            no_bid_size=5,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        probe_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=56,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-yes-nochange",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert probe_intent is not None
        probe_intent = replace(probe_intent, max_acceptable_entry_price_cents=56)
        await trade_intent_source.queue.put(probe_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        assert filled.limit_price_cents == 56
        assert fake_rest.create_order_calls[0]["yes_price"] == 56

        submit_event = next(event for event in logger.events if event["event_type"] == "submit_requested")
        submit_payload = submit_event["payload"]
        assert submit_payload["limit_price_cents"] == 56
        assert submit_payload["model_limit_price_cents"] == 56
        assert submit_payload["execution_limit_adjustment_cents"] == 0

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_yes_probe_ioc_cushion_increases_submitted_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            skip_rest_orderbook_check_for_immediate_orders=True,
            probe_order_time_in_force="immediate_or_cancel",
            yes_probe_immediate_limit_cushion_cents=2,
        ),
        trade_intent_source=trade_intent_source,
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST"))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=54,
            yes_ask_cents=56,
            no_bid_cents=44,
            no_ask_cents=46,
            yes_bid_size=3,
            yes_ask_size=5,
            no_bid_size=5,
            no_ask_size=3,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )

        probe_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=56,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-yes-cushion",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert probe_intent is not None
        probe_intent = replace(probe_intent, max_acceptable_entry_price_cents=56)
        await trade_intent_source.queue.put(probe_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        assert filled.limit_price_cents == 58
        assert filled.cash_required_dollars == pytest.approx(0.56)
        assert fake_rest.create_order_calls[0]["yes_price"] == 58
        assert signal_engine.get_portfolio_state().open_positions[0].cash_required_dollars == pytest.approx(0.56)

        pre_submit_event = next(event for event in logger.events if event["event_type"] == "pre_submit_orderbook_check")
        pre_submit_payload = pre_submit_event["payload"]
        assert pre_submit_payload["limit_price_cents"] == 58
        assert pre_submit_payload["model_limit_price_cents"] == 56
        assert pre_submit_payload["execution_limit_adjustment_cents"] == 2

        submit_event = next(event for event in logger.events if event["event_type"] == "submit_requested")
        submit_payload = submit_event["payload"]
        assert submit_payload["limit_price_cents"] == 58
        assert submit_payload["model_limit_price_cents"] == 56
        assert submit_payload["execution_limit_adjustment_cents"] == 2

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_probe_orders_use_short_ttl_gtc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            probe_order_time_in_force="good_till_canceled",
            probe_order_ttl_seconds=2.0,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST", yes_bid_size=2, yes_ask_size=3))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=54,
            yes_ask_cents=56,
            no_bid_cents=44,
            no_ask_cents=46,
            yes_bid_size=2,
            yes_ask_size=3,
            no_bid_size=3,
            no_ask_size=2,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 200000, tzinfo=UTC),
        )
        fake_rest.orderbook_results["KXBTC15M-TEST"] = {
            "orderbook_fp": {
                "yes_dollars": [["0.5400", "2.00"]],
                "no_dollars": [["0.4400", "3.00"]],
            }
        }

        probe_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=56,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-1",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert probe_intent is not None
        probe_intent = replace(probe_intent, max_acceptable_entry_price_cents=56)
        await execution_engine._submit_live_intent(probe_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        submit_event = next(event for event in logger.events if event["event_type"] == "submit_requested")
        submit_payload = submit_event["payload"]
        assert submit_payload["is_probe_order"] is True
        assert submit_payload["time_in_force"] == "good_till_canceled"
        assert isinstance(submit_payload["expiration_ts"], int)
        assert submit_payload["expiration_ts"] > int(datetime(2026, 1, 1, 12, 0, tzinfo=UTC).timestamp())
        assert fake_rest.create_order_calls[0]["time_in_force"] == "good_till_canceled"
        assert "expiration_ts" in fake_rest.create_order_calls[0]

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_probe_gtc_submits_even_when_book_has_moved_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            probe_order_time_in_force="good_till_canceled",
            probe_order_ttl_seconds=2.0,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST", yes_bid_size=2, yes_ask_size=3))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=54,
            yes_ask_cents=56,
            no_bid_cents=44,
            no_ask_cents=46,
            yes_bid_size=2,
            yes_ask_size=3,
            no_bid_size=3,
            no_ask_size=2,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 200000, tzinfo=UTC),
        )
        fake_rest.orderbook_results["KXBTC15M-TEST"] = {
            "orderbook_fp": {
                "yes_dollars": [["0.5400", "2.00"]],
                "no_dollars": [["0.3900", "3.00"]],
            }
        }

        probe_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=56,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-1",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert probe_intent is not None
        probe_intent = replace(probe_intent, max_acceptable_entry_price_cents=56)
        await execution_engine._submit_live_intent(probe_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        pre_submit_event = next(event for event in logger.events if event["event_type"] == "pre_submit_orderbook_check")
        pre_submit_payload = pre_submit_event["payload"]
        assert pre_submit_payload["requires_immediate_match"] is False
        assert pre_submit_payload["passed"] is False
        assert pre_submit_payload["reason"] == "live_orderbook_limit_moved_away"
        assert fake_rest.create_order_calls[0]["time_in_force"] == "good_till_canceled"

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_probe_gtc_submits_even_when_top_of_book_size_is_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.20))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=2,
        ),
    )
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            probe_order_time_in_force="good_till_canceled",
            probe_order_ttl_seconds=2.0,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    logger = _CapturingAsyncLogger()
    execution_engine._logger = logger  # type: ignore[assignment]
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update(ticker="KXBTC15M-TEST", yes_bid_size=2, yes_ask_size=3))

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        collector._states["KXBTC15M-TEST"] = KalshiTickerState(
            ticker="KXBTC15M-TEST",
            last_yes_price_cents=36,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=35,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=True,
            yes_bid_cents=36,
            yes_ask_cents=37,
            no_bid_cents=63,
            no_ask_cents=64,
            yes_bid_size=1,
            yes_ask_size=6,
            no_bid_size=6,
            no_ask_size=1,
            ticker_update_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 12, 0, 0, 150000, tzinfo=UTC),
        )
        fake_rest.orderbook_results["KXBTC15M-TEST"] = {
            "orderbook_fp": {
                "yes_dollars": [["0.3600", "1.00"]],
                "no_dollars": [["0.6300", "8.00"]],
            }
        }

        probe_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="NO",
            entry_price_cents=64,
            contracts=2,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-2",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert probe_intent is not None
        probe_intent = replace(probe_intent, max_acceptable_entry_price_cents=90)
        await execution_engine._submit_live_intent(probe_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        pre_submit_event = next(event for event in logger.events if event["event_type"] == "pre_submit_orderbook_check")
        pre_submit_payload = pre_submit_event["payload"]
        assert pre_submit_payload["requires_immediate_match"] is False
        assert pre_submit_payload["passed"] is False
        assert pre_submit_payload["reason"] == "live_orderbook_insufficient_size"
        assert fake_rest.create_order_calls[0]["time_in_force"] == "good_till_canceled"

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_ambiguous_submit_reconciles_by_client_order_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    request = httpx.Request("POST", "https://demo-api.kalshi.co/trade-api/v2/portfolio/orders")
    connect_error = httpx.ConnectError("network down", request=request)
    fake_rest = _FakeRestClient(create_order_results=[connect_error])

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            order_reconcile_timeout_seconds=0.2,
        ),
    )
    execution_engine._rest_client = fake_rest
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()

        # Seed the recovered order on the fake REST side after the claim happens.
        async def publish_and_seed() -> None:
            await collector._publish_update(_ticker_update())
            await asyncio.sleep(0.05)
            claimed = execution_engine.snapshot_states()
            assert claimed
            decision_id = next(iter(claimed))
            fake_rest.orders["recovered-order"] = fake_rest._build_order(
                {
                    "ticker": "KXBTC15M-TEST",
                    "client_order_id": decision_id,
                    "side": "yes",
                    "action": "buy",
                    "type": "limit",
                    "count": 1,
                    "yes_price": 55,
                },
                order_id="recovered-order",
                status="executed",
                fill_count=1,
            )
            fake_rest.positions = [fake_rest._build_position(fake_rest.orders["recovered-order"])]

        await publish_and_seed()
        await asyncio.sleep(2.0)
        updates: list = []
        while not queue.empty():
            updates.append(await queue.get())
        filled = next((update for update in updates if update.status == "filled"), None)
        assert filled is not None
        assert filled.order_id == "recovered-order"
        assert len(fake_rest.create_order_calls) == 1
        assert fake_rest.create_order_calls[0]["subaccount"] == 7
        assert 7 in fake_rest.get_orders_subaccounts

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_shadow_mode_releases_reservation_without_submitting_order(
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
            mode=KalshiExecutionMode.SHADOW,
            subaccount=7,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    execution_engine._rest_client = fake_rest
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        await asyncio.sleep(1.0)
        updates: list = []
        while not queue.empty():
            updates.append(await queue.get())
        filled = next((update for update in updates if update.status == "filled"), None)
        assert filled is not None
        assert filled.message == "shadow_fill"
        assert not fake_rest.create_order_calls
        assert fake_rest.balance_subaccounts
        assert set(fake_rest.balance_subaccounts) == {7}
        assert fake_rest.positions_subaccounts == [7]

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_shadow_mode_requotes_fill_price_before_simulated_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.SHADOW,
            shadow_fill_latency_seconds=0.35,
            reconcile_interval_seconds=0.05,
        ),
        trade_intent_source=trade_intent_source,
    )
    execution_engine._rest_client = _FakeRestClient()
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=55,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=56)
        await trade_intent_source.queue.put(manual_intent)

        accepted = await _wait_for_status(execution_queue, "accepted", timeout=1.0)
        assert accepted.message == "shadow_submit_accepted"

        await asyncio.sleep(0.02)
        await collector._publish_update(
            _ticker_update(
                event_time=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
                last_yes_price_cents=56,
                previous_yes_price_cents=55,
                yes_bid_cents=55,
                yes_ask_cents=56,
                ticker_update_time=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            )
        )
        await _wait_for_condition(
            lambda: (
                scorer.get_state("KXBTC15M-TEST") is not None
                and scorer.get_state("KXBTC15M-TEST").buy_yes_price_cents == 56
            ),
            timeout=1.0,
        )

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.fill_price_cents == 56
        assert filled.message == "shadow_fill_requoted"
        assert filled.cash_required_dollars == pytest.approx(0.58)

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_shadow_mode_cancels_when_quote_moves_beyond_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.SHADOW,
            shadow_fill_latency_seconds=0.35,
            reconcile_interval_seconds=0.05,
        ),
        trade_intent_source=trade_intent_source,
    )
    execution_engine._rest_client = _FakeRestClient()
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=55,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
        )
        assert manual_intent is not None
        manual_intent = replace(manual_intent, max_acceptable_entry_price_cents=55)
        await trade_intent_source.queue.put(manual_intent)

        accepted = await _wait_for_status(execution_queue, "accepted", timeout=1.0)
        assert accepted.message == "shadow_submit_accepted"

        await asyncio.sleep(0.02)
        await collector._publish_update(
            _ticker_update(
                event_time=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
                last_yes_price_cents=57,
                previous_yes_price_cents=55,
                yes_bid_cents=56,
                yes_ask_cents=57,
                ticker_update_time=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            )
        )
        await _wait_for_condition(
            lambda: (
                scorer.get_state("KXBTC15M-TEST") is not None
                and scorer.get_state("KXBTC15M-TEST").buy_yes_price_cents == 57
            ),
            timeout=1.0,
        )

        cancelled = await _wait_for_status(execution_queue, "cancelled", timeout=2.0)
        assert cancelled.fill_price_cents is None
        assert cancelled.message == "shadow_cancelled_limit_moved_away"

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_fill_emits_settled_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(scorer)
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
        ),
    )
    fake_rest = _FakeRestClient()
    ticker = "KXBTC15M-TEST"
    fake_rest.market_results[ticker] = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=55,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="yes",
        created_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
    )
    execution_engine._rest_client = fake_rest
    queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(
            _ticker_update(
                ticker=ticker,
                event_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            )
        )

        await _wait_for_status(queue, "filled", timeout=1.5)
        collector._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=55,
            last_trade_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            previous_yes_price_cents=54,
            close_time=datetime(2026, 1, 1, 12, 7, tzinfo=UTC),
            is_open=False,
        )
        settled = await _wait_for_status(queue, "settled", timeout=1.5)
        assert settled.settlement_result == "YES"
        assert settled.realized_pnl_dollars == pytest.approx(0.44)
        assert settled.cumulative_realized_pnl_dollars == pytest.approx(0.44)
        assert fake_rest.get_market_calls >= 1

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_consumes_manual_trade_intent_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=5,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
        trade_intent_source=trade_intent_source,
    )
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True
        assert approved.trade_intent is None

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
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
        assert manual_intent is not None
        await trade_intent_source.queue.put(manual_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.thesis_id == "thesis-1"
        assert filled.tranche_window == "10m"
        assert filled.tranche_reason == "opened"
        assert filled.lifecycle_state == "probe_pending"
        assert filled.filled_contracts == 2

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_live_direct_trade_intent_handoff_skips_source_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiExecutionEngine, "_private_ws_loop", _noop_private_ws)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=5,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(
            mode=KalshiExecutionMode.LIVE,
            enable_live_trading=True,
            subaccount=7,
            reconcile_interval_seconds=0.05,
            skip_rest_orderbook_check_for_immediate_orders=True,
            enable_direct_trade_intent_handoff_in_live_mode=True,
        ),
        trade_intent_source=trade_intent_source,
    )
    execution_engine._rest_client = _FakeRestClient()
    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        assert execution_engine._consume_task is None
        assert trade_intent_source.subscribe_queue_calls == 0
        assert len(trade_intent_source.callbacks) == 1

        await collector._publish_update(_ticker_update())

        approved = await asyncio.wait_for(signal_queue.get(), timeout=0.5)
        assert approved.approved is True

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=56,
            contracts=2,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-live-direct",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
            total_thesis_budget_dollars=2.0,
        )
        assert manual_intent is not None

        await trade_intent_source.publish(manual_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.order_id == "server-order"
        assert execution_engine.get_state(manual_intent.decision_id) is not None

        await execution_engine.stop()
        assert trade_intent_source.callbacks == []
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())


def test_execution_engine_recovers_pruned_manual_reservation_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: _fake_model(0.80))
    monkeypatch.setattr(KalshiSignalRiskEngine, "_EXPIRED_RESERVATION_GRACE_SECONDS", 0.05)

    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(feature_engine)
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        KalshiSignalRiskConfig(
            auto_reserve_trade_intents=False,
            starting_cash_dollars=100.0,
            contracts_per_order=1,
            reservation_ttl_seconds=0.05,
        ),
    )
    trade_intent_source = _QueueTradeIntentSource()
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        KalshiExecutionConfig(mode=KalshiExecutionMode.PAPER),
        trade_intent_source=trade_intent_source,
    )
    decision_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()

    async def run() -> None:
        await feature_engine.start()
        await scorer.start()
        await signal_engine.start()
        await execution_engine.start()
        await collector._publish_update(_ticker_update())

        approved = await asyncio.wait_for(decision_queue.get(), timeout=0.5)
        assert approved.approved is True

        manual_intent = signal_engine.reserve_manual_trade_intent(
            decision_state=approved,
            side="YES",
            entry_price_cents=55,
            contracts=1,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            thesis_id="thesis-recover",
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
        )
        assert manual_intent is not None

        await asyncio.sleep(0.25)
        reservation, source = signal_engine._lookup_reservation(manual_intent.decision_id)
        assert reservation is None
        assert source is None

        await trade_intent_source.queue.put(manual_intent)

        filled = await _wait_for_status(execution_queue, "filled", timeout=2.0)
        assert filled.decision_id == manual_intent.decision_id
        assert filled.thesis_id == "thesis-recover"
        assert signal_engine.get_portfolio_state().open_positions[0].decision_id == manual_intent.decision_id

        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()

    asyncio.run(run())
