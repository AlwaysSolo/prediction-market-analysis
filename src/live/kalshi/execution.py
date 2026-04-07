from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from websockets.asyncio.client import connect

from src.indexers.kalshi.models import parse_count, parse_datetime, parse_price_cents
from src.live.kalshi.auth import build_auth_headers
from src.live.kalshi.client import KalshiLiveRestClient
from src.live.kalshi.config import KalshiEnvironment, KalshiReconnectConfig
from src.live.kalshi.signal_risk import (
    KalshiExecutionFeedback,
    KalshiPortfolioPosition,
    KalshiPortfolioSnapshot,
    KalshiSignalRiskEngine,
    KalshiTradeIntent,
)

Callback = Callable[["KalshiExecutionUpdate"], Awaitable[None] | None]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _env_var_names(environment: KalshiEnvironment, suffix: str) -> tuple[str, str]:
    env_prefix = "DEMO" if environment is KalshiEnvironment.DEMO else "PROD"
    return (f"KALSHI_{env_prefix}_EXECUTION_{suffix}", f"KALSHI_EXECUTION_{suffix}")


def _resolve_env_value(environment: KalshiEnvironment, suffix: str) -> str | None:
    for env_name in _env_var_names(environment, suffix):
        value = os.getenv(env_name)
        if value is not None and value != "":
            return value
    return None


def _parse_fixed_point_dollars(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(Decimal(str(value)))


def _parse_timestamp(value: Any) -> datetime:
    if value is None or value == "":
        return utc_now()
    if isinstance(value, str):
        return parse_datetime(value)
    numeric = int(value)
    if numeric > 10_000_000_000:
        return datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
    return datetime.fromtimestamp(numeric, tz=UTC)


class JsonlEventLogger:
    def __init__(self, base_dir: Path, environment: str):
        self.base_dir = base_dir
        self.environment = environment
        self._lock = asyncio.Lock()

    async def write(self, event_type: str, payload: dict[str, Any], event_time: datetime | None = None) -> None:
        event_time = event_time or utc_now()
        date_dir = self.base_dir / self.environment / event_time.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / "events.jsonl"
        row = {
            "logged_at": utc_now().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        async with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")


class KalshiExecutionMode(str, Enum):
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class KalshiExecutionConfig:
    mode: KalshiExecutionMode = KalshiExecutionMode.PAPER
    enable_live_trading: bool = False
    simulate_immediate_fills: bool = False
    subaccount: int = 0
    reconcile_interval_seconds: float = 15.0
    order_reconcile_timeout_seconds: float = 10.0
    reconnect: KalshiReconnectConfig = KalshiReconnectConfig()
    log_dir: Path = Path("output/live/kalshi/execution")

    def __post_init__(self) -> None:
        if self.subaccount < 0:
            raise ValueError("subaccount must be non-negative")
        if self.reconcile_interval_seconds <= 0:
            raise ValueError("reconcile_interval_seconds must be positive")
        if self.order_reconcile_timeout_seconds <= 0:
            raise ValueError("order_reconcile_timeout_seconds must be positive")

    @property
    def live_order_submission_allowed(self) -> bool:
        return self.mode is KalshiExecutionMode.LIVE and self.enable_live_trading

    @classmethod
    def from_env(cls, environment: KalshiEnvironment) -> KalshiExecutionConfig:
        mode = KalshiExecutionMode(_resolve_env_value(environment, "MODE") or KalshiExecutionMode.PAPER.value)
        return cls(
            mode=mode,
            enable_live_trading=_parse_bool(_resolve_env_value(environment, "ENABLE_LIVE_TRADING") or "false"),
            simulate_immediate_fills=_parse_bool(
                _resolve_env_value(environment, "SIMULATE_IMMEDIATE_FILLS") or "false"
            ),
            subaccount=int(_resolve_env_value(environment, "SUBACCOUNT") or 0),
            reconcile_interval_seconds=float(_resolve_env_value(environment, "RECONCILE_INTERVAL_SECONDS") or 15.0),
            order_reconcile_timeout_seconds=float(
                _resolve_env_value(environment, "ORDER_RECONCILE_TIMEOUT_SECONDS") or 10.0
            ),
        )


@dataclass(frozen=True)
class KalshiLiveOrderRecord:
    order_id: str
    client_order_id: str | None
    ticker: str
    side: str
    status: str
    yes_price_cents: int | None
    no_price_cents: int | None
    fill_count: int
    remaining_count: int
    initial_count: int
    taker_fees_dollars: float
    maker_fees_dollars: float
    taker_fill_cost_dollars: float
    maker_fill_cost_dollars: float
    created_time: datetime
    last_update_time: datetime

    @property
    def displayed_price_cents(self) -> int | None:
        if self.side.upper() == "YES":
            return self.yes_price_cents
        return self.no_price_cents


@dataclass(frozen=True)
class KalshiExecutionIntentState:
    decision_id: str
    ticker: str
    side: str
    contracts: int
    mode: KalshiExecutionMode
    status: str
    event_time: datetime
    reference_price_cents: int
    limit_price_cents: int
    client_order_id: str
    order_id: str | None
    filled_contracts: int
    remaining_contracts: int
    fill_price_cents: int | None
    entry_cost_dollars: float
    fees_dollars: float
    cash_required_dollars: float
    available_cash_dollars: float | None
    realized_pnl_dollars: float | None
    cumulative_realized_pnl_dollars: float | None
    settlement_result: str | None
    message: str | None
    live_order: KalshiLiveOrderRecord | None


@dataclass(frozen=True)
class KalshiExecutionUpdate:
    decision_id: str
    ticker: str
    side: str
    contracts: int
    mode: KalshiExecutionMode
    status: str
    event_time: datetime
    reference_price_cents: int
    limit_price_cents: int
    client_order_id: str
    order_id: str | None
    filled_contracts: int
    remaining_contracts: int
    fill_price_cents: int | None
    entry_cost_dollars: float
    fees_dollars: float
    cash_required_dollars: float
    available_cash_dollars: float | None
    realized_pnl_dollars: float | None
    cumulative_realized_pnl_dollars: float | None
    settlement_result: str | None
    message: str | None
    live_order: KalshiLiveOrderRecord | None


@dataclass(frozen=True)
class _FillEvent:
    order_id: str
    client_order_id: str | None
    ticker: str
    side: str
    price_cents: int | None
    count: int
    fees_dollars: float
    event_time: datetime


@dataclass(frozen=True)
class _MarketPositionEvent:
    ticker: str
    side: str | None
    contracts: int
    exposure_dollars: float
    fees_paid_dollars: float
    event_time: datetime


def execution_update_from_state(state: KalshiExecutionIntentState) -> KalshiExecutionUpdate:
    return KalshiExecutionUpdate(
        decision_id=state.decision_id,
        ticker=state.ticker,
        side=state.side,
        contracts=state.contracts,
        mode=state.mode,
        status=state.status,
        event_time=state.event_time,
        reference_price_cents=state.reference_price_cents,
        limit_price_cents=state.limit_price_cents,
        client_order_id=state.client_order_id,
        order_id=state.order_id,
        filled_contracts=state.filled_contracts,
        remaining_contracts=state.remaining_contracts,
        fill_price_cents=state.fill_price_cents,
        entry_cost_dollars=state.entry_cost_dollars,
        fees_dollars=state.fees_dollars,
        cash_required_dollars=state.cash_required_dollars,
        available_cash_dollars=state.available_cash_dollars,
        realized_pnl_dollars=state.realized_pnl_dollars,
        cumulative_realized_pnl_dollars=state.cumulative_realized_pnl_dollars,
        settlement_result=state.settlement_result,
        message=state.message,
        live_order=state.live_order,
    )


def parse_live_order_record(order: dict[str, Any]) -> KalshiLiveOrderRecord:
    return KalshiLiveOrderRecord(
        order_id=order["order_id"],
        client_order_id=order.get("client_order_id"),
        ticker=order["ticker"],
        side=str(order["side"]).upper(),
        status=str(order["status"]).lower(),
        yes_price_cents=parse_price_cents(order.get("yes_price", order.get("yes_price_dollars"))),
        no_price_cents=parse_price_cents(order.get("no_price", order.get("no_price_dollars"))),
        fill_count=parse_count(order.get("fill_count"), order.get("fill_count_fp")),
        remaining_count=parse_count(order.get("remaining_count"), order.get("remaining_count_fp")),
        initial_count=parse_count(order.get("initial_count"), order.get("initial_count_fp")),
        taker_fees_dollars=_parse_fixed_point_dollars(order.get("taker_fees_dollars")),
        maker_fees_dollars=_parse_fixed_point_dollars(order.get("maker_fees_dollars")),
        taker_fill_cost_dollars=_parse_fixed_point_dollars(order.get("taker_fill_cost_dollars")),
        maker_fill_cost_dollars=_parse_fixed_point_dollars(order.get("maker_fill_cost_dollars")),
        created_time=_parse_timestamp(order.get("created_time")),
        last_update_time=_parse_timestamp(order.get("last_update_time") or order.get("created_time")),
    )


def parse_fill_event(msg: dict[str, Any]) -> _FillEvent:
    return _FillEvent(
        order_id=msg["order_id"],
        client_order_id=msg.get("client_order_id"),
        ticker=msg["market_ticker"],
        side=str(msg.get("side", "")).upper(),
        price_cents=parse_price_cents(msg.get("yes_price", msg.get("yes_price_dollars"))),
        count=parse_count(msg.get("count"), msg.get("count_fp")),
        fees_dollars=_parse_fixed_point_dollars(msg.get("fee_cost")),
        event_time=_parse_timestamp(msg.get("ts")),
    )


def parse_market_position_event(msg: dict[str, Any]) -> _MarketPositionEvent:
    position_fp = Decimal(str(msg.get("position_fp", "0")))
    side: str | None
    if position_fp > 0:
        side = "YES"
    elif position_fp < 0:
        side = "NO"
    else:
        side = None
    return _MarketPositionEvent(
        ticker=msg["market_ticker"],
        side=side,
        contracts=int(abs(position_fp)),
        exposure_dollars=_parse_fixed_point_dollars(msg.get("position_cost_dollars")),
        fees_paid_dollars=_parse_fixed_point_dollars(msg.get("fees_paid_dollars")),
        event_time=utc_now(),
    )


def portfolio_position_from_market_position_event(event: _MarketPositionEvent) -> KalshiPortfolioPosition | None:
    if event.side is None or event.contracts <= 0:
        return None
    return KalshiPortfolioPosition(
        ticker=event.ticker,
        side=event.side,
        contracts=event.contracts,
        entry_cost_dollars=event.exposure_dollars,
        fees_dollars=event.fees_paid_dollars,
        cash_required_dollars=event.exposure_dollars + event.fees_paid_dollars,
    )


def portfolio_position_from_rest_position(position: dict[str, Any]) -> KalshiPortfolioPosition | None:
    contracts_fp = Decimal(str(position.get("position_fp", "0")))
    if contracts_fp == 0:
        return None
    side = "YES" if contracts_fp > 0 else "NO"
    contracts = int(abs(contracts_fp))
    exposure_dollars = _parse_fixed_point_dollars(position.get("market_exposure_dollars"))
    fees_paid_dollars = _parse_fixed_point_dollars(position.get("fees_paid_dollars"))
    return KalshiPortfolioPosition(
        ticker=position["ticker"],
        side=side,
        contracts=contracts,
        entry_cost_dollars=exposure_dollars,
        fees_dollars=fees_paid_dollars,
        cash_required_dollars=exposure_dollars + fees_paid_dollars,
    )


def portfolio_position_from_live_order(record: KalshiLiveOrderRecord) -> KalshiPortfolioPosition | None:
    if record.fill_count <= 0:
        return None
    entry_cost_dollars = record.taker_fill_cost_dollars or record.maker_fill_cost_dollars
    if entry_cost_dollars == 0.0 and record.displayed_price_cents is not None:
        entry_cost_dollars = (record.displayed_price_cents / 100.0) * record.fill_count
    fees_dollars = record.taker_fees_dollars + record.maker_fees_dollars
    return KalshiPortfolioPosition(
        ticker=record.ticker,
        side=record.side,
        contracts=record.fill_count,
        entry_cost_dollars=entry_cost_dollars,
        fees_dollars=fees_dollars,
        cash_required_dollars=entry_cost_dollars + fees_dollars,
    )


def build_create_order_payload(intent: KalshiTradeIntent, *, subaccount: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": intent.ticker,
        "client_order_id": intent.decision_id,
        "side": intent.side.lower(),
        "action": "buy",
        "count": intent.contracts,
        "type": "limit",
        "time_in_force": "immediate_or_cancel",
    }
    price_field = "yes_price" if intent.side.upper() == "YES" else "no_price"
    payload[price_field] = intent.max_acceptable_entry_price_cents
    if subaccount is not None:
        payload["subaccount"] = subaccount
    return payload


class KalshiExecutionEngine:
    def __init__(
        self,
        signal_engine: KalshiSignalRiskEngine,
        config: KalshiExecutionConfig | None = None,
    ):
        self.signal_engine = signal_engine
        self._collector = self.signal_engine.scorer.feature_engine.collector
        self._collector_config = self.signal_engine.scorer.feature_engine.collector.config
        self.environment = self._collector_config.environment
        self.config = config or KalshiExecutionConfig.from_env(self.environment)
        self._rest_client = KalshiLiveRestClient(self._collector_config)
        self._logger = JsonlEventLogger(self.config.log_dir, self.environment.value)
        self._signal_queue: asyncio.Queue[KalshiTradeIntent] | None = None
        self._states: dict[str, KalshiExecutionIntentState] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiExecutionUpdate]] = []
        self._client_order_to_decision: dict[str, str] = {}
        self._order_to_decision: dict[str, str] = {}
        self._ws_market_positions: dict[str, KalshiPortfolioPosition] = {}
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._reconcile_event = asyncio.Event()
        self._full_reconcile_requested = False
        self._consume_task: asyncio.Task[Any] | None = None
        self._ws_task: asyncio.Task[Any] | None = None
        self._reconcile_task: asyncio.Task[Any] | None = None
        self._message_id = 1
        self._last_portfolio_snapshot: KalshiPortfolioSnapshot | None = None
        self._simulated_realized_pnl_dollars = 0.0
        self._live_realized_pnl_dollars = 0.0

    async def start(self) -> None:
        if self._consume_task and not self._consume_task.done():
            return

        if (
            self.config.mode is KalshiExecutionMode.LIVE
            and not self.config.enable_live_trading
            and not self._simulation_enabled()
        ):
            raise RuntimeError(
                "Live execution mode requires KALSHI_EXECUTION_ENABLE_LIVE_TRADING=true."
            )

        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        self._reconcile_event = asyncio.Event()
        self._full_reconcile_requested = False
        if self._signal_queue is None:
            self._signal_queue = self.signal_engine.subscribe_trade_intent_queue()

        await self._logger.write(
            "execution_started",
            {
                "environment": self.environment.value,
                "mode": self.config.mode.value,
                "simulate_immediate_fills": self._simulation_enabled(),
                "subaccount": self.config.subaccount,
            },
        )
        if self.config.mode is KalshiExecutionMode.SHADOW:
            await self._refresh_portfolio_snapshot()
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="kalshi-execution-reconcile")
        elif self._simulation_enabled():
            await self._sync_signal_portfolio_snapshot(
                event_time=utc_now(),
                log_event_type="simulated_portfolio_initialized",
            )
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="kalshi-execution-reconcile")
        elif self.config.mode in {KalshiExecutionMode.LIVE, KalshiExecutionMode.SHADOW}:
            await self._refresh_portfolio_snapshot()
            if self.config.mode is KalshiExecutionMode.LIVE:
                await self._recover_recent_orders()
                self._ws_task = asyncio.create_task(self._private_ws_loop(), name="kalshi-execution-private-ws")
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="kalshi-execution-reconcile")

        await self._bootstrap_from_signal_engine()
        self._consume_task = asyncio.create_task(self._consume_loop(), name="kalshi-execution-consume")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        for task in (self._consume_task, self._ws_task, self._reconcile_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._consume_task = None
        self._ws_task = None
        self._reconcile_task = None
        self._rest_client.close()
        await self._logger.write(
            "execution_stopped",
            {
                "environment": self.environment.value,
                "mode": self.config.mode.value,
                "subaccount": self.config.subaccount,
            },
        )

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def get_state(self, decision_id: str) -> KalshiExecutionIntentState | None:
        return self._states.get(decision_id)

    def snapshot_states(self) -> dict[str, KalshiExecutionIntentState]:
        return dict(self._states)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiExecutionUpdate]:
        queue: asyncio.Queue[KalshiExecutionUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def get_portfolio_snapshot(self) -> KalshiPortfolioSnapshot | None:
        return self._last_portfolio_snapshot

    def get_realized_pnl_dollars(self) -> float:
        return self._simulated_realized_pnl_dollars + self._live_realized_pnl_dollars

    def _simulation_enabled(self) -> bool:
        return self.config.simulate_immediate_fills

    def _simulation_label(self) -> str:
        if self.config.mode is KalshiExecutionMode.PAPER:
            return "paper"
        if self.config.mode is KalshiExecutionMode.SHADOW:
            return "shadow"
        return "simulated"

    def _current_open_positions(self) -> tuple[KalshiPortfolioPosition, ...]:
        return tuple(self._ws_market_positions[ticker] for ticker in sorted(self._ws_market_positions))

    async def _apply_cached_portfolio_snapshot(
        self,
        available_cash_dollars: float,
        *,
        event_time: datetime,
        log_event_type: str,
    ) -> None:
        open_positions = self._current_open_positions()
        deployed_capital_dollars = sum(position.entry_cost_dollars for position in open_positions)
        snapshot = KalshiPortfolioSnapshot(
            event_time=event_time,
            available_cash_dollars=available_cash_dollars,
            deployed_capital_dollars=deployed_capital_dollars,
            open_positions=open_positions,
        )
        self._last_portfolio_snapshot = snapshot
        await self.signal_engine.apply_portfolio_snapshot(snapshot)
        await self._logger.write(
            log_event_type,
            {
                "available_cash_dollars": available_cash_dollars,
                "deployed_capital_dollars": deployed_capital_dollars,
                "open_positions": [position.ticker for position in open_positions],
            },
            event_time=event_time,
        )

    async def _sync_signal_portfolio_snapshot(self, *, event_time: datetime, log_event_type: str) -> None:
        portfolio_state = self.signal_engine.get_portfolio_state()
        snapshot = KalshiPortfolioSnapshot(
            event_time=event_time,
            available_cash_dollars=portfolio_state.available_cash_dollars,
            deployed_capital_dollars=portfolio_state.deployed_capital_dollars,
            open_positions=portfolio_state.open_positions,
        )
        self._last_portfolio_snapshot = snapshot
        await self._logger.write(
            log_event_type,
            {
                "available_cash_dollars": portfolio_state.available_cash_dollars,
                "deployed_capital_dollars": portfolio_state.deployed_capital_dollars,
                "open_positions": [position.ticker for position in portfolio_state.open_positions],
                "pending_reservations": [position.ticker for position in portfolio_state.pending_reservations],
            },
            event_time=event_time,
        )

    async def _bootstrap_from_signal_engine(self) -> None:
        for decision_state in self.signal_engine.snapshot_states().values():
            if not decision_state.approved or decision_state.trade_intent is None:
                continue
            await self._handle_trade_intent(decision_state.trade_intent, source="bootstrap")

    async def _consume_loop(self) -> None:
        if self._signal_queue is None:
            return

        try:
            while not self._stop_event.is_set():
                intent = await self._signal_queue.get()
                try:
                    await self._handle_trade_intent(intent, source="live_signal")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._logger.write(
                        "consume_error",
                        {
                            "decision_id": intent.decision_id,
                            "ticker": intent.ticker,
                            "side": intent.side,
                            "error": repr(exc),
                        },
                    )
        except asyncio.CancelledError:
            raise

    async def _handle_trade_intent(self, intent: KalshiTradeIntent, *, source: str) -> None:
        existing = self._states.get(intent.decision_id)
        if existing is not None:
            return

        claimed_state = KalshiExecutionIntentState(
            decision_id=intent.decision_id,
            ticker=intent.ticker,
            side=intent.side,
            contracts=intent.contracts,
            mode=self.config.mode,
            status="claimed",
            event_time=utc_now(),
            reference_price_cents=intent.reference_price_cents,
            limit_price_cents=intent.max_acceptable_entry_price_cents,
            client_order_id=intent.decision_id,
            order_id=None,
            filled_contracts=0,
            remaining_contracts=intent.contracts,
            fill_price_cents=None,
            entry_cost_dollars=intent.estimated_entry_cost_dollars,
            fees_dollars=intent.estimated_fees_dollars,
            cash_required_dollars=intent.estimated_cash_required_dollars,
            available_cash_dollars=self._current_available_cash_dollars(),
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            settlement_result=None,
            message=source,
            live_order=None,
        )
        self._states[intent.decision_id] = claimed_state
        self._client_order_to_decision[intent.decision_id] = intent.decision_id
        await self._logger.write(
            "intent_claimed",
            {"decision_id": intent.decision_id, "ticker": intent.ticker, "side": intent.side, "source": source},
        )
        if self.config.mode in {KalshiExecutionMode.PAPER, KalshiExecutionMode.SHADOW} or self._simulation_enabled():
            await self._sync_signal_portfolio_snapshot(
                event_time=claimed_state.event_time,
                log_event_type=f"{self._simulation_label()}_portfolio_claimed",
            )
        await self._publish_state(claimed_state)

        if self.config.mode in {KalshiExecutionMode.PAPER, KalshiExecutionMode.SHADOW} or self._simulation_enabled():
            await self._execute_simulated_intent(intent)
            return

        await self._submit_live_intent(intent)

    async def _execute_simulated_intent(self, intent: KalshiTradeIntent) -> None:
        simulation_label = self._simulation_label()
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=intent.decision_id,
                status="accepted",
                event_time=utc_now(),
            )
        )
        accepted_state = replace(
            self._states[intent.decision_id],
            status="accepted",
            event_time=utc_now(),
            message=f"{simulation_label}_submit_accepted",
            available_cash_dollars=self.signal_engine.get_portfolio_state().available_cash_dollars,
        )
        self._states[intent.decision_id] = accepted_state
        await self._sync_signal_portfolio_snapshot(
            event_time=accepted_state.event_time,
            log_event_type=f"{simulation_label}_portfolio_accepted",
        )
        await self._publish_state(accepted_state)

        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=intent.decision_id,
                status="filled",
                event_time=utc_now(),
                filled_contracts=intent.contracts,
                filled_price_cents=intent.reference_price_cents,
            )
        )
        filled_state = replace(
            accepted_state,
            status="filled",
            event_time=utc_now(),
            filled_contracts=intent.contracts,
            remaining_contracts=0,
            fill_price_cents=intent.reference_price_cents,
            available_cash_dollars=self.signal_engine.get_portfolio_state().available_cash_dollars,
            message=f"{simulation_label}_fill",
        )
        self._states[intent.decision_id] = filled_state
        await self._sync_signal_portfolio_snapshot(
            event_time=filled_state.event_time,
            log_event_type=f"{simulation_label}_portfolio_filled",
        )
        await self._publish_state(filled_state)

    async def _execute_shadow_intent(self, intent: KalshiTradeIntent) -> None:
        await self._execute_simulated_intent(intent)

    async def _submit_live_intent(self, intent: KalshiTradeIntent) -> None:
        payload = build_create_order_payload(intent, subaccount=self.config.subaccount)
        try:
            response = await self._call_rest(self._rest_client.create_order, payload)
            await self._logger.write("submit_response", {"decision_id": intent.decision_id, "response": response})
        except httpx.HTTPStatusError as exc:
            await self._logger.write(
                "submit_error",
                {
                    "decision_id": intent.decision_id,
                    "status_code": exc.response.status_code,
                    "response": exc.response.text,
                },
            )
            if exc.response.status_code == 409:
                if await self._reconcile_or_retry(intent, reason="duplicate_client_order_id"):
                    return
                await self._finalize_rejected(intent.decision_id, "duplicate_client_order_id")
                return
            if 400 <= exc.response.status_code < 500:
                await self._finalize_rejected(intent.decision_id, f"http_{exc.response.status_code}")
                return
            if await self._reconcile_or_retry(intent, reason=f"http_{exc.response.status_code}"):
                return
            await self._finalize_error(intent.decision_id, f"http_{exc.response.status_code}")
            return
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            await self._logger.write(
                "submit_error",
                {"decision_id": intent.decision_id, "error": repr(exc)},
            )
            if await self._reconcile_or_retry(intent, reason=type(exc).__name__):
                return
            await self._finalize_error(intent.decision_id, type(exc).__name__)
            return

        await self._ensure_signal_accepted(intent.decision_id)
        order_payload = response.get("order")
        if isinstance(order_payload, dict) and order_payload:
            await self._apply_order_record(parse_live_order_record(order_payload), allow_signal_feedback=True)
        else:
            accepted_state = replace(
                self._states[intent.decision_id],
                status="accepted",
                event_time=utc_now(),
                message="submit_accepted",
                available_cash_dollars=self._current_available_cash_dollars(),
            )
            self._states[intent.decision_id] = accepted_state
            await self._publish_state(accepted_state)
            self._request_reconcile()

    async def _reconcile_or_retry(self, intent: KalshiTradeIntent, *, reason: str) -> bool:
        await self._ensure_signal_accepted(intent.decision_id)
        reconciling_state = replace(
            self._states[intent.decision_id],
            status="reconciling",
            event_time=utc_now(),
            message=reason,
            available_cash_dollars=self._current_available_cash_dollars(),
        )
        self._states[intent.decision_id] = reconciling_state
        await self._publish_state(reconciling_state)
        await self._logger.write("submit_reconciling", {"decision_id": intent.decision_id, "reason": reason})

        deadline = utc_now() + timedelta(seconds=self.config.order_reconcile_timeout_seconds)
        while utc_now() < deadline:
            record = await self._find_order_by_client_order_id(intent.decision_id, intent.ticker)
            if record is not None:
                await self._apply_order_record(record, allow_signal_feedback=True)
                return True
            remaining_seconds = max(0.0, (deadline - utc_now()).total_seconds())
            await asyncio.sleep(min(0.5, remaining_seconds))

        record = await self._find_order_by_client_order_id(intent.decision_id, intent.ticker)
        if record is not None:
            await self._apply_order_record(record, allow_signal_feedback=True)
            return True

        try:
            response = await self._call_rest(
                self._rest_client.create_order,
                build_create_order_payload(intent, subaccount=self.config.subaccount),
            )
            await self._logger.write("submit_retry_response", {"decision_id": intent.decision_id, "response": response})
        except httpx.HTTPStatusError as exc:
            await self._logger.write(
                "submit_retry_error",
                {
                    "decision_id": intent.decision_id,
                    "status_code": exc.response.status_code,
                    "response": exc.response.text,
                },
            )
            if exc.response.status_code == 409:
                record = await self._find_order_by_client_order_id(intent.decision_id, intent.ticker)
                if record is not None:
                    await self._apply_order_record(record, allow_signal_feedback=True)
                    return True
            return False
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            await self._logger.write(
                "submit_retry_error",
                {"decision_id": intent.decision_id, "error": repr(exc)},
            )
            return False

        order_payload = response.get("order")
        if isinstance(order_payload, dict) and order_payload:
            await self._apply_order_record(parse_live_order_record(order_payload), allow_signal_feedback=True)
            return True
        return False

    async def _ensure_signal_accepted(self, decision_id: str) -> None:
        state = self._states.get(decision_id)
        if state is None or state.status not in {"claimed", "reconciling"}:
            return
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(decision_id=decision_id, status="accepted", event_time=utc_now())
        )
        accepted_state = replace(
            state,
            status="accepted",
            event_time=utc_now(),
            message="accepted",
            available_cash_dollars=self._current_available_cash_dollars(),
        )
        self._states[decision_id] = accepted_state
        await self._publish_state(accepted_state)

    async def _finalize_rejected(self, decision_id: str, reason: str) -> None:
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(decision_id=decision_id, status="rejected", event_time=utc_now())
        )
        state = self._states[decision_id]
        rejected_state = replace(
            state,
            status="rejected",
            event_time=utc_now(),
            message=reason,
            available_cash_dollars=self._current_available_cash_dollars(),
        )
        self._states[decision_id] = rejected_state
        await self._publish_state(rejected_state)

    async def _finalize_cancelled(self, decision_id: str, reason: str) -> None:
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(decision_id=decision_id, status="cancelled", event_time=utc_now())
        )
        state = self._states[decision_id]
        cancelled_state = replace(
            state,
            status="cancelled",
            event_time=utc_now(),
            message=reason,
            available_cash_dollars=self._current_available_cash_dollars(),
        )
        self._states[decision_id] = cancelled_state
        await self._publish_state(cancelled_state)

    async def _finalize_error(self, decision_id: str, reason: str) -> None:
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(decision_id=decision_id, status="rejected", event_time=utc_now())
        )
        state = self._states[decision_id]
        error_state = replace(
            state,
            status="error",
            event_time=utc_now(),
            message=reason,
            available_cash_dollars=self._current_available_cash_dollars(),
        )
        self._states[decision_id] = error_state
        await self._publish_state(error_state)

    async def _apply_order_record(
        self,
        record: KalshiLiveOrderRecord,
        *,
        allow_signal_feedback: bool,
    ) -> None:
        decision_id = record.client_order_id or self._order_to_decision.get(record.order_id)
        if decision_id is None:
            return

        existing = self._states.get(decision_id)
        if existing is None:
            existing = KalshiExecutionIntentState(
                decision_id=decision_id,
                ticker=record.ticker,
                side=record.side,
                contracts=record.initial_count or (record.fill_count + record.remaining_count),
                mode=KalshiExecutionMode.LIVE,
                status="recovered",
                event_time=record.last_update_time,
                reference_price_cents=record.displayed_price_cents or 0,
                limit_price_cents=record.displayed_price_cents or 0,
                client_order_id=decision_id,
                order_id=record.order_id,
                filled_contracts=record.fill_count,
                remaining_contracts=record.remaining_count,
                fill_price_cents=record.displayed_price_cents,
                entry_cost_dollars=record.taker_fill_cost_dollars or record.maker_fill_cost_dollars,
                fees_dollars=record.taker_fees_dollars + record.maker_fees_dollars,
                cash_required_dollars=(
                    (record.taker_fill_cost_dollars or record.maker_fill_cost_dollars)
                    + record.taker_fees_dollars
                    + record.maker_fees_dollars
                ),
                available_cash_dollars=self._current_available_cash_dollars(),
                realized_pnl_dollars=None,
                cumulative_realized_pnl_dollars=None,
                settlement_result=None,
                message="recovered_order",
                live_order=record,
            )

        self._client_order_to_decision[decision_id] = decision_id
        self._order_to_decision[record.order_id] = decision_id

        if existing.status in {"filled", "cancelled", "rejected"}:
            updated_state = replace(
                existing,
                event_time=record.last_update_time,
                live_order=record,
                order_id=record.order_id,
                message="terminal_order_refresh",
                available_cash_dollars=self._current_available_cash_dollars(),
            )
            self._states[decision_id] = updated_state
            await self._publish_state(updated_state)
            return

        if record.fill_count > 0 and (record.status in {"executed", "canceled"} or record.remaining_count == 0):
            position = portfolio_position_from_live_order(record)
            if position is not None:
                self._ws_market_positions[record.ticker] = position
            if allow_signal_feedback:
                await self.signal_engine.apply_execution_feedback(
                    KalshiExecutionFeedback(
                        decision_id=decision_id,
                        status="filled",
                        event_time=record.last_update_time,
                        filled_contracts=record.fill_count,
                        filled_price_cents=record.displayed_price_cents,
                    )
                )
            filled_state = replace(
                existing,
                status="filled",
                event_time=record.last_update_time,
                order_id=record.order_id,
                filled_contracts=record.fill_count,
                remaining_contracts=record.remaining_count,
                fill_price_cents=record.displayed_price_cents,
                available_cash_dollars=self._current_available_cash_dollars(),
                message="order_terminal_fill",
                live_order=record,
            )
            self._states[decision_id] = filled_state
            await self._publish_state(filled_state)
            return

        if record.status == "canceled" and record.fill_count == 0:
            self._ws_market_positions.pop(record.ticker, None)
            if allow_signal_feedback:
                await self.signal_engine.apply_execution_feedback(
                    KalshiExecutionFeedback(
                        decision_id=decision_id,
                        status="cancelled",
                        event_time=record.last_update_time,
                    )
                )
            cancelled_state = replace(
                existing,
                status="cancelled",
                event_time=record.last_update_time,
                order_id=record.order_id,
                available_cash_dollars=self._current_available_cash_dollars(),
                message="order_cancelled",
                live_order=record,
            )
            self._states[decision_id] = cancelled_state
            await self._publish_state(cancelled_state)
            return

        position = portfolio_position_from_live_order(record)
        if position is not None:
            self._ws_market_positions[record.ticker] = position
        if allow_signal_feedback:
            await self._ensure_signal_accepted(decision_id)
        accepted_state = replace(
            existing,
            status="accepted",
            event_time=record.last_update_time,
            order_id=record.order_id,
            filled_contracts=record.fill_count,
            remaining_contracts=record.remaining_count,
            fill_price_cents=record.displayed_price_cents if record.fill_count > 0 else existing.fill_price_cents,
            available_cash_dollars=self._current_available_cash_dollars(),
            message="order_update",
            live_order=record,
        )
        self._states[decision_id] = accepted_state
        await self._publish_state(accepted_state)

    async def _handle_fill_event(self, fill: _FillEvent) -> None:
        decision_id = fill.client_order_id or self._order_to_decision.get(fill.order_id)
        if decision_id is None or decision_id not in self._states:
            return
        state = self._states[decision_id]
        if state.status in {"filled", "cancelled", "rejected"}:
            return
        filled_contracts = min(state.contracts, state.filled_contracts + fill.count)
        remaining_contracts = max(0, state.contracts - filled_contracts)
        next_status = "partially_filled" if remaining_contracts > 0 else "filled"
        updated_state = replace(
            state,
            status=next_status,
            event_time=fill.event_time,
            filled_contracts=filled_contracts,
            remaining_contracts=remaining_contracts,
            fill_price_cents=fill.price_cents,
            available_cash_dollars=self._current_available_cash_dollars(),
            message="fill_update",
        )
        self._states[decision_id] = updated_state
        await self._publish_state(updated_state)

    async def _handle_market_position_event(self, event: _MarketPositionEvent) -> None:
        position = portfolio_position_from_market_position_event(event)
        if position is None:
            self._ws_market_positions.pop(event.ticker, None)
            return
        self._ws_market_positions[event.ticker] = position

    async def _private_ws_loop(self) -> None:
        attempt = 0
        try:
            while not self._stop_event.is_set():
                try:
                    await self._connect_and_consume_private_ws()
                    attempt = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    attempt += 1
                    delay = min(
                        self.config.reconnect.initial_backoff_seconds * (2 ** (attempt - 1)),
                        self.config.reconnect.max_backoff_seconds,
                    )
                    await self._logger.write(
                        "private_ws_disconnected",
                        {"attempt": attempt, "delay_seconds": delay, "error": repr(exc)},
                    )
                    await asyncio.sleep(delay)
                    self._request_reconcile()
        except asyncio.CancelledError:
            raise

    async def _connect_and_consume_private_ws(self) -> None:
        path = "/trade-api/ws/v2"
        timestamp_ms = int(utc_now().timestamp() * 1000)
        headers = build_auth_headers(self._collector_config.credentials, "GET", path, timestamp_ms)
        async with connect(
            self.environment.websocket_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            await self._logger.write(
                "private_ws_connected",
                {"mode": self.config.mode.value},
            )
            for message in self._private_subscription_messages():
                await websocket.send(json.dumps(message))
                await self._logger.write("private_ws_subscribed", message)
            async for raw_message in websocket:
                payload = json.loads(raw_message)
                await self._logger.write("private_ws_message", {"message": payload})
                await self._handle_private_ws_message(payload)

    def _private_subscription_messages(self) -> list[dict[str, Any]]:
        message = {
            "id": self._message_id,
            "cmd": "subscribe",
            "params": {"channels": ["user_orders", "fill", "market_positions"]},
        }
        self._message_id += 1
        return [message]

    async def _handle_private_ws_message(self, payload: dict[str, Any]) -> None:
        message_type = payload.get("type")
        if message_type in {"subscribed", "ok", "unsubscribed"}:
            return
        if message_type == "error":
            raise RuntimeError(f"Kalshi private websocket error: {payload}")
        if message_type == "user_order":
            msg = payload["msg"]
            if self._message_subaccount(msg.get("subaccount_number")) != self.config.subaccount:
                return
            await self._apply_order_record(parse_live_order_record(msg), allow_signal_feedback=True)
            return
        if message_type == "fill":
            msg = payload["msg"]
            if self._message_subaccount(msg.get("subaccount")) != self.config.subaccount:
                return
            await self._handle_fill_event(parse_fill_event(msg))
            return
        if message_type == "market_position":
            msg = payload["msg"]
            if self._message_subaccount(msg.get("subaccount")) != self.config.subaccount:
                return
            await self._handle_market_position_event(parse_market_position_event(msg))

    def _message_subaccount(self, value: Any) -> int:
        if value is None or value == "":
            return 0
        return int(value)

    async def _reconcile_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                full_reconcile = False
                try:
                    await asyncio.wait_for(self._reconcile_event.wait(), timeout=self.config.reconcile_interval_seconds)
                    full_reconcile = self._full_reconcile_requested
                except asyncio.TimeoutError:
                    pass
                else:
                    self._full_reconcile_requested = False
                self._reconcile_event.clear()
                if self._stop_event.is_set():
                    break
                if self.config.mode is KalshiExecutionMode.SHADOW or self._simulation_enabled():
                    await self._settle_simulated_positions()
                elif full_reconcile:
                    await self._refresh_portfolio_snapshot()
                    await self._reconcile_live_states()
                    await self._settle_live_positions()
                else:
                    await self._refresh_balance_snapshot()
                    await self._settle_live_positions()
        except asyncio.CancelledError:
            raise

    def _request_reconcile(self, *, full: bool = True) -> None:
        self._full_reconcile_requested = self._full_reconcile_requested or full
        self._reconcile_event.set()

    async def _call_rest(self, method: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(method, *args, **kwargs)

    async def _refresh_portfolio_snapshot(self) -> None:
        balance, positions_payload = await asyncio.gather(
            self._call_rest(self._rest_client.get_balance, subaccount=self.config.subaccount),
            self._call_rest(self._rest_client.get_positions, subaccount=self.config.subaccount),
        )
        self._ws_market_positions.clear()
        for position in positions_payload:
            parsed_position = portfolio_position_from_rest_position(position)
            if parsed_position is None:
                continue
            self._ws_market_positions[parsed_position.ticker] = parsed_position

        await self._apply_cached_portfolio_snapshot(
            balance.get("balance", 0) / 100.0,
            event_time=_parse_timestamp(balance.get("updated_ts")),
            log_event_type="portfolio_snapshot_full",
        )

    async def _refresh_balance_snapshot(self) -> None:
        balance = await self._call_rest(self._rest_client.get_balance, subaccount=self.config.subaccount)
        await self._apply_cached_portfolio_snapshot(
            balance.get("balance", 0) / 100.0,
            event_time=_parse_timestamp(balance.get("updated_ts")),
            log_event_type="portfolio_snapshot_balance",
        )

    async def _settle_simulated_positions(self) -> None:
        for state in list(self._states.values()):
            if state.status != "filled":
                continue

            collector_state = self._collector.get_state(state.ticker)
            now = utc_now()
            if collector_state is not None:
                close_time = collector_state.close_time
                if collector_state.is_open and (close_time is None or close_time > now):
                    continue
            elif state.event_time + timedelta(minutes=20) > now:
                continue

            try:
                market = await self._call_rest(self._rest_client.get_market, state.ticker)
            except Exception as exc:
                await self._logger.write(
                    "live_settlement_lookup_failed",
                    {
                        "decision_id": state.decision_id,
                        "ticker": state.ticker,
                        "subaccount": self.config.subaccount,
                        "error": str(exc),
                    },
                    event_time=utc_now(),
                )
                continue
            settlement_result = market.result.strip().upper()
            if settlement_result not in {"YES", "NO"}:
                continue

            payout_dollars = float(state.contracts) if settlement_result == state.side.upper() else 0.0
            realized_pnl_dollars = payout_dollars - state.cash_required_dollars
            self._simulated_realized_pnl_dollars += realized_pnl_dollars

            await self.signal_engine.apply_execution_feedback(
                KalshiExecutionFeedback(
                    decision_id=state.decision_id,
                    status="released",
                    event_time=utc_now(),
                    cash_delta_dollars=realized_pnl_dollars,
                )
            )
            settled_at = utc_now()
            settled_state = replace(
                state,
                status="settled",
                event_time=settled_at,
                available_cash_dollars=self.signal_engine.get_portfolio_state().available_cash_dollars,
                realized_pnl_dollars=realized_pnl_dollars,
                cumulative_realized_pnl_dollars=self._simulated_realized_pnl_dollars,
                settlement_result=settlement_result,
                message=f"{self._simulation_label()}_settlement",
            )
            self._states[state.decision_id] = settled_state
            await self._logger.write(
                "simulated_position_settled",
                {
                    "decision_id": state.decision_id,
                    "ticker": state.ticker,
                    "side": state.side,
                    "settlement_result": settlement_result,
                    "contracts": state.contracts,
                    "cash_required_dollars": state.cash_required_dollars,
                    "realized_pnl_dollars": realized_pnl_dollars,
                    "cumulative_realized_pnl_dollars": self._simulated_realized_pnl_dollars,
                },
                event_time=settled_at,
            )
            await self._sync_signal_portfolio_snapshot(
                event_time=settled_at,
                log_event_type="simulated_portfolio_settled",
            )
            await self._publish_state(settled_state)

    async def _settle_live_positions(self) -> None:
        for state in list(self._states.values()):
            if state.mode is not KalshiExecutionMode.LIVE or state.status != "filled":
                continue

            collector_state = self._collector.get_state(state.ticker)
            now = utc_now()
            if collector_state is not None:
                close_time = collector_state.close_time
                if collector_state.is_open and (close_time is None or close_time > now):
                    continue
            elif state.event_time + timedelta(minutes=20) > now:
                continue

            try:
                market = await self._call_rest(self._rest_client.get_market, state.ticker)
            except Exception as exc:
                await self._logger.write(
                    "live_settlement_lookup_failed",
                    {
                        "decision_id": state.decision_id,
                        "ticker": state.ticker,
                        "subaccount": self.config.subaccount,
                        "error": str(exc),
                    },
                    event_time=utc_now(),
                )
                continue
            settlement_result = market.result.strip().upper()
            if settlement_result not in {"YES", "NO"}:
                continue

            payout_dollars = float(state.contracts) if settlement_result == state.side.upper() else 0.0
            realized_pnl_dollars = payout_dollars - state.cash_required_dollars
            self._live_realized_pnl_dollars += realized_pnl_dollars

            settled_at = utc_now()
            settled_state = replace(
                state,
                status="settled",
                event_time=settled_at,
                available_cash_dollars=self._current_available_cash_dollars(),
                realized_pnl_dollars=realized_pnl_dollars,
                cumulative_realized_pnl_dollars=self._live_realized_pnl_dollars,
                settlement_result=settlement_result,
                message="live_settlement",
            )
            self._states[state.decision_id] = settled_state
            await self._logger.write(
                "live_position_settled",
                {
                    "decision_id": state.decision_id,
                    "ticker": state.ticker,
                    "side": state.side,
                    "subaccount": self.config.subaccount,
                    "settlement_result": settlement_result,
                    "contracts": state.contracts,
                    "cash_required_dollars": state.cash_required_dollars,
                    "realized_pnl_dollars": realized_pnl_dollars,
                    "cumulative_realized_pnl_dollars": self._live_realized_pnl_dollars,
                },
                event_time=settled_at,
            )
            await self._publish_state(settled_state)

    async def _recover_recent_orders(self) -> None:
        min_ts = int((utc_now() - timedelta(hours=1)).timestamp())
        for order in await self._call_rest(
            self._rest_client.get_orders,
            min_ts=min_ts,
            subaccount=self.config.subaccount,
        ):
            client_order_id = order.get("client_order_id")
            if not client_order_id:
                continue
            await self._apply_order_record(parse_live_order_record(order), allow_signal_feedback=False)

    async def _reconcile_live_states(self) -> None:
        for state in list(self._states.values()):
            if state.mode is not KalshiExecutionMode.LIVE:
                continue
            if state.status in {"filled", "cancelled", "rejected", "error"}:
                continue
            record: KalshiLiveOrderRecord | None = None
            if state.order_id:
                payload = await self._call_rest(self._rest_client.get_order, state.order_id, subaccount=self.config.subaccount)
                order_payload = payload.get("order")
                if isinstance(order_payload, dict):
                    record = parse_live_order_record(order_payload)
            if record is None:
                record = await self._find_order_by_client_order_id(state.client_order_id, state.ticker)
            if record is None:
                continue
            await self._apply_order_record(record, allow_signal_feedback=True)

    async def _find_order_by_client_order_id(self, client_order_id: str, ticker: str | None = None) -> KalshiLiveOrderRecord | None:
        return await self._call_rest(self._find_order_by_client_order_id_sync, client_order_id, ticker)

    def _find_order_by_client_order_id_sync(
        self,
        client_order_id: str,
        ticker: str | None = None,
    ) -> KalshiLiveOrderRecord | None:
        min_ts = int((utc_now() - timedelta(hours=1)).timestamp())
        for order in self._rest_client.get_orders(
            ticker=ticker,
            min_ts=min_ts,
            subaccount=self.config.subaccount,
        ):
            if order.get("client_order_id") != client_order_id:
                continue
            return parse_live_order_record(order)
        return None

    def _current_available_cash_dollars(self) -> float | None:
        if self._last_portfolio_snapshot is not None:
            return self._last_portfolio_snapshot.available_cash_dollars
        return self.signal_engine.get_portfolio_state().available_cash_dollars

    async def _publish_state(self, state: KalshiExecutionIntentState) -> None:
        update = execution_update_from_state(state)
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)
