from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import uuid
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
from src.live.kalshi.jsonl_logger import JsonlEventLogger
from src.live.kalshi.signal_risk import (
    KalshiExecutionFeedback,
    KalshiPortfolioPosition,
    KalshiPortfolioSnapshot,
    KalshiSignalRiskEngine,
    KalshiTradeIntent,
    calculate_realized_cash_metrics,
    find_max_acceptable_entry_price_cents,
)
from src.live.kalshi.trade_intent_source import KalshiTradeIntentSource

Callback = Callable[["KalshiExecutionUpdate"], Awaitable[None] | None]
TradeIntentDispatchCallback = Callable[[KalshiTradeIntent], Awaitable[None] | None]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _normalize_time_in_force(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "ioc": "immediate_or_cancel",
        "fok": "fill_or_kill",
        "gtc": "good_till_canceled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"immediate_or_cancel", "fill_or_kill", "good_till_canceled"}:
        raise ValueError(f"Unsupported time_in_force value: {value}")
    return normalized


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


def _compact_error_detail(value: str | None, *, max_length: int = 180) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split())
    if not compact:
        return None
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."


class KalshiExecutionMode(str, Enum):
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class KalshiExecutionConfig:
    mode: KalshiExecutionMode = KalshiExecutionMode.PAPER
    enable_live_trading: bool = False
    simulate_immediate_fills: bool = False
    shadow_fill_latency_seconds: float = 0.25
    enable_pre_submit_orderbook_check: bool = True
    skip_rest_orderbook_check_for_immediate_orders: bool = False
    enable_direct_trade_intent_handoff_in_live_mode: bool = False
    yes_probe_immediate_limit_cushion_cents: int = 0
    no_probe_immediate_limit_cushion_cents: int = 0
    pre_submit_orderbook_depth: int = 1
    probe_order_time_in_force: str = "good_till_canceled"
    probe_order_ttl_seconds: float = 2.0
    subaccount: int = 0
    reconcile_interval_seconds: float = 15.0
    order_reconcile_timeout_seconds: float = 10.0
    reconnect: KalshiReconnectConfig = KalshiReconnectConfig()
    log_dir: Path = Path("output/live/kalshi/execution")

    def __post_init__(self) -> None:
        if self.subaccount < 0:
            raise ValueError("subaccount must be non-negative")
        if self.shadow_fill_latency_seconds < 0:
            raise ValueError("shadow_fill_latency_seconds must be non-negative")
        if self.pre_submit_orderbook_depth <= 0:
            raise ValueError("pre_submit_orderbook_depth must be positive")
        if self.yes_probe_immediate_limit_cushion_cents < 0:
            raise ValueError("yes_probe_immediate_limit_cushion_cents must be non-negative")
        if self.no_probe_immediate_limit_cushion_cents < 0:
            raise ValueError("no_probe_immediate_limit_cushion_cents must be non-negative")
        if self.probe_order_ttl_seconds < 0:
            raise ValueError("probe_order_ttl_seconds must be non-negative")
        if self.reconcile_interval_seconds <= 0:
            raise ValueError("reconcile_interval_seconds must be positive")
        if self.order_reconcile_timeout_seconds <= 0:
            raise ValueError("order_reconcile_timeout_seconds must be positive")
        object.__setattr__(
            self,
            "probe_order_time_in_force",
            _normalize_time_in_force(self.probe_order_time_in_force),
        )

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
            shadow_fill_latency_seconds=float(
                _resolve_env_value(environment, "SHADOW_FILL_LATENCY_SECONDS") or 0.25
            ),
            enable_pre_submit_orderbook_check=_parse_bool(
                _resolve_env_value(environment, "ENABLE_PRE_SUBMIT_ORDERBOOK_CHECK") or "true"
            ),
            skip_rest_orderbook_check_for_immediate_orders=_parse_bool(
                _resolve_env_value(environment, "SKIP_REST_ORDERBOOK_CHECK_FOR_IMMEDIATE_ORDERS") or "false"
            ),
            enable_direct_trade_intent_handoff_in_live_mode=_parse_bool(
                _resolve_env_value(environment, "ENABLE_DIRECT_TRADE_INTENT_HANDOFF_IN_LIVE_MODE") or "false"
            ),
            yes_probe_immediate_limit_cushion_cents=int(
                _resolve_env_value(environment, "YES_PROBE_IMMEDIATE_LIMIT_CUSHION_CENTS") or 0
            ),
            no_probe_immediate_limit_cushion_cents=int(
                _resolve_env_value(environment, "NO_PROBE_IMMEDIATE_LIMIT_CUSHION_CENTS") or 0
            ),
            pre_submit_orderbook_depth=int(_resolve_env_value(environment, "PRE_SUBMIT_ORDERBOOK_DEPTH") or 1),
            probe_order_time_in_force=_normalize_time_in_force(
                _resolve_env_value(environment, "PROBE_ORDER_TIME_IN_FORCE") or "good_till_canceled"
            ),
            probe_order_ttl_seconds=float(_resolve_env_value(environment, "PROBE_ORDER_TTL_SECONDS") or 2.0),
            subaccount=int(_resolve_env_value(environment, "SUBACCOUNT") or 0),
            reconcile_interval_seconds=float(_resolve_env_value(environment, "RECONCILE_INTERVAL_SECONDS") or 15.0),
            order_reconcile_timeout_seconds=float(
                _resolve_env_value(environment, "ORDER_RECONCILE_TIMEOUT_SECONDS") or 10.0
            ),
        )


@dataclass(frozen=True)
class _SimulatedFillResolution:
    filled: bool
    fill_price_cents: int | None
    message: str


@dataclass(frozen=True)
class _OrderbookLevel:
    price_cents: int
    contracts: int


@dataclass(frozen=True)
class _LiveOrderbookSnapshot:
    checked_at: datetime
    yes_bids: tuple[_OrderbookLevel, ...]
    no_bids: tuple[_OrderbookLevel, ...]


@dataclass(frozen=True)
class _FeedQuoteContext:
    ticker_update_time: datetime | None
    received_at: datetime | None
    top_book_side: str | None
    top_book_price_cents: int | None
    top_book_contracts: int | None
    executable_ask_cents: int | None


@dataclass(frozen=True)
class _LiveOrderbookCheckResult:
    passed: bool
    reason: str | None
    checked_at: datetime
    top_book_side: str | None
    top_book_price_cents: int | None
    top_book_contracts: int | None
    executable_ask_cents: int | None
    feed_top_book_side: str | None
    feed_top_book_price_cents: int | None
    feed_top_book_contracts: int | None
    feed_executable_ask_cents: int | None
    ticker_update_time: datetime | None
    received_at: datetime | None
    ticker_update_to_check_ms: float | None
    received_to_check_ms: float | None
    orderbook_roundtrip_ms: float | None
    error: str | None = None
    check_source: str = "rest_orderbook"


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
    thesis_id: str | None
    tranche_index: int | None
    tranche_window: str | None
    tranche_reason: str | None
    lifecycle_state: str | None
    total_thesis_budget_dollars: float | None
    payout_if_yes_dollars: float | None
    payout_if_no_dollars: float | None
    expected_value_dollars: float | None
    worst_case_loss_dollars: float | None
    target_id: str | None = None
    attempt_index: int | None = None
    desired_contracts: int | None = None
    remaining_contracts_before_submit: int | None = None
    hard_max_price_cents: int | None = None
    retry_reason: str | None = None
    was_first_attempt: bool | None = None
    time_in_force: str | None = None


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
    thesis_id: str | None
    tranche_index: int | None
    tranche_window: str | None
    tranche_reason: str | None
    lifecycle_state: str | None
    total_thesis_budget_dollars: float | None
    payout_if_yes_dollars: float | None
    payout_if_no_dollars: float | None
    expected_value_dollars: float | None
    worst_case_loss_dollars: float | None
    target_id: str | None = None
    attempt_index: int | None = None
    desired_contracts: int | None = None
    remaining_contracts_before_submit: int | None = None
    hard_max_price_cents: int | None = None
    retry_reason: str | None = None
    was_first_attempt: bool | None = None
    time_in_force: str | None = None


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
        thesis_id=state.thesis_id,
        tranche_index=state.tranche_index,
        tranche_window=state.tranche_window,
        tranche_reason=state.tranche_reason,
        lifecycle_state=state.lifecycle_state,
        total_thesis_budget_dollars=state.total_thesis_budget_dollars,
        payout_if_yes_dollars=state.payout_if_yes_dollars,
        payout_if_no_dollars=state.payout_if_no_dollars,
        expected_value_dollars=state.expected_value_dollars,
        worst_case_loss_dollars=state.worst_case_loss_dollars,
        target_id=state.target_id,
        attempt_index=state.attempt_index,
        desired_contracts=state.desired_contracts,
        remaining_contracts_before_submit=state.remaining_contracts_before_submit,
        hard_max_price_cents=state.hard_max_price_cents,
        retry_reason=state.retry_reason,
        was_first_attempt=state.was_first_attempt,
        time_in_force=state.time_in_force,
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


def build_create_order_payload(
    intent: KalshiTradeIntent,
    *,
    subaccount: int | None = None,
    time_in_force: str = "immediate_or_cancel",
    expiration_ts: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": intent.ticker,
        "client_order_id": intent.decision_id,
        "side": intent.side.lower(),
        "action": "buy",
        "count": intent.contracts,
        "type": "limit",
        "time_in_force": time_in_force,
    }
    price_field = "yes_price" if intent.side.upper() == "YES" else "no_price"
    payload[price_field] = intent.max_acceptable_entry_price_cents
    if expiration_ts is not None:
        payload["expiration_ts"] = expiration_ts
    if subaccount is not None:
        payload["subaccount"] = subaccount
    return payload


def build_close_position_payload(
    position: KalshiPortfolioPosition,
    *,
    client_order_id: str,
    limit_price_cents: int,
    subaccount: int | None = None,
    time_in_force: str = "immediate_or_cancel",
    expiration_ts: int | None = None,
) -> dict[str, Any]:
    if position.side.upper() == "YES":
        close_side = "NO"
        price_field = "no_price"
    else:
        close_side = "YES"
        price_field = "yes_price"
    payload: dict[str, Any] = {
        "ticker": position.ticker,
        "client_order_id": client_order_id,
        "side": close_side.lower(),
        "action": "sell",
        "count": position.contracts,
        "type": "limit",
        "time_in_force": time_in_force,
        "reduce_only": True,
        price_field: limit_price_cents,
    }
    if expiration_ts is not None:
        payload["expiration_ts"] = expiration_ts
    if subaccount is not None:
        payload["subaccount"] = subaccount
    return payload


def _parse_orderbook_levels(payload: Any) -> tuple[_OrderbookLevel, ...]:
    if not isinstance(payload, list):
        return ()
    levels: list[_OrderbookLevel] = []
    for raw_level in payload:
        if not isinstance(raw_level, (list, tuple)) or len(raw_level) < 2:
            continue
        price_cents = parse_price_cents(raw_level[0])
        contracts = parse_count(None, raw_level[1])
        if price_cents is None or contracts <= 0:
            continue
        levels.append(_OrderbookLevel(price_cents=price_cents, contracts=contracts))
    return tuple(levels)


def parse_live_orderbook_snapshot(payload: dict[str, Any], *, checked_at: datetime | None = None) -> _LiveOrderbookSnapshot:
    checked_at = checked_at or utc_now()
    orderbook_payload = payload.get("orderbook_fp")
    if not isinstance(orderbook_payload, dict):
        orderbook_payload = payload.get("orderbook")
    if not isinstance(orderbook_payload, dict):
        orderbook_payload = {}
    yes_levels = orderbook_payload.get("yes_dollars")
    if yes_levels is None:
        yes_levels = orderbook_payload.get("yes")
    no_levels = orderbook_payload.get("no_dollars")
    if no_levels is None:
        no_levels = orderbook_payload.get("no")
    return _LiveOrderbookSnapshot(
        checked_at=checked_at,
        yes_bids=_parse_orderbook_levels(yes_levels),
        no_bids=_parse_orderbook_levels(no_levels),
    )


def _latency_ms(reference_time: datetime | None, target_time: datetime) -> float | None:
    if reference_time is None:
        return None
    return max(0.0, (target_time - reference_time).total_seconds() * 1000.0)


@dataclass(frozen=True)
class _LiveSubmissionPolicy:
    time_in_force: str
    expiration_ts: int | None
    is_probe_order: bool
    requires_immediate_match: bool


class KalshiExecutionEngine:
    def __init__(
        self,
        signal_engine: KalshiSignalRiskEngine,
        config: KalshiExecutionConfig | None = None,
        *,
        trade_intent_source: KalshiTradeIntentSource | None = None,
    ):
        self.signal_engine = signal_engine
        self.trade_intent_source = trade_intent_source or signal_engine
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
        self._intent_dispatch_lock = asyncio.Lock()
        self._direct_trade_intent_handoff_active = False
        self._direct_trade_intent_tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        if (
            self._direct_trade_intent_handoff_active
            or any(
                task is not None and not task.done()
                for task in (self._consume_task, self._ws_task, self._reconcile_task)
            )
        ):
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
        if self._should_use_direct_trade_intent_handoff():
            self._subscribe_direct_trade_intent_handoff()
        elif self._signal_queue is None:
            self._signal_queue = self.trade_intent_source.subscribe_trade_intent_queue()

        await self._logger.write(
            "execution_started",
            {
                "environment": self.environment.value,
                "mode": self.config.mode.value,
                "simulate_immediate_fills": self._simulation_enabled(),
                "shadow_fill_latency_seconds": self.config.shadow_fill_latency_seconds,
                "subaccount": self.config.subaccount,
                "direct_trade_intent_handoff_active": self._direct_trade_intent_handoff_active,
            },
        )
        if self.config.mode is KalshiExecutionMode.SHADOW:
            await self._refresh_portfolio_snapshot()
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="kalshi-execution-reconcile")
        elif self.config.mode is KalshiExecutionMode.PAPER or self._simulation_enabled():
            await self._sync_signal_portfolio_snapshot(
                event_time=utc_now(),
                log_event_type=f"{self._simulation_label()}_portfolio_initialized",
            )
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="kalshi-execution-reconcile")
        elif self.config.mode in {KalshiExecutionMode.LIVE, KalshiExecutionMode.SHADOW}:
            await self._refresh_portfolio_snapshot()
            if self.config.mode is KalshiExecutionMode.LIVE:
                await self._recover_recent_orders()
                self._ws_task = asyncio.create_task(self._private_ws_loop(), name="kalshi-execution-private-ws")
            self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="kalshi-execution-reconcile")

        if self.trade_intent_source is self.signal_engine:
            await self._bootstrap_from_signal_engine()
        if not self._direct_trade_intent_handoff_active:
            self._consume_task = asyncio.create_task(self._consume_loop(), name="kalshi-execution-consume")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        self._unsubscribe_direct_trade_intent_handoff()
        for task in tuple(self._direct_trade_intent_tasks):
            task.cancel()
        for task in (*tuple(self._direct_trade_intent_tasks), self._consume_task, self._ws_task, self._reconcile_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._direct_trade_intent_tasks.clear()
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
        close = getattr(self._logger, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

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

    def request_full_reconcile(self) -> None:
        self._request_reconcile(full=True)

    def _simulation_enabled(self) -> bool:
        return self.config.simulate_immediate_fills

    def _simulation_label(self) -> str:
        if self.config.mode is KalshiExecutionMode.PAPER:
            return "paper"
        if self.config.mode is KalshiExecutionMode.SHADOW:
            return "shadow"
        return "simulated"

    def _should_use_direct_trade_intent_handoff(self) -> bool:
        if self.config.mode is not KalshiExecutionMode.LIVE:
            return False
        if not self.config.enable_direct_trade_intent_handoff_in_live_mode:
            return False
        subscribe = getattr(self.trade_intent_source, "subscribe_trade_intent_callback", None)
        unsubscribe = getattr(self.trade_intent_source, "unsubscribe_trade_intent_callback", None)
        return callable(subscribe) and callable(unsubscribe)

    def _subscribe_direct_trade_intent_handoff(self) -> None:
        if self._direct_trade_intent_handoff_active:
            return
        subscribe = getattr(self.trade_intent_source, "subscribe_trade_intent_callback", None)
        if not callable(subscribe):
            return
        subscribe(self._handle_trade_intent_callback)
        self._direct_trade_intent_handoff_active = True

    def _unsubscribe_direct_trade_intent_handoff(self) -> None:
        if not self._direct_trade_intent_handoff_active:
            return
        unsubscribe = getattr(self.trade_intent_source, "unsubscribe_trade_intent_callback", None)
        if callable(unsubscribe):
            unsubscribe(self._handle_trade_intent_callback)
        self._direct_trade_intent_handoff_active = False

    def _handle_trade_intent_callback(self, intent: KalshiTradeIntent) -> None:
        if self._stop_event.is_set():
            return
        task = asyncio.create_task(
            self._consume_direct_trade_intent(intent, source="live_signal_direct"),
            name=f"kalshi-execution-direct-{intent.decision_id}",
        )
        self._direct_trade_intent_tasks.add(task)
        task.add_done_callback(self._direct_trade_intent_tasks.discard)

    async def _consume_direct_trade_intent(self, intent: KalshiTradeIntent, *, source: str) -> None:
        try:
            async with self._intent_dispatch_lock:
                await self._handle_trade_intent(intent, source=source)
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
                    "source": source,
                },
            )

    def _simulated_fill_latency_seconds(self) -> float:
        if self.config.mode is KalshiExecutionMode.SHADOW:
            return self.config.shadow_fill_latency_seconds
        return 0.0

    def _requires_quote_aware_simulated_fill(self) -> bool:
        return self.config.mode in {KalshiExecutionMode.PAPER, KalshiExecutionMode.SHADOW} or self._simulation_enabled()

    def _score_state_for_ticker(self, ticker: str) -> Any | None:
        get_state = getattr(self.signal_engine.scorer, "get_state", None)
        if callable(get_state):
            return get_state(ticker)
        snapshot_states = getattr(self.signal_engine.scorer, "snapshot_states", None)
        if callable(snapshot_states):
            return snapshot_states().get(ticker)
        return None

    def _resolve_simulated_executable_price(
        self,
        intent: KalshiTradeIntent,
    ) -> _SimulatedFillResolution:
        simulation_label = self._simulation_label()
        score_state = self._score_state_for_ticker(intent.ticker)
        if score_state is None:
            return _SimulatedFillResolution(
                filled=False,
                fill_price_cents=None,
                message=f"{simulation_label}_cancelled_missing_score_state",
            )
        if (
            score_state.yes_bid_cents is None
            or score_state.yes_ask_cents is None
            or score_state.buy_yes_price_cents is None
            or score_state.buy_no_price_cents is None
        ):
            return _SimulatedFillResolution(
                filled=False,
                fill_price_cents=None,
                message=f"{simulation_label}_cancelled_missing_quote",
            )
        if score_state.yes_bid_cents >= score_state.yes_ask_cents:
            return _SimulatedFillResolution(
                filled=False,
                fill_price_cents=None,
                message=f"{simulation_label}_cancelled_crossed_quote",
            )
        if (
            score_state.quote_age_seconds is None
            or score_state.quote_age_seconds > self.signal_engine.config.quote_max_age_seconds
        ):
            return _SimulatedFillResolution(
                filled=False,
                fill_price_cents=None,
                message=f"{simulation_label}_cancelled_stale_quote",
            )
        fill_price_cents = (
            score_state.buy_yes_price_cents if intent.side.upper() == "YES" else score_state.buy_no_price_cents
        )
        if fill_price_cents is None:
            return _SimulatedFillResolution(
                filled=False,
                fill_price_cents=None,
                message=f"{simulation_label}_cancelled_missing_executable_price",
            )
        if fill_price_cents > intent.max_acceptable_entry_price_cents:
            refreshed_max_acceptable_entry_price_cents = find_max_acceptable_entry_price_cents(
                side=intent.side.upper(),
                predicted_yes_probability=score_state.predicted_yes_probability,
                config=self.signal_engine.config,
                contracts=intent.contracts,
                edge_threshold_cents=self.signal_engine.config.maintain_edge_cents,
            )
            if (
                refreshed_max_acceptable_entry_price_cents is not None
                and fill_price_cents <= refreshed_max_acceptable_entry_price_cents
            ):
                return _SimulatedFillResolution(
                    filled=True,
                    fill_price_cents=fill_price_cents,
                    message=f"{simulation_label}_fill_requoted_edge_maintained",
                )
            return _SimulatedFillResolution(
                filled=False,
                fill_price_cents=None,
                message=f"{simulation_label}_cancelled_limit_moved_away",
            )
        return _SimulatedFillResolution(
            filled=True,
            fill_price_cents=fill_price_cents,
            message=f"{simulation_label}_fill_requoted",
        )

    def _current_quote_context(self, ticker: str, side: str) -> _FeedQuoteContext:
        collector_state = self._collector.get_state(ticker)
        if collector_state is not None:
            if side.upper() == "NO":
                return _FeedQuoteContext(
                    ticker_update_time=collector_state.ticker_update_time,
                    received_at=collector_state.received_at,
                    top_book_side="yes_bid",
                    top_book_price_cents=collector_state.yes_bid_cents,
                    top_book_contracts=collector_state.yes_bid_size,
                    executable_ask_cents=collector_state.buy_no_price_cents,
                )
            return _FeedQuoteContext(
                ticker_update_time=collector_state.ticker_update_time,
                received_at=collector_state.received_at,
                top_book_side="no_bid",
                top_book_price_cents=collector_state.no_bid_cents,
                top_book_contracts=collector_state.no_bid_size,
                executable_ask_cents=collector_state.buy_yes_price_cents,
            )
        score_state = self._score_state_for_ticker(ticker)
        if score_state is not None:
            executable_ask_cents = score_state.buy_no_price_cents if side.upper() == "NO" else score_state.buy_yes_price_cents
            top_book_side = "yes_bid" if side.upper() == "NO" else "no_bid"
            top_book_price_cents = score_state.yes_bid_cents if side.upper() == "NO" else score_state.no_bid_cents
            return _FeedQuoteContext(
                ticker_update_time=score_state.ticker_update_time,
                received_at=score_state.received_at,
                top_book_side=top_book_side,
                top_book_price_cents=top_book_price_cents,
                top_book_contracts=None,
                executable_ask_cents=executable_ask_cents,
            )
        return _FeedQuoteContext(
            ticker_update_time=None,
            received_at=None,
            top_book_side="yes_bid" if side.upper() == "NO" else "no_bid",
            top_book_price_cents=None,
            top_book_contracts=None,
            executable_ask_cents=None,
        )

    def _is_probe_order_intent(self, intent: KalshiTradeIntent) -> bool:
        return (
            intent.tranche_index == 0
            and intent.tranche_window == "10m"
            and intent.tranche_reason == "opened"
            and intent.lifecycle_state == "probe_pending"
        )

    def _build_live_submission_policy(
        self,
        intent: KalshiTradeIntent,
        *,
        submitted_at: datetime,
    ) -> _LiveSubmissionPolicy:
        if intent.target_id is not None:
            return _LiveSubmissionPolicy(
                time_in_force="immediate_or_cancel",
                expiration_ts=None,
                is_probe_order=self._is_probe_order_intent(intent),
                requires_immediate_match=True,
            )
        if not self._is_probe_order_intent(intent):
            return _LiveSubmissionPolicy(
                time_in_force="immediate_or_cancel",
                expiration_ts=None,
                is_probe_order=False,
                requires_immediate_match=True,
            )
        time_in_force = self.config.probe_order_time_in_force
        expiration_ts: int | None = None
        if time_in_force == "good_till_canceled" and self.config.probe_order_ttl_seconds > 0:
            expires_at = submitted_at + timedelta(seconds=self.config.probe_order_ttl_seconds)
            expiration_ts = max(math.ceil(expires_at.timestamp()), math.floor(submitted_at.timestamp()) + 1)
        return _LiveSubmissionPolicy(
            time_in_force=time_in_force,
            expiration_ts=expiration_ts,
            is_probe_order=True,
            requires_immediate_match=(time_in_force != "good_till_canceled"),
        )

    def _apply_live_limit_adjustments(
        self,
        intent: KalshiTradeIntent,
        *,
        submission_policy: _LiveSubmissionPolicy,
    ) -> tuple[KalshiTradeIntent, int]:
        limit_cushion_cents = 0
        if submission_policy.requires_immediate_match and submission_policy.is_probe_order:
            if intent.side.upper() == "YES":
                limit_cushion_cents = self.config.yes_probe_immediate_limit_cushion_cents
            elif intent.side.upper() == "NO":
                limit_cushion_cents = self.config.no_probe_immediate_limit_cushion_cents
        if limit_cushion_cents > 0:
            adjusted_limit_cents = min(
                99,
                intent.max_acceptable_entry_price_cents + limit_cushion_cents,
            )
            limit_adjustment_cents = max(0, adjusted_limit_cents - intent.max_acceptable_entry_price_cents)
            if limit_adjustment_cents > 0:
                adjusted_entry_cost_dollars, adjusted_fees_dollars, adjusted_cash_required_dollars = (
                    calculate_realized_cash_metrics(
                        entry_price_cents=adjusted_limit_cents,
                        contracts=intent.contracts,
                    )
                )
                return replace(
                    intent,
                    max_acceptable_entry_price_cents=adjusted_limit_cents,
                    estimated_entry_cost_dollars=adjusted_entry_cost_dollars,
                    estimated_fees_dollars=adjusted_fees_dollars,
                    estimated_cash_required_dollars=adjusted_cash_required_dollars,
                ), limit_adjustment_cents
        return intent, 0

    def _orderbook_check_log_payload(
        self,
        intent: KalshiTradeIntent,
        result: _LiveOrderbookCheckResult,
        *,
        policy: _LiveSubmissionPolicy | None = None,
        model_limit_price_cents: int | None = None,
        execution_limit_adjustment_cents: int = 0,
    ) -> dict[str, Any]:
        if model_limit_price_cents is None:
            model_limit_price_cents = intent.max_acceptable_entry_price_cents
        return {
            "decision_id": intent.decision_id,
            "ticker": intent.ticker,
            "side": intent.side,
            "contracts": intent.contracts,
            "limit_price_cents": intent.max_acceptable_entry_price_cents,
            "model_limit_price_cents": model_limit_price_cents,
            "execution_limit_adjustment_cents": execution_limit_adjustment_cents,
            "reference_price_cents": intent.reference_price_cents,
            "time_in_force": None if policy is None else policy.time_in_force,
            "expiration_ts": None if policy is None else policy.expiration_ts,
            "is_probe_order": None if policy is None else policy.is_probe_order,
            "requires_immediate_match": None if policy is None else policy.requires_immediate_match,
            "depth": self.config.pre_submit_orderbook_depth,
            "check_source": result.check_source,
            "passed": result.passed,
            "reason": result.reason,
            "checked_at": result.checked_at.isoformat(),
            "top_book_side": result.top_book_side,
            "top_book_price_cents": result.top_book_price_cents,
            "top_book_contracts": result.top_book_contracts,
            "current_executable_ask_cents": result.executable_ask_cents,
            "feed_top_book_side": result.feed_top_book_side,
            "feed_top_book_price_cents": result.feed_top_book_price_cents,
            "feed_top_book_contracts": result.feed_top_book_contracts,
            "feed_executable_ask_cents": result.feed_executable_ask_cents,
            "ticker_update_time": (
                result.ticker_update_time.isoformat() if result.ticker_update_time is not None else None
            ),
            "received_at": result.received_at.isoformat() if result.received_at is not None else None,
            "ticker_update_to_check_ms": result.ticker_update_to_check_ms,
            "received_at_to_check_ms": result.received_to_check_ms,
            "orderbook_roundtrip_ms": result.orderbook_roundtrip_ms,
            "error": result.error,
        }

    async def _check_live_orderbook_before_submit(
        self,
        intent: KalshiTradeIntent,
    ) -> _LiveOrderbookCheckResult:
        checked_at = utc_now()
        quote_context = self._current_quote_context(intent.ticker, intent.side)
        if not self.config.enable_pre_submit_orderbook_check:
            return _LiveOrderbookCheckResult(
                passed=True,
                reason="disabled",
                checked_at=checked_at,
                top_book_side=None,
                top_book_price_cents=None,
                top_book_contracts=None,
                executable_ask_cents=None,
                feed_top_book_side=quote_context.top_book_side,
                feed_top_book_price_cents=quote_context.top_book_price_cents,
                feed_top_book_contracts=quote_context.top_book_contracts,
                feed_executable_ask_cents=quote_context.executable_ask_cents,
                ticker_update_time=quote_context.ticker_update_time,
                received_at=quote_context.received_at,
                ticker_update_to_check_ms=_latency_ms(quote_context.ticker_update_time, checked_at),
                received_to_check_ms=_latency_ms(quote_context.received_at, checked_at),
                orderbook_roundtrip_ms=0.0,
                check_source="disabled",
            )

        request_started_at = utc_now()
        try:
            payload = await self._call_rest(
                self._rest_client.get_market_orderbook,
                intent.ticker,
                depth=self.config.pre_submit_orderbook_depth,
            )
        except Exception as exc:
            checked_at = utc_now()
            return _LiveOrderbookCheckResult(
                passed=False,
                reason="live_orderbook_check_failed",
                checked_at=checked_at,
                top_book_side=None,
                top_book_price_cents=None,
                top_book_contracts=None,
                executable_ask_cents=None,
                feed_top_book_side=quote_context.top_book_side,
                feed_top_book_price_cents=quote_context.top_book_price_cents,
                feed_top_book_contracts=quote_context.top_book_contracts,
                feed_executable_ask_cents=quote_context.executable_ask_cents,
                ticker_update_time=quote_context.ticker_update_time,
                received_at=quote_context.received_at,
                ticker_update_to_check_ms=_latency_ms(quote_context.ticker_update_time, checked_at),
                received_to_check_ms=_latency_ms(quote_context.received_at, checked_at),
                orderbook_roundtrip_ms=_latency_ms(request_started_at, checked_at),
                error=_compact_error_detail(repr(exc)),
                check_source="rest_orderbook",
            )

        checked_at = utc_now()
        snapshot = parse_live_orderbook_snapshot(payload, checked_at=checked_at)
        if intent.side.upper() == "NO":
            top_level = snapshot.yes_bids[0] if snapshot.yes_bids else None
            top_book_side = "yes_bid"
        else:
            top_level = snapshot.no_bids[0] if snapshot.no_bids else None
            top_book_side = "no_bid"
        if top_level is None:
            return _LiveOrderbookCheckResult(
                passed=False,
                reason="live_orderbook_missing_top_of_book",
                checked_at=checked_at,
                top_book_side=top_book_side,
                top_book_price_cents=None,
                top_book_contracts=None,
                executable_ask_cents=None,
                feed_top_book_side=quote_context.top_book_side,
                feed_top_book_price_cents=quote_context.top_book_price_cents,
                feed_top_book_contracts=quote_context.top_book_contracts,
                feed_executable_ask_cents=quote_context.executable_ask_cents,
                ticker_update_time=quote_context.ticker_update_time,
                received_at=quote_context.received_at,
                ticker_update_to_check_ms=_latency_ms(quote_context.ticker_update_time, checked_at),
                received_to_check_ms=_latency_ms(quote_context.received_at, checked_at),
                orderbook_roundtrip_ms=_latency_ms(request_started_at, checked_at),
                check_source="rest_orderbook",
            )

        executable_ask_cents = 100 - top_level.price_cents
        reason: str | None = None
        if executable_ask_cents > intent.max_acceptable_entry_price_cents:
            reason = "live_orderbook_limit_moved_away"
        elif top_level.contracts < intent.contracts:
            reason = "live_orderbook_insufficient_size"
        return _LiveOrderbookCheckResult(
            passed=reason is None,
            reason=reason,
            checked_at=checked_at,
            top_book_side=top_book_side,
            top_book_price_cents=top_level.price_cents,
            top_book_contracts=top_level.contracts,
            executable_ask_cents=executable_ask_cents,
            feed_top_book_side=quote_context.top_book_side,
            feed_top_book_price_cents=quote_context.top_book_price_cents,
            feed_top_book_contracts=quote_context.top_book_contracts,
            feed_executable_ask_cents=quote_context.executable_ask_cents,
            ticker_update_time=quote_context.ticker_update_time,
            received_at=quote_context.received_at,
            ticker_update_to_check_ms=_latency_ms(quote_context.ticker_update_time, checked_at),
            received_to_check_ms=_latency_ms(quote_context.received_at, checked_at),
            orderbook_roundtrip_ms=_latency_ms(request_started_at, checked_at),
            check_source="rest_orderbook",
        )

    def _check_feed_quote_before_submit(
        self,
        intent: KalshiTradeIntent,
    ) -> _LiveOrderbookCheckResult:
        checked_at = utc_now()
        quote_context = self._current_quote_context(intent.ticker, intent.side)
        reason: str | None = None
        if (
            quote_context.executable_ask_cents is not None
            and quote_context.executable_ask_cents > intent.max_acceptable_entry_price_cents
        ):
            reason = "feed_limit_moved_away"
        elif (
            quote_context.top_book_contracts is not None
            and quote_context.top_book_contracts < intent.contracts
        ):
            reason = "feed_insufficient_size"
        return _LiveOrderbookCheckResult(
            passed=reason is None,
            reason=reason,
            checked_at=checked_at,
            top_book_side=quote_context.top_book_side,
            top_book_price_cents=quote_context.top_book_price_cents,
            top_book_contracts=quote_context.top_book_contracts,
            executable_ask_cents=quote_context.executable_ask_cents,
            feed_top_book_side=quote_context.top_book_side,
            feed_top_book_price_cents=quote_context.top_book_price_cents,
            feed_top_book_contracts=quote_context.top_book_contracts,
            feed_executable_ask_cents=quote_context.executable_ask_cents,
            ticker_update_time=quote_context.ticker_update_time,
            received_at=quote_context.received_at,
            ticker_update_to_check_ms=_latency_ms(quote_context.ticker_update_time, checked_at),
            received_to_check_ms=_latency_ms(quote_context.received_at, checked_at),
            orderbook_roundtrip_ms=0.0,
            check_source="feed_quote",
        )

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

        execution_intent = intent
        limit_adjustment_cents = 0
        claimed_time_in_force = "immediate_or_cancel"
        if self.config.mode is KalshiExecutionMode.LIVE and not self._simulation_enabled():
            preview_submission_policy = self._build_live_submission_policy(
                intent,
                submitted_at=utc_now(),
            )
            execution_intent, limit_adjustment_cents = self._apply_live_limit_adjustments(
                intent,
                submission_policy=preview_submission_policy,
            )
            claimed_time_in_force = preview_submission_policy.time_in_force
        self.signal_engine.claim_trade_intent_reservation(execution_intent)

        claimed_state = KalshiExecutionIntentState(
            decision_id=execution_intent.decision_id,
            ticker=execution_intent.ticker,
            side=execution_intent.side,
            contracts=execution_intent.contracts,
            mode=self.config.mode,
            status="claimed",
            event_time=utc_now(),
            reference_price_cents=execution_intent.reference_price_cents,
            limit_price_cents=execution_intent.max_acceptable_entry_price_cents,
            client_order_id=execution_intent.decision_id,
            order_id=None,
            filled_contracts=0,
            remaining_contracts=execution_intent.contracts,
            fill_price_cents=None,
            entry_cost_dollars=execution_intent.estimated_entry_cost_dollars,
            fees_dollars=execution_intent.estimated_fees_dollars,
            cash_required_dollars=execution_intent.estimated_cash_required_dollars,
            available_cash_dollars=self._current_available_cash_dollars(),
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            settlement_result=None,
            message=source,
            live_order=None,
            thesis_id=execution_intent.thesis_id,
            tranche_index=execution_intent.tranche_index,
            tranche_window=execution_intent.tranche_window,
            tranche_reason=execution_intent.tranche_reason,
            lifecycle_state=execution_intent.lifecycle_state,
            total_thesis_budget_dollars=execution_intent.total_thesis_budget_dollars,
            payout_if_yes_dollars=execution_intent.payout_if_yes_dollars,
            payout_if_no_dollars=execution_intent.payout_if_no_dollars,
            expected_value_dollars=execution_intent.expected_value_dollars,
            worst_case_loss_dollars=execution_intent.worst_case_loss_dollars,
            target_id=execution_intent.target_id,
            attempt_index=execution_intent.attempt_index,
            desired_contracts=execution_intent.desired_contracts,
            remaining_contracts_before_submit=execution_intent.remaining_contracts_before_submit,
            hard_max_price_cents=execution_intent.hard_max_price_cents,
            retry_reason=execution_intent.retry_reason,
            was_first_attempt=execution_intent.was_first_attempt,
            time_in_force=claimed_time_in_force,
        )
        self._states[execution_intent.decision_id] = claimed_state
        self._client_order_to_decision[execution_intent.decision_id] = execution_intent.decision_id
        await self._logger.write(
            "intent_claimed",
            {
                "decision_id": execution_intent.decision_id,
                "ticker": execution_intent.ticker,
                "side": execution_intent.side,
                "source": source,
                "limit_price_cents": execution_intent.max_acceptable_entry_price_cents,
                "model_limit_price_cents": intent.max_acceptable_entry_price_cents,
                "execution_limit_adjustment_cents": limit_adjustment_cents,
                "target_id": execution_intent.target_id,
                "attempt_index": execution_intent.attempt_index,
                "desired_contracts": execution_intent.desired_contracts,
                "remaining_contracts_before_submit": execution_intent.remaining_contracts_before_submit,
                "hard_max_price_cents": execution_intent.hard_max_price_cents,
                "retry_reason": execution_intent.retry_reason,
                "was_first_attempt": execution_intent.was_first_attempt,
            },
        )
        if self.config.mode in {KalshiExecutionMode.PAPER, KalshiExecutionMode.SHADOW} or self._simulation_enabled():
            await self._sync_signal_portfolio_snapshot(
                event_time=claimed_state.event_time,
                log_event_type=f"{self._simulation_label()}_portfolio_claimed",
            )
        await self._publish_state(claimed_state)

        if self.config.mode in {KalshiExecutionMode.PAPER, KalshiExecutionMode.SHADOW} or self._simulation_enabled():
            await self._execute_simulated_intent(execution_intent)
            return

        await self._submit_live_intent(execution_intent, model_intent=intent)

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

        simulated_fill_price_cents = intent.reference_price_cents
        fill_message = f"{simulation_label}_fill"
        if self._requires_quote_aware_simulated_fill():
            latency_seconds = self._simulated_fill_latency_seconds()
            if latency_seconds > 0:
                await asyncio.sleep(latency_seconds)
            resolution = self._resolve_simulated_executable_price(intent)
            await self._logger.write(
                f"{simulation_label}_quote_recheck",
                {
                    "decision_id": intent.decision_id,
                    "ticker": intent.ticker,
                    "side": intent.side,
                    "reference_price_cents": intent.reference_price_cents,
                    "limit_price_cents": intent.max_acceptable_entry_price_cents,
                    "filled": resolution.filled,
                    "fill_price_cents": resolution.fill_price_cents,
                    "message": resolution.message,
                },
            )
            if not resolution.filled or resolution.fill_price_cents is None:
                await self.signal_engine.apply_execution_feedback(
                    KalshiExecutionFeedback(
                        decision_id=intent.decision_id,
                        status="cancelled",
                        event_time=utc_now(),
                    )
                )
                cancelled_state = replace(
                    accepted_state,
                    status="cancelled",
                    event_time=utc_now(),
                    available_cash_dollars=self.signal_engine.get_portfolio_state().available_cash_dollars,
                    message=resolution.message,
                )
                self._states[intent.decision_id] = cancelled_state
                await self._sync_signal_portfolio_snapshot(
                    event_time=cancelled_state.event_time,
                    log_event_type=f"{simulation_label}_portfolio_cancelled",
                )
                await self._publish_state(cancelled_state)
                return
            simulated_fill_price_cents = resolution.fill_price_cents
            fill_message = (
                f"{simulation_label}_fill"
                if simulated_fill_price_cents == intent.reference_price_cents
                else resolution.message
            )
            self.signal_engine.update_pending_reservation_fill_pricing(
                intent.decision_id,
                fill_price_cents=simulated_fill_price_cents,
            )

        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(
                decision_id=intent.decision_id,
                status="filled",
                event_time=utc_now(),
                filled_contracts=intent.contracts,
                filled_price_cents=simulated_fill_price_cents,
            )
        )
        entry_cost_dollars = accepted_state.entry_cost_dollars
        fees_dollars = accepted_state.fees_dollars
        cash_required_dollars = accepted_state.cash_required_dollars
        if self._requires_quote_aware_simulated_fill():
            entry_cost_dollars, fees_dollars, cash_required_dollars = calculate_realized_cash_metrics(
                entry_price_cents=simulated_fill_price_cents,
                contracts=intent.contracts,
            )
        filled_state = replace(
            accepted_state,
            status="filled",
            event_time=utc_now(),
            filled_contracts=intent.contracts,
            remaining_contracts=0,
            fill_price_cents=simulated_fill_price_cents,
            entry_cost_dollars=entry_cost_dollars,
            fees_dollars=fees_dollars,
            cash_required_dollars=cash_required_dollars,
            available_cash_dollars=self.signal_engine.get_portfolio_state().available_cash_dollars,
            message=fill_message,
        )
        self._states[intent.decision_id] = filled_state
        await self._sync_signal_portfolio_snapshot(
            event_time=filled_state.event_time,
            log_event_type=f"{simulation_label}_portfolio_filled",
        )
        await self._publish_state(filled_state)

    async def _execute_shadow_intent(self, intent: KalshiTradeIntent) -> None:
        await self._execute_simulated_intent(intent)

    async def _submit_live_intent(
        self,
        intent: KalshiTradeIntent,
        *,
        model_intent: KalshiTradeIntent | None = None,
    ) -> None:
        model_intent = model_intent or intent
        limit_adjustment_cents = max(
            0,
            intent.max_acceptable_entry_price_cents - model_intent.max_acceptable_entry_price_cents,
        )
        submit_requested_at = utc_now()
        submission_policy = self._build_live_submission_policy(
            intent,
            submitted_at=submit_requested_at,
        )
        if (
            submission_policy.requires_immediate_match
            and self.config.skip_rest_orderbook_check_for_immediate_orders
        ):
            orderbook_check = self._check_feed_quote_before_submit(intent)
        else:
            orderbook_check = await self._check_live_orderbook_before_submit(intent)
        await self._logger.write(
            "pre_submit_orderbook_check",
            self._orderbook_check_log_payload(
                intent,
                orderbook_check,
                policy=submission_policy,
                model_limit_price_cents=model_intent.max_acceptable_entry_price_cents,
                execution_limit_adjustment_cents=limit_adjustment_cents,
            ),
        )
        if submission_policy.requires_immediate_match and not orderbook_check.passed:
            await self._finalize_cancelled(intent.decision_id, orderbook_check.reason or "live_orderbook_blocked")
            return

        payload = build_create_order_payload(
            intent,
            subaccount=self.config.subaccount,
            time_in_force=submission_policy.time_in_force,
            expiration_ts=submission_policy.expiration_ts,
        )
        await self._logger.write(
            "submit_requested",
            {
                "decision_id": intent.decision_id,
                "ticker": intent.ticker,
                "side": intent.side,
                "contracts": intent.contracts,
                "limit_price_cents": intent.max_acceptable_entry_price_cents,
                "model_limit_price_cents": model_intent.max_acceptable_entry_price_cents,
                "execution_limit_adjustment_cents": limit_adjustment_cents,
                "reference_price_cents": intent.reference_price_cents,
                "target_id": intent.target_id,
                "attempt_index": intent.attempt_index,
                "desired_contracts": intent.desired_contracts,
                "remaining_contracts_before_submit": intent.remaining_contracts_before_submit,
                "hard_max_price_cents": intent.hard_max_price_cents,
                "retry_reason": intent.retry_reason,
                "was_first_attempt": intent.was_first_attempt,
                "subaccount": self.config.subaccount,
                "is_probe_order": submission_policy.is_probe_order,
                "time_in_force": submission_policy.time_in_force,
                "expiration_ts": submission_policy.expiration_ts,
                "requires_immediate_match": submission_policy.requires_immediate_match,
                "ticker_update_time": (
                    orderbook_check.ticker_update_time.isoformat()
                    if orderbook_check.ticker_update_time is not None
                    else None
                ),
                "received_at": (
                    orderbook_check.received_at.isoformat() if orderbook_check.received_at is not None else None
                ),
                "ticker_update_to_submit_requested_ms": _latency_ms(
                    orderbook_check.ticker_update_time,
                    submit_requested_at,
                ),
                "received_at_to_submit_requested_ms": _latency_ms(
                    orderbook_check.received_at,
                    submit_requested_at,
                ),
                "pre_submit_orderbook_checked_at": orderbook_check.checked_at.isoformat(),
                "pre_submit_orderbook_roundtrip_ms": orderbook_check.orderbook_roundtrip_ms,
                "pre_submit_check_source": orderbook_check.check_source,
                "current_executable_ask_cents": orderbook_check.executable_ask_cents,
                "top_book_side": orderbook_check.top_book_side,
                "top_book_price_cents": orderbook_check.top_book_price_cents,
                "top_book_contracts": orderbook_check.top_book_contracts,
                "feed_top_book_side": orderbook_check.feed_top_book_side,
                "feed_top_book_price_cents": orderbook_check.feed_top_book_price_cents,
                "feed_top_book_contracts": orderbook_check.feed_top_book_contracts,
                "feed_executable_ask_cents": orderbook_check.feed_executable_ask_cents,
                "payload": payload,
            },
            event_time=submit_requested_at,
        )
        try:
            response = await self._call_rest(self._rest_client.create_order, payload)
            await self._logger.write("submit_response", {"decision_id": intent.decision_id, "response": response})
        except httpx.HTTPStatusError as exc:
            response_detail = _compact_error_detail(exc.response.text)
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
                await self._finalize_rejected(intent.decision_id, "duplicate_client_order_id", detail=response_detail)
                return
            if 400 <= exc.response.status_code < 500:
                await self._finalize_rejected(
                    intent.decision_id,
                    f"http_{exc.response.status_code}",
                    detail=response_detail,
                )
                return
            if await self._reconcile_or_retry(intent, reason=f"http_{exc.response.status_code}"):
                return
            await self._finalize_error(
                intent.decision_id,
                f"http_{exc.response.status_code}",
                detail=response_detail,
            )
            return
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            error_detail = _compact_error_detail(repr(exc))
            await self._logger.write(
                "submit_error",
                {"decision_id": intent.decision_id, "error": repr(exc)},
            )
            if await self._reconcile_or_retry(intent, reason=type(exc).__name__):
                return
            await self._finalize_error(intent.decision_id, type(exc).__name__, detail=error_detail)
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

    async def cancel_live_intent(self, decision_id: str, *, reason: str) -> bool:
        state = self._states.get(decision_id)
        if state is None:
            return False
        if state.mode is not KalshiExecutionMode.LIVE or self._simulation_enabled():
            return False
        if state.status in {"cancelled", "rejected", "error", "filled", "settled"}:
            return False
        await self._logger.write(
            "cancel_requested",
            {
                "decision_id": decision_id,
                "order_id": state.order_id,
                "ticker": state.ticker,
                "side": state.side,
                "reason": reason,
                "target_id": state.target_id,
                "attempt_index": state.attempt_index,
            },
        )
        if state.order_id is None:
            await self._finalize_cancelled(decision_id, reason)
            return True
        try:
            response = await self._call_rest(
                self._rest_client.cancel_order,
                state.order_id,
                subaccount=self.config.subaccount,
            )
            await self._logger.write(
                "cancel_response",
                {
                    "decision_id": decision_id,
                    "order_id": state.order_id,
                    "reason": reason,
                    "response": response,
                },
            )
        except httpx.HTTPStatusError as exc:
            await self._logger.write(
                "cancel_error",
                {
                    "decision_id": decision_id,
                    "order_id": state.order_id,
                    "status_code": exc.response.status_code,
                    "response": exc.response.text,
                    "reason": reason,
                },
            )
            return False
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            await self._logger.write(
                "cancel_error",
                {
                    "decision_id": decision_id,
                    "order_id": state.order_id,
                    "error": repr(exc),
                    "reason": reason,
                },
            )
            return False

        order_payload = response.get("order")
        if isinstance(order_payload, dict) and order_payload:
            await self._apply_order_record(parse_live_order_record(order_payload), allow_signal_feedback=True)
        else:
            await self._finalize_cancelled(decision_id, reason)
        return True

    async def cancel_open_gtc_orders(self, *, reason: str) -> list[str]:
        cancelled_decision_ids: list[str] = []
        for decision_id, state in list(self._states.items()):
            if state.mode is not KalshiExecutionMode.LIVE or self._simulation_enabled():
                continue
            if state.status in {"cancelled", "rejected", "error", "filled", "settled"}:
                continue
            is_resting_gtc = state.time_in_force == "good_till_canceled"
            if state.live_order is not None and state.live_order.status == "resting":
                is_resting_gtc = True
            if not is_resting_gtc:
                continue
            if await self.cancel_live_intent(decision_id, reason=reason):
                cancelled_decision_ids.append(decision_id)
        return cancelled_decision_ids

    async def flatten_open_positions_if_rational(
        self,
        *,
        reason: str,
        settle_instead_of_close_tau_minutes: float,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        portfolio_state = self.signal_engine.get_portfolio_state()
        for position in portfolio_state.open_positions:
            score_state = self._score_state_for_ticker(position.ticker)
            collector_state = self._collector.get_state(position.ticker)
            tau_minutes: float | None = None
            if score_state is not None:
                tau_minutes = score_state.tau_minutes
            elif collector_state is not None and collector_state.close_time is not None:
                tau_minutes = max(0.0, (collector_state.close_time - utc_now()).total_seconds() / 60.0)
            if tau_minutes is not None and tau_minutes <= settle_instead_of_close_tau_minutes:
                actions.append(
                    {
                        "ticker": position.ticker,
                        "side": position.side,
                        "contracts": position.contracts,
                        "tau_minutes": tau_minutes,
                        "action": "settle",
                        "reason": "within_settlement_tau_window",
                    }
                )
                continue
            if collector_state is not None and not collector_state.is_open:
                actions.append(
                    {
                        "ticker": position.ticker,
                        "side": position.side,
                        "contracts": position.contracts,
                        "tau_minutes": tau_minutes,
                        "action": "settle",
                        "reason": "market_already_closed",
                    }
                )
                continue

            close_limit_price_cents: int | None
            if position.side.upper() == "YES":
                close_limit_price_cents = None if score_state is None else score_state.no_ask_cents
            else:
                close_limit_price_cents = None if score_state is None else score_state.yes_ask_cents
            if close_limit_price_cents is None:
                actions.append(
                    {
                        "ticker": position.ticker,
                        "side": position.side,
                        "contracts": position.contracts,
                        "tau_minutes": tau_minutes,
                        "action": "skip",
                        "reason": "no_closing_quote_available",
                    }
                )
                continue

            if self.config.mode is not KalshiExecutionMode.LIVE or self._simulation_enabled():
                actions.append(
                    {
                        "ticker": position.ticker,
                        "side": position.side,
                        "contracts": position.contracts,
                        "tau_minutes": tau_minutes,
                        "action": "settle",
                        "reason": "non_live_mode_close_not_submitted",
                    }
                )
                continue

            client_order_id = f"risk-close-{uuid.uuid4()}"
            payload = build_close_position_payload(
                position,
                client_order_id=client_order_id,
                limit_price_cents=close_limit_price_cents,
                subaccount=self.config.subaccount,
            )
            await self._logger.write(
                "risk_governor_flatten_requested",
                {
                    "reason": reason,
                    "ticker": position.ticker,
                    "side": position.side,
                    "contracts": position.contracts,
                    "tau_minutes": tau_minutes,
                    "client_order_id": client_order_id,
                    "limit_price_cents": close_limit_price_cents,
                    "payload": payload,
                },
            )
            try:
                response = await self._call_rest(self._rest_client.create_order, payload)
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
                error_detail = repr(exc)
                await self._logger.write(
                    "risk_governor_flatten_failed",
                    {
                        "reason": reason,
                        "ticker": position.ticker,
                        "side": position.side,
                        "contracts": position.contracts,
                        "tau_minutes": tau_minutes,
                        "client_order_id": client_order_id,
                        "limit_price_cents": close_limit_price_cents,
                        "error": error_detail,
                    },
                )
                actions.append(
                    {
                        "ticker": position.ticker,
                        "side": position.side,
                        "contracts": position.contracts,
                        "tau_minutes": tau_minutes,
                        "action": "skip",
                        "reason": "close_submission_failed",
                        "client_order_id": client_order_id,
                        "close_limit_price_cents": close_limit_price_cents,
                        "error": error_detail,
                    }
                )
                continue
            actions.append(
                {
                    "ticker": position.ticker,
                    "side": position.side,
                    "contracts": position.contracts,
                    "tau_minutes": tau_minutes,
                    "action": "close_submitted",
                    "reason": "close_order_submitted",
                    "client_order_id": client_order_id,
                    "close_limit_price_cents": close_limit_price_cents,
                    "response": response,
                }
            )

        if actions and self.config.mode is KalshiExecutionMode.LIVE and not self._simulation_enabled():
            self._request_reconcile(full=True)
        return actions

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

        retry_requested_at = utc_now()
        submission_policy = self._build_live_submission_policy(
            intent,
            submitted_at=retry_requested_at,
        )
        retry_payload = build_create_order_payload(
            intent,
            subaccount=self.config.subaccount,
            time_in_force=submission_policy.time_in_force,
            expiration_ts=submission_policy.expiration_ts,
        )
        try:
            response = await self._call_rest(
                self._rest_client.create_order,
                retry_payload,
            )
            await self._logger.write(
                "submit_retry_response",
                {
                    "decision_id": intent.decision_id,
                    "time_in_force": submission_policy.time_in_force,
                    "expiration_ts": submission_policy.expiration_ts,
                    "response": response,
                },
                event_time=retry_requested_at,
            )
        except httpx.HTTPStatusError as exc:
            await self._logger.write(
                "submit_retry_error",
                {
                    "decision_id": intent.decision_id,
                    "status_code": exc.response.status_code,
                    "response": exc.response.text,
                    "time_in_force": submission_policy.time_in_force,
                    "expiration_ts": submission_policy.expiration_ts,
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
                {
                    "decision_id": intent.decision_id,
                    "error": repr(exc),
                    "time_in_force": submission_policy.time_in_force,
                    "expiration_ts": submission_policy.expiration_ts,
                },
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

    async def _finalize_rejected(self, decision_id: str, reason: str, *, detail: str | None = None) -> None:
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(decision_id=decision_id, status="rejected", event_time=utc_now())
        )
        state = self._states[decision_id]
        rejected_state = replace(
            state,
            status="rejected",
            event_time=utc_now(),
            message=(reason if detail is None else f"{reason}: {detail}"),
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

    async def _finalize_error(self, decision_id: str, reason: str, *, detail: str | None = None) -> None:
        await self.signal_engine.apply_execution_feedback(
            KalshiExecutionFeedback(decision_id=decision_id, status="rejected", event_time=utc_now())
        )
        state = self._states[decision_id]
        error_state = replace(
            state,
            status="error",
            event_time=utc_now(),
            message=(reason if detail is None else f"{reason}: {detail}"),
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
                thesis_id=None,
                tranche_index=None,
                tranche_window=None,
                tranche_reason=None,
                lifecycle_state=None,
                total_thesis_budget_dollars=None,
                payout_if_yes_dollars=None,
                payout_if_no_dollars=None,
                expected_value_dollars=None,
                worst_case_loss_dollars=None,
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
            entry_cost_dollars = record.taker_fill_cost_dollars or record.maker_fill_cost_dollars or existing.entry_cost_dollars
            fees_dollars = (
                record.taker_fees_dollars + record.maker_fees_dollars
                if (record.taker_fees_dollars or record.maker_fees_dollars)
                else existing.fees_dollars
            )
            cash_required_dollars = entry_cost_dollars + fees_dollars
            if allow_signal_feedback:
                if record.displayed_price_cents is not None:
                    self.signal_engine.update_pending_reservation_fill_pricing(
                        decision_id,
                        fill_price_cents=record.displayed_price_cents,
                        contracts=record.fill_count,
                        entry_cost_dollars=entry_cost_dollars,
                        fees_dollars=fees_dollars,
                        cash_required_dollars=cash_required_dollars,
                    )
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
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
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
                if self.config.mode in {KalshiExecutionMode.PAPER, KalshiExecutionMode.SHADOW} or self._simulation_enabled():
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
                log_event_type=f"{self._simulation_label()}_portfolio_settled",
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
