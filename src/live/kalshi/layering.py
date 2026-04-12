from __future__ import annotations

import asyncio
import inspect
import json
import math
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.live.kalshi.execution import KalshiExecutionEngine, KalshiExecutionUpdate
from src.live.kalshi.jsonl_logger import JsonlEventLogger
from src.live.kalshi.signal_risk import (
    KalshiSignalDecisionState,
    KalshiSignalDecisionUpdate,
    KalshiSignalRiskEngine,
    KalshiTradeIntent,
    calculate_cost_metrics,
)
from src.live.kalshi.trade_intent_source import KalshiTradeIntentSource

LayeringCallback = Callable[["KalshiLayeringDecision"], Awaitable[None] | None]

WINDOW_ORDER: tuple[str, ...] = ("10m", "5m", "4m", "3m")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class KalshiPathDependentBinaryLayeringConfig:
    supported_series_prefixes: tuple[str, ...] = ("KXBTC15M",)
    tranche_schedule: tuple[float, float, float, float] = (0.20, 0.20, 0.30, 0.30)
    minimum_contracts_per_thesis: int = 5
    allow_full_flip_relayer: bool = True
    hold_to_settlement: bool = True
    enable_early_exit: bool = False
    recovery_enabled: bool = True
    recovery_lookback_hours: float = 8.0
    log_dir: Path = Path("output/live/kalshi/layering")

    def __post_init__(self) -> None:
        if not self.supported_series_prefixes:
            raise ValueError("supported_series_prefixes must not be empty")
        if len(self.tranche_schedule) != len(WINDOW_ORDER):
            raise ValueError("tranche_schedule must contain exactly four tranche percentages")
        if abs(sum(self.tranche_schedule) - 1.0) > 1e-9:
            raise ValueError("tranche_schedule must sum to 1.0")
        if self.minimum_contracts_per_thesis <= 0:
            raise ValueError("minimum_contracts_per_thesis must be positive")
        if self.recovery_lookback_hours <= 0:
            raise ValueError("recovery_lookback_hours must be positive")


@dataclass(frozen=True)
class KalshiBinaryTranche:
    thesis_id: str
    decision_id: str
    index: int
    side: str
    decision_window: str
    decision_reason: str
    requested_contracts: int
    filled_contracts: int
    entry_price_cents: int
    fill_price_cents: int | None
    cash_required_dollars: float
    fees_dollars: float
    entry_cost_dollars: float
    requested_at: datetime
    last_update_time: datetime
    status: str


@dataclass(frozen=True)
class KalshiBinaryThesisLedger:
    thesis_id: str
    ticker: str
    close_time: datetime
    lifecycle_state: str
    initial_signal_side: str
    latest_signal_side: str | None
    latest_predicted_yes_probability: float
    decision_windows_hit: tuple[str, ...]
    total_thesis_budget_dollars: float
    remaining_budget_dollars: float
    tranches: tuple[KalshiBinaryTranche, ...]
    payout_if_yes_dollars: float
    payout_if_no_dollars: float
    current_expected_value_dollars: float
    worst_case_loss_dollars: float
    next_eligible_decision_window: str | None
    opened_at: datetime
    updated_at: datetime
    settled_at: datetime | None = None
    settlement_result: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class KalshiLayeringWindowState:
    ticker: str
    event_time: datetime
    close_time: datetime | None
    tau_minutes: float
    window_label: str | None
    supported: bool
    observation_only: bool


@dataclass(frozen=True)
class KalshiLayeringDecision:
    event_time: datetime
    ticker: str
    thesis_id: str | None
    action: str
    status: str
    decision_window: str | None
    side: str | None
    tranche_index: int | None
    contracts: int | None
    entry_price_cents: int | None
    payout_if_yes_dollars: float | None
    payout_if_no_dollars: float | None
    expected_value_dollars: float | None
    worst_case_loss_dollars: float | None
    total_thesis_budget_dollars: float | None
    lifecycle_state: str | None
    message: str | None = None


class KalshiPathDependentBinaryLayeringEngine(KalshiTradeIntentSource):
    def __init__(
        self,
        signal_engine: KalshiSignalRiskEngine,
        config: KalshiPathDependentBinaryLayeringConfig | None = None,
        *,
        log_dir: Path | None = None,
    ):
        self.signal_engine = signal_engine
        self.config = config or KalshiPathDependentBinaryLayeringConfig()
        self._collector = self.signal_engine.scorer.feature_engine.collector
        self._execution_engine: KalshiExecutionEngine | None = None
        collector_config = self._collector.config
        self._logger = JsonlEventLogger(log_dir or self.config.log_dir, collector_config.environment.value)
        self._signal_queue: asyncio.Queue[KalshiSignalDecisionUpdate] | None = None
        self._execution_queue: asyncio.Queue[KalshiExecutionUpdate] | None = None
        self._trade_intent_queues: list[asyncio.Queue[KalshiTradeIntent]] = []
        self._callbacks: list[LayeringCallback] = []
        self._queues: list[asyncio.Queue[KalshiLayeringDecision]] = []
        self._signal_task: asyncio.Task[Any] | None = None
        self._execution_task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._ledgers: dict[str, KalshiBinaryThesisLedger] = {}
        self._active_thesis_by_ticker: dict[str, str] = {}
        self._thesis_by_decision_id: dict[str, str] = {}

    def bind_execution_engine(self, execution_engine: KalshiExecutionEngine) -> None:
        self._execution_engine = execution_engine

    async def start(self) -> None:
        if self._signal_task and not self._signal_task.done():
            return
        if self._execution_engine is None:
            raise RuntimeError("KalshiPathDependentBinaryLayeringEngine requires bind_execution_engine() before start().")
        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._signal_queue is None:
            self._signal_queue = self.signal_engine.subscribe_queue()
        if self._execution_queue is None:
            self._execution_queue = self._execution_engine.subscribe_queue()
        if self.config.recovery_enabled:
            await self._recover_ledgers()
        await self._bootstrap_from_signal_snapshot()
        self._signal_task = asyncio.create_task(self._consume_signal_loop(), name="kalshi-layering-signal")
        self._execution_task = asyncio.create_task(self._consume_execution_loop(), name="kalshi-layering-execution")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        for task in (self._signal_task, self._execution_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._signal_task = None
        self._execution_task = None
        close = getattr(self._logger, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def subscribe(self, callback: LayeringCallback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLayeringDecision]:
        queue: asyncio.Queue[KalshiLayeringDecision] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def subscribe_trade_intent_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiTradeIntent]:
        queue: asyncio.Queue[KalshiTradeIntent] = asyncio.Queue(maxsize=maxsize)
        self._trade_intent_queues.append(queue)
        return queue

    def snapshot_ledgers(self) -> dict[str, KalshiBinaryThesisLedger]:
        return dict(self._ledgers)

    def get_active_ledger(self, ticker: str) -> KalshiBinaryThesisLedger | None:
        thesis_id = self._active_thesis_by_ticker.get(ticker)
        return None if thesis_id is None else self._ledgers.get(thesis_id)

    def _tranche_fraction(self, decision_window: str) -> float:
        try:
            index = WINDOW_ORDER.index(decision_window)
        except ValueError as exc:
            raise ValueError(f"Unsupported decision window: {decision_window}") from exc
        return self.config.tranche_schedule[index]

    def _minimum_contracts_for_schedule(self) -> int:
        schedule_floor = max(
            math.ceil(1.0 / fraction)
            for fraction in self.config.tranche_schedule
            if fraction > 0.0
        )
        return max(self.config.minimum_contracts_per_thesis, schedule_floor)

    def _minimum_contracts_for_probe(
        self,
        *,
        side: str,
        predicted_yes_probability: float,
        entry_price_cents: int,
    ) -> int:
        minimum_contracts = self._minimum_contracts_for_schedule()
        probe_fraction = self._tranche_fraction("10m")
        _edge_1, _entry_1, _fees_1, one_contract_cash = calculate_cost_metrics(
            side=side,
            predicted_yes_probability=predicted_yes_probability,
            displayed_entry_price_cents=entry_price_cents,
            contracts=1,
            slippage=self.signal_engine.config.slippage,
        )
        contracts = max(1, minimum_contracts)
        while contracts < 512:
            _edge_n, _entry_n, _fees_n, thesis_cash = calculate_cost_metrics(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=entry_price_cents,
                contracts=contracts,
                slippage=self.signal_engine.config.slippage,
            )
            if thesis_cash * probe_fraction >= one_contract_cash - 1e-12:
                return contracts
            contracts += 1
        return contracts

    async def _bootstrap_from_signal_snapshot(self) -> None:
        for decision_state in self.signal_engine.snapshot_states().values():
            await self._handle_signal_update(_decision_update_from_state(decision_state))

    async def _consume_signal_loop(self) -> None:
        if self._signal_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                update = await self._signal_queue.get()
                await self._handle_signal_update(update)
        except asyncio.CancelledError:
            raise

    async def _consume_execution_loop(self) -> None:
        if self._execution_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                update = await self._execution_queue.get()
                await self._handle_execution_update(update)
        except asyncio.CancelledError:
            raise

    async def _handle_signal_update(self, update: KalshiSignalDecisionUpdate) -> None:
        window = self._window_state(update)
        if not window.supported:
            return

        ledger = self.get_active_ledger(update.ticker)
        if ledger is not None:
            ledger = replace(
                ledger,
                latest_signal_side=update.side if update.approved else ledger.latest_signal_side,
                latest_predicted_yes_probability=update.predicted_yes_probability,
                updated_at=update.event_time,
            )
            ledger = self._recompute_ledger_metrics(ledger)
            self._store_ledger(ledger)

        if window.window_label is None:
            return
        if ledger is None:
            await self._maybe_open_thesis(update, window)
            return
        if window.window_label in ledger.decision_windows_hit:
            return
        await self._maybe_add_tranche(ledger, update, window)

    async def _maybe_open_thesis(
        self,
        update: KalshiSignalDecisionUpdate,
        window: KalshiLayeringWindowState,
    ) -> None:
        if window.window_label != "10m":
            return
        if not update.approved or update.side is None:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=None,
                    action="no_signal",
                    status="skipped",
                    decision_window="10m",
                    side=None,
                    tranche_index=None,
                    contracts=None,
                    entry_price_cents=None,
                    payout_if_yes_dollars=None,
                    payout_if_no_dollars=None,
                    expected_value_dollars=None,
                    worst_case_loss_dollars=None,
                    total_thesis_budget_dollars=None,
                    lifecycle_state=None,
                    message="no approved signal at first-entry window",
                ),
                ledger=None,
            )
            return

        entry_price_cents = _entry_price_for_side(update, update.side)
        if entry_price_cents is None or window.close_time is None:
            return
        total_contracts = self.signal_engine.resolve_manual_trade_contracts(
            decision_state=update,
            side=update.side,
            entry_price_cents=entry_price_cents,
        )
        total_contracts = max(
            total_contracts,
            self._minimum_contracts_for_probe(
                side=update.side,
                predicted_yes_probability=update.predicted_yes_probability,
                entry_price_cents=entry_price_cents,
            ),
        )
        if total_contracts <= 0:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=None,
                    action="budget_blocked",
                    status="blocked",
                    decision_window="10m",
                    side=update.side,
                    tranche_index=0,
                    contracts=0,
                    entry_price_cents=entry_price_cents,
                    payout_if_yes_dollars=None,
                    payout_if_no_dollars=None,
                    expected_value_dollars=None,
                    worst_case_loss_dollars=None,
                    total_thesis_budget_dollars=None,
                    lifecycle_state=None,
                    message="total thesis budget could not afford one contract",
                ),
                ledger=None,
            )
            return

        _edge, _entry_cost, _fees, total_budget = calculate_cost_metrics(
            side=update.side,
            predicted_yes_probability=update.predicted_yes_probability,
            displayed_entry_price_cents=entry_price_cents,
            contracts=total_contracts,
            slippage=self.signal_engine.config.slippage,
        )
        thesis_id = uuid.uuid4().hex
        intent = self.signal_engine.reserve_manual_trade_intent(
            decision_state=update,
            side=update.side,
            entry_price_cents=entry_price_cents,
            cash_budget_dollars=total_budget * self._tranche_fraction("10m"),
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            stacking_signature=(thesis_id, "10m", update.side, "opened", None),
            thesis_id=thesis_id,
            tranche_index=0,
            tranche_window="10m",
            tranche_reason="opened",
            lifecycle_state="probe_pending",
            total_thesis_budget_dollars=total_budget,
        )
        if intent is None:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=thesis_id,
                    action="budget_blocked",
                    status="blocked",
                    decision_window="10m",
                    side=update.side,
                    tranche_index=0,
                    contracts=0,
                    entry_price_cents=entry_price_cents,
                    payout_if_yes_dollars=None,
                    payout_if_no_dollars=None,
                    expected_value_dollars=None,
                    worst_case_loss_dollars=None,
                    total_thesis_budget_dollars=total_budget,
                    lifecycle_state="probe_pending",
                    message="probe tranche could not be reserved",
                ),
                ledger=None,
            )
            return

        tranche = KalshiBinaryTranche(
            thesis_id=thesis_id,
            decision_id=intent.decision_id,
            index=0,
            side=intent.side,
            decision_window="10m",
            decision_reason="opened",
            requested_contracts=intent.contracts,
            filled_contracts=0,
            entry_price_cents=entry_price_cents,
            fill_price_cents=None,
            cash_required_dollars=intent.estimated_cash_required_dollars,
            fees_dollars=intent.estimated_fees_dollars,
            entry_cost_dollars=intent.estimated_entry_cost_dollars,
            requested_at=update.event_time,
            last_update_time=update.event_time,
            status="pending",
        )
        ledger = KalshiBinaryThesisLedger(
            thesis_id=thesis_id,
            ticker=update.ticker,
            close_time=window.close_time,
            lifecycle_state="probe_pending",
            initial_signal_side=update.side,
            latest_signal_side=update.side,
            latest_predicted_yes_probability=update.predicted_yes_probability,
            decision_windows_hit=("10m",),
            total_thesis_budget_dollars=total_budget,
            remaining_budget_dollars=0.0,
            tranches=(tranche,),
            payout_if_yes_dollars=0.0,
            payout_if_no_dollars=0.0,
            current_expected_value_dollars=0.0,
            worst_case_loss_dollars=0.0,
            next_eligible_decision_window="5m",
            opened_at=update.event_time,
            updated_at=update.event_time,
        )
        ledger = self._recompute_ledger_metrics(ledger)
        self._store_ledger(ledger)
        self._thesis_by_decision_id[intent.decision_id] = thesis_id
        for queue in self._trade_intent_queues:
            await queue.put(
                replace(
                    intent,
                    payout_if_yes_dollars=ledger.payout_if_yes_dollars,
                    payout_if_no_dollars=ledger.payout_if_no_dollars,
                    expected_value_dollars=ledger.current_expected_value_dollars,
                    worst_case_loss_dollars=ledger.worst_case_loss_dollars,
                )
            )
        await self._publish_decision(
            KalshiLayeringDecision(
                event_time=update.event_time,
                ticker=update.ticker,
                thesis_id=thesis_id,
                action="opened",
                status="emitted",
                decision_window="10m",
                side=intent.side,
                tranche_index=0,
                contracts=intent.contracts,
                entry_price_cents=entry_price_cents,
                payout_if_yes_dollars=ledger.payout_if_yes_dollars,
                payout_if_no_dollars=ledger.payout_if_no_dollars,
                expected_value_dollars=ledger.current_expected_value_dollars,
                worst_case_loss_dollars=ledger.worst_case_loss_dollars,
                total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                lifecycle_state=ledger.lifecycle_state,
            ),
            ledger=ledger,
        )

    async def _maybe_add_tranche(
        self,
        ledger: KalshiBinaryThesisLedger,
        update: KalshiSignalDecisionUpdate,
        window: KalshiLayeringWindowState,
    ) -> None:
        next_index = len(ledger.tranches)
        next_window = window.window_label or ""
        updated_windows = tuple([*ledger.decision_windows_hit, next_window])
        if not update.approved or update.side is None:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=ledger.thesis_id,
                    action="no_signal",
                    status="skipped",
                    decision_window=next_window,
                    side=None,
                    tranche_index=next_index,
                    contracts=None,
                    entry_price_cents=None,
                    payout_if_yes_dollars=ledger.payout_if_yes_dollars,
                    payout_if_no_dollars=ledger.payout_if_no_dollars,
                    expected_value_dollars=ledger.current_expected_value_dollars,
                    worst_case_loss_dollars=ledger.worst_case_loss_dollars,
                    total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                    lifecycle_state=ledger.lifecycle_state,
                    message="current signal not approved",
                ),
                ledger=replace(ledger, decision_windows_hit=updated_windows, updated_at=update.event_time),
            )
            return

        last_tranche = ledger.tranches[-1]
        action = "same_side_add" if update.side == last_tranche.side else "flip_add"
        if action == "flip_add" and not self.config.allow_full_flip_relayer:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=ledger.thesis_id,
                    action="flip_blocked",
                    status="blocked",
                    decision_window=next_window,
                    side=update.side,
                    tranche_index=next_index,
                    contracts=0,
                    entry_price_cents=None,
                    payout_if_yes_dollars=ledger.payout_if_yes_dollars,
                    payout_if_no_dollars=ledger.payout_if_no_dollars,
                    expected_value_dollars=ledger.current_expected_value_dollars,
                    worst_case_loss_dollars=ledger.worst_case_loss_dollars,
                    total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                    lifecycle_state=ledger.lifecycle_state,
                    message="opposite-side layering is disabled",
                ),
                ledger=replace(ledger, decision_windows_hit=updated_windows, updated_at=update.event_time),
            )
            return

        entry_price_cents = _entry_price_for_side(update, update.side)
        if entry_price_cents is None:
            return
        tranche_budget = min(
            ledger.total_thesis_budget_dollars * self._tranche_fraction(next_window),
            ledger.remaining_budget_dollars,
        )
        if tranche_budget <= 0:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=ledger.thesis_id,
                    action="budget_blocked",
                    status="blocked",
                    decision_window=next_window,
                    side=update.side,
                    tranche_index=next_index,
                    contracts=0,
                    entry_price_cents=entry_price_cents,
                    payout_if_yes_dollars=ledger.payout_if_yes_dollars,
                    payout_if_no_dollars=ledger.payout_if_no_dollars,
                    expected_value_dollars=ledger.current_expected_value_dollars,
                    worst_case_loss_dollars=ledger.worst_case_loss_dollars,
                    total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                    lifecycle_state=ledger.lifecycle_state,
                    message="no remaining thesis budget",
                ),
                ledger=replace(ledger, decision_windows_hit=updated_windows, updated_at=update.event_time),
            )
            return

        projected_contracts = self.signal_engine.resolve_manual_trade_contracts(
            decision_state=update,
            side=update.side,
            entry_price_cents=entry_price_cents,
            cash_budget_dollars=tranche_budget,
        )
        if projected_contracts <= 0:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=ledger.thesis_id,
                    action="budget_blocked",
                    status="blocked",
                    decision_window=next_window,
                    side=update.side,
                    tranche_index=next_index,
                    contracts=0,
                    entry_price_cents=entry_price_cents,
                    payout_if_yes_dollars=ledger.payout_if_yes_dollars,
                    payout_if_no_dollars=ledger.payout_if_no_dollars,
                    expected_value_dollars=ledger.current_expected_value_dollars,
                    worst_case_loss_dollars=ledger.worst_case_loss_dollars,
                    total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                    lifecycle_state=ledger.lifecycle_state,
                    message="tranche budget could not afford one contract",
                ),
                ledger=replace(ledger, decision_windows_hit=updated_windows, updated_at=update.event_time),
            )
            return

        projected_payout_yes, projected_payout_no = _apply_tranche_payout(
            payout_if_yes_dollars=ledger.payout_if_yes_dollars,
            payout_if_no_dollars=ledger.payout_if_no_dollars,
            side=update.side,
            contracts=projected_contracts,
            price_cents=entry_price_cents,
        )
        projected_ev = _expected_value(update.predicted_yes_probability, projected_payout_yes, projected_payout_no)
        projected_worst_case = _worst_case_loss(projected_payout_yes, projected_payout_no)
        if projected_ev <= 0.0 or projected_worst_case > ledger.total_thesis_budget_dollars + 1e-12:
            block_action = "ev_blocked" if projected_ev <= 0.0 else "loss_cap_blocked"
            block_message = (
                "projected EV would be non-positive"
                if projected_ev <= 0.0
                else "projected worst-case loss breaches thesis budget"
            )
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=ledger.thesis_id,
                    action=block_action,
                    status="blocked",
                    decision_window=next_window,
                    side=update.side,
                    tranche_index=next_index,
                    contracts=projected_contracts,
                    entry_price_cents=entry_price_cents,
                    payout_if_yes_dollars=projected_payout_yes,
                    payout_if_no_dollars=projected_payout_no,
                    expected_value_dollars=projected_ev,
                    worst_case_loss_dollars=projected_worst_case,
                    total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                    lifecycle_state=ledger.lifecycle_state,
                    message=block_message,
                ),
                ledger=replace(ledger, decision_windows_hit=updated_windows, updated_at=update.event_time),
            )
            return

        intent = self.signal_engine.reserve_manual_trade_intent(
            decision_state=update,
            side=update.side,
            entry_price_cents=entry_price_cents,
            cash_budget_dollars=tranche_budget,
            allow_ticker_lock_bypass=True,
            ignore_trade_cooldown=True,
            stacking_signature=(ledger.thesis_id, next_window, update.side, action, str(next_index)),
            thesis_id=ledger.thesis_id,
            tranche_index=next_index,
            tranche_window=next_window,
            tranche_reason=action,
            lifecycle_state="max_size_reached" if next_window == WINDOW_ORDER[-1] else "scaling_active",
            total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
            payout_if_yes_dollars=projected_payout_yes,
            payout_if_no_dollars=projected_payout_no,
            expected_value_dollars=projected_ev,
            worst_case_loss_dollars=projected_worst_case,
        )
        if intent is None:
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=update.event_time,
                    ticker=update.ticker,
                    thesis_id=ledger.thesis_id,
                    action="budget_blocked",
                    status="blocked",
                    decision_window=next_window,
                    side=update.side,
                    tranche_index=next_index,
                    contracts=projected_contracts,
                    entry_price_cents=entry_price_cents,
                    payout_if_yes_dollars=projected_payout_yes,
                    payout_if_no_dollars=projected_payout_no,
                    expected_value_dollars=projected_ev,
                    worst_case_loss_dollars=projected_worst_case,
                    total_thesis_budget_dollars=ledger.total_thesis_budget_dollars,
                    lifecycle_state=ledger.lifecycle_state,
                    message="tranche reservation failed after projection pass",
                ),
                ledger=replace(ledger, decision_windows_hit=updated_windows, updated_at=update.event_time),
            )
            return

        tranche = KalshiBinaryTranche(
            thesis_id=ledger.thesis_id,
            decision_id=intent.decision_id,
            index=next_index,
            side=intent.side,
            decision_window=next_window,
            decision_reason=action,
            requested_contracts=intent.contracts,
            filled_contracts=0,
            entry_price_cents=entry_price_cents,
            fill_price_cents=None,
            cash_required_dollars=intent.estimated_cash_required_dollars,
            fees_dollars=intent.estimated_fees_dollars,
            entry_cost_dollars=intent.estimated_entry_cost_dollars,
            requested_at=update.event_time,
            last_update_time=update.event_time,
            status="pending",
        )
        updated_ledger = replace(
            ledger,
            lifecycle_state="max_size_reached" if next_window == WINDOW_ORDER[-1] else "scaling_active",
            latest_signal_side=update.side,
            latest_predicted_yes_probability=update.predicted_yes_probability,
            decision_windows_hit=updated_windows,
            tranches=tuple([*ledger.tranches, tranche]),
            updated_at=update.event_time,
        )
        updated_ledger = self._recompute_ledger_metrics(updated_ledger)
        self._store_ledger(updated_ledger)
        self._thesis_by_decision_id[intent.decision_id] = ledger.thesis_id
        for queue in self._trade_intent_queues:
            await queue.put(intent)
        await self._publish_decision(
            KalshiLayeringDecision(
                event_time=update.event_time,
                ticker=update.ticker,
                thesis_id=ledger.thesis_id,
                action=action,
                status="emitted",
                decision_window=next_window,
                side=intent.side,
                tranche_index=next_index,
                contracts=intent.contracts,
                entry_price_cents=entry_price_cents,
                payout_if_yes_dollars=updated_ledger.payout_if_yes_dollars,
                payout_if_no_dollars=updated_ledger.payout_if_no_dollars,
                expected_value_dollars=updated_ledger.current_expected_value_dollars,
                worst_case_loss_dollars=updated_ledger.worst_case_loss_dollars,
                total_thesis_budget_dollars=updated_ledger.total_thesis_budget_dollars,
                lifecycle_state=updated_ledger.lifecycle_state,
            ),
            ledger=updated_ledger,
        )

    async def _handle_execution_update(self, update: KalshiExecutionUpdate) -> None:
        thesis_id = self._thesis_by_decision_id.get(update.decision_id)
        if thesis_id is None:
            return
        ledger = self._ledgers.get(thesis_id)
        if ledger is None:
            return
        tranches = list(ledger.tranches)
        tranche_index = None
        for idx, tranche in enumerate(tranches):
            if tranche.decision_id != update.decision_id:
                continue
            tranches[idx] = replace(
                tranche,
                filled_contracts=update.filled_contracts if update.filled_contracts > 0 else tranche.filled_contracts,
                fill_price_cents=update.fill_price_cents or tranche.fill_price_cents,
                last_update_time=update.event_time,
                status=update.status,
            )
            tranche_index = idx
            break
        if tranche_index is None:
            return
        updated_ledger = replace(
            ledger,
            tranches=tuple(tranches),
            lifecycle_state=("held_to_settlement" if update.status == "settled" else ledger.lifecycle_state),
            settled_at=(update.event_time if update.status == "settled" else ledger.settled_at),
            settlement_result=(update.settlement_result if update.status == "settled" else ledger.settlement_result),
            updated_at=update.event_time,
        )
        updated_ledger = self._recompute_ledger_metrics(updated_ledger)
        self._store_ledger(updated_ledger)
        await self._publish_decision(
            KalshiLayeringDecision(
                event_time=update.event_time,
                ticker=update.ticker,
                thesis_id=updated_ledger.thesis_id,
                action=f"execution_{update.status}",
                status="observed",
                decision_window=tranches[tranche_index].decision_window,
                side=update.side,
                tranche_index=tranches[tranche_index].index,
                contracts=update.filled_contracts or update.contracts,
                entry_price_cents=tranches[tranche_index].entry_price_cents,
                payout_if_yes_dollars=updated_ledger.payout_if_yes_dollars,
                payout_if_no_dollars=updated_ledger.payout_if_no_dollars,
                expected_value_dollars=updated_ledger.current_expected_value_dollars,
                worst_case_loss_dollars=updated_ledger.worst_case_loss_dollars,
                total_thesis_budget_dollars=updated_ledger.total_thesis_budget_dollars,
                lifecycle_state=updated_ledger.lifecycle_state,
            ),
            ledger=updated_ledger,
        )

    def _window_state(self, update: KalshiSignalDecisionUpdate) -> KalshiLayeringWindowState:
        collector_state = self._collector.get_state(update.ticker)
        close_time = collector_state.close_time if collector_state is not None else None
        tau_minutes = update.tau_minutes
        supported = any(update.ticker.startswith(prefix) for prefix in self.config.supported_series_prefixes)
        window_label: str | None = None
        if supported:
            if 5.0 < tau_minutes <= 10.0:
                window_label = "10m"
            elif 4.0 < tau_minutes <= 5.0:
                window_label = "5m"
            elif 3.0 < tau_minutes <= 4.0:
                window_label = "4m"
            elif 0.0 <= tau_minutes <= 3.0:
                window_label = "3m"
        return KalshiLayeringWindowState(
            ticker=update.ticker,
            event_time=update.event_time,
            close_time=close_time,
            tau_minutes=tau_minutes,
            window_label=window_label,
            supported=supported,
            observation_only=window_label is None,
        )

    def _recompute_ledger_metrics(self, ledger: KalshiBinaryThesisLedger) -> KalshiBinaryThesisLedger:
        payout_if_yes = 0.0
        payout_if_no = 0.0
        active_cash_required = 0.0
        for tranche in ledger.tranches:
            contracts = tranche.requested_contracts
            if tranche.status in {"rejected", "cancelled"} and tranche.filled_contracts <= 0:
                contracts = 0
            elif tranche.status in {"partially_filled", "filled", "settled"} and tranche.filled_contracts > 0:
                contracts = tranche.filled_contracts
            if contracts <= 0:
                continue
            price_cents = tranche.fill_price_cents or tranche.entry_price_cents
            payout_if_yes, payout_if_no = _apply_tranche_payout(
                payout_if_yes_dollars=payout_if_yes,
                payout_if_no_dollars=payout_if_no,
                side=tranche.side,
                contracts=contracts,
                price_cents=price_cents,
            )
            active_cash_required += tranche.cash_required_dollars * (contracts / max(1, tranche.requested_contracts))
        remaining_budget = max(0.0, ledger.total_thesis_budget_dollars - active_cash_required)
        next_window = next((window for window in WINDOW_ORDER if window not in ledger.decision_windows_hit), None)
        return replace(
            ledger,
            remaining_budget_dollars=remaining_budget,
            payout_if_yes_dollars=payout_if_yes,
            payout_if_no_dollars=payout_if_no,
            current_expected_value_dollars=_expected_value(
                ledger.latest_predicted_yes_probability,
                payout_if_yes,
                payout_if_no,
            ),
            worst_case_loss_dollars=_worst_case_loss(payout_if_yes, payout_if_no),
            next_eligible_decision_window=next_window,
        )

    def _store_ledger(self, ledger: KalshiBinaryThesisLedger) -> None:
        self._ledgers[ledger.thesis_id] = ledger
        if ledger.lifecycle_state in {"held_to_settlement", "closed_with_warning"}:
            self._active_thesis_by_ticker.pop(ledger.ticker, None)
            return
        self._active_thesis_by_ticker[ledger.ticker] = ledger.thesis_id

    async def _publish_decision(
        self,
        decision: KalshiLayeringDecision,
        *,
        ledger: KalshiBinaryThesisLedger | None,
    ) -> None:
        payload: dict[str, Any] = {"decision": asdict(decision)}
        if ledger is not None:
            self._store_ledger(ledger)
            payload["ledger"] = _serialize_ledger(ledger)
        await self._logger.write("layering_decision", payload, event_time=decision.event_time)
        for callback in self._callbacks:
            result = callback(decision)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(decision)

    async def _recover_ledgers(self) -> None:
        for ledger in self._load_latest_ledgers_from_journal().values():
            if ledger.lifecycle_state in {"held_to_settlement", "closed_with_warning"}:
                continue
            if self._ledger_is_recoverable(ledger):
                self._store_ledger(self._recompute_ledger_metrics(ledger))
                for tranche in ledger.tranches:
                    self._thesis_by_decision_id[tranche.decision_id] = ledger.thesis_id
                continue
            warning_ledger = replace(
                ledger,
                lifecycle_state="closed_with_warning",
                warning="ledger could not be confidently recovered",
                updated_at=utc_now(),
            )
            await self._publish_decision(
                KalshiLayeringDecision(
                    event_time=warning_ledger.updated_at,
                    ticker=warning_ledger.ticker,
                    thesis_id=warning_ledger.thesis_id,
                    action="closed_with_warning",
                    status="warning",
                    decision_window=None,
                    side=warning_ledger.latest_signal_side,
                    tranche_index=None,
                    contracts=None,
                    entry_price_cents=None,
                    payout_if_yes_dollars=warning_ledger.payout_if_yes_dollars,
                    payout_if_no_dollars=warning_ledger.payout_if_no_dollars,
                    expected_value_dollars=warning_ledger.current_expected_value_dollars,
                    worst_case_loss_dollars=warning_ledger.worst_case_loss_dollars,
                    total_thesis_budget_dollars=warning_ledger.total_thesis_budget_dollars,
                    lifecycle_state=warning_ledger.lifecycle_state,
                    message=warning_ledger.warning,
                ),
                ledger=warning_ledger,
            )

    def _ledger_is_recoverable(self, ledger: KalshiBinaryThesisLedger) -> bool:
        if self._execution_engine is None:
            return False
        execution_states = self._execution_engine.snapshot_states()
        if any(tranche.decision_id in execution_states for tranche in ledger.tranches):
            return True
        snapshot = self._execution_engine.get_portfolio_snapshot()
        if snapshot is not None and any(position.ticker == ledger.ticker for position in snapshot.open_positions):
            return True
        collector_state = self._collector.get_state(ledger.ticker)
        if collector_state is not None and collector_state.is_open:
            return True
        return False

    def _load_latest_ledgers_from_journal(self) -> dict[str, KalshiBinaryThesisLedger]:
        root = self.config.log_dir / self._collector.config.environment.value
        if not root.exists():
            return {}
        cutoff = utc_now() - timedelta(hours=self.config.recovery_lookback_hours)
        latest: dict[str, tuple[datetime, KalshiBinaryThesisLedger]] = {}
        for path in root.rglob("events.jsonl"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event_type") != "layering_decision":
                    continue
                payload = row.get("payload", {})
                ledger_payload = payload.get("ledger")
                if not isinstance(ledger_payload, dict):
                    continue
                ledger = _deserialize_ledger(ledger_payload)
                if ledger.updated_at < cutoff:
                    continue
                current = latest.get(ledger.thesis_id)
                if current is None or ledger.updated_at >= current[0]:
                    latest[ledger.thesis_id] = (ledger.updated_at, ledger)
        return {thesis_id: item[1] for thesis_id, item in latest.items()}


def _entry_price_for_side(update: KalshiSignalDecisionUpdate, side: str) -> int | None:
    if side == "YES":
        return update.buy_yes_price_cents
    return update.buy_no_price_cents


def _apply_tranche_payout(
    *,
    payout_if_yes_dollars: float,
    payout_if_no_dollars: float,
    side: str,
    contracts: int,
    price_cents: int,
) -> tuple[float, float]:
    price = price_cents / 100.0
    if side == "YES":
        return (
            payout_if_yes_dollars + (contracts * (1.0 - price)),
            payout_if_no_dollars - (contracts * price),
        )
    return (
        payout_if_yes_dollars - (contracts * price),
        payout_if_no_dollars + (contracts * (1.0 - price)),
    )


def _expected_value(predicted_yes_probability: float, payout_if_yes_dollars: float, payout_if_no_dollars: float) -> float:
    return (predicted_yes_probability * payout_if_yes_dollars) + (
        (1.0 - predicted_yes_probability) * payout_if_no_dollars
    )


def _worst_case_loss(payout_if_yes_dollars: float, payout_if_no_dollars: float) -> float:
    return max(0.0, -min(payout_if_yes_dollars, payout_if_no_dollars))


def _serialize_ledger(ledger: KalshiBinaryThesisLedger) -> dict[str, Any]:
    payload = asdict(ledger)
    payload["opened_at"] = ledger.opened_at.isoformat()
    payload["updated_at"] = ledger.updated_at.isoformat()
    payload["close_time"] = ledger.close_time.isoformat()
    payload["settled_at"] = None if ledger.settled_at is None else ledger.settled_at.isoformat()
    for tranche in payload["tranches"]:
        tranche["requested_at"] = tranche["requested_at"].isoformat()
        tranche["last_update_time"] = tranche["last_update_time"].isoformat()
    return payload


def _deserialize_ledger(payload: dict[str, Any]) -> KalshiBinaryThesisLedger:
    tranches = tuple(
        KalshiBinaryTranche(
            thesis_id=str(tranche["thesis_id"]),
            decision_id=str(tranche["decision_id"]),
            index=int(tranche["index"]),
            side=str(tranche["side"]),
            decision_window=str(tranche["decision_window"]),
            decision_reason=str(tranche["decision_reason"]),
            requested_contracts=int(tranche["requested_contracts"]),
            filled_contracts=int(tranche["filled_contracts"]),
            entry_price_cents=int(tranche["entry_price_cents"]),
            fill_price_cents=None if tranche.get("fill_price_cents") is None else int(tranche["fill_price_cents"]),
            cash_required_dollars=float(tranche["cash_required_dollars"]),
            fees_dollars=float(tranche["fees_dollars"]),
            entry_cost_dollars=float(tranche["entry_cost_dollars"]),
            requested_at=datetime.fromisoformat(tranche["requested_at"]),
            last_update_time=datetime.fromisoformat(tranche["last_update_time"]),
            status=str(tranche["status"]),
        )
        for tranche in payload.get("tranches", [])
    )
    return KalshiBinaryThesisLedger(
        thesis_id=str(payload["thesis_id"]),
        ticker=str(payload["ticker"]),
        close_time=datetime.fromisoformat(payload["close_time"]),
        lifecycle_state=str(payload["lifecycle_state"]),
        initial_signal_side=str(payload["initial_signal_side"]),
        latest_signal_side=None if payload.get("latest_signal_side") is None else str(payload["latest_signal_side"]),
        latest_predicted_yes_probability=float(payload["latest_predicted_yes_probability"]),
        decision_windows_hit=tuple(str(window) for window in payload.get("decision_windows_hit", [])),
        total_thesis_budget_dollars=float(payload["total_thesis_budget_dollars"]),
        remaining_budget_dollars=float(payload["remaining_budget_dollars"]),
        tranches=tranches,
        payout_if_yes_dollars=float(payload["payout_if_yes_dollars"]),
        payout_if_no_dollars=float(payload["payout_if_no_dollars"]),
        current_expected_value_dollars=float(payload["current_expected_value_dollars"]),
        worst_case_loss_dollars=float(payload["worst_case_loss_dollars"]),
        next_eligible_decision_window=payload.get("next_eligible_decision_window"),
        opened_at=datetime.fromisoformat(payload["opened_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
        settled_at=None if payload.get("settled_at") is None else datetime.fromisoformat(payload["settled_at"]),
        settlement_result=payload.get("settlement_result"),
        warning=payload.get("warning"),
    )


def _decision_update_from_state(state: KalshiSignalDecisionState) -> KalshiSignalDecisionUpdate:
    return KalshiSignalDecisionUpdate(
        ticker=state.ticker,
        event_time=state.event_time,
        approved=state.approved,
        side=state.side,
        predicted_yes_probability=state.predicted_yes_probability,
        predicted_no_probability=state.predicted_no_probability,
        feature_basis_market_prob=state.feature_basis_market_prob,
        raw_model_edge=state.raw_model_edge,
        post_cost_edge=state.post_cost_edge,
        yes_post_cost_edge=state.yes_post_cost_edge,
        no_post_cost_edge=state.no_post_cost_edge,
        tau_minutes=state.tau_minutes,
        reference_price_cents=state.reference_price_cents,
        max_acceptable_entry_price_cents=state.max_acceptable_entry_price_cents,
        last_yes_price_cents=state.last_yes_price_cents,
        yes_bid_cents=state.yes_bid_cents,
        yes_ask_cents=state.yes_ask_cents,
        buy_yes_price_cents=state.buy_yes_price_cents,
        buy_no_price_cents=state.buy_no_price_cents,
        quote_mid_prob=state.quote_mid_prob,
        quote_spread_cents=state.quote_spread_cents,
        quote_age_seconds=state.quote_age_seconds,
        regime_label=state.regime_label,
        bearish_vote_count=state.bearish_vote_count,
        bullish_vote_count=state.bullish_vote_count,
        regime_price_momentum_bearish=state.regime_price_momentum_bearish,
        regime_signed_flow_bearish=state.regime_signed_flow_bearish,
        regime_yes_share_bearish=state.regime_yes_share_bearish,
        regime_price_momentum_bullish=state.regime_price_momentum_bullish,
        regime_signed_flow_bullish=state.regime_signed_flow_bullish,
        regime_yes_share_bullish=state.regime_yes_share_bullish,
        tau_bucket=state.tau_bucket,
        price_bucket=state.price_bucket,
        chosen_side_probability_bucket=state.chosen_side_probability_bucket,
        chosen_side_edge_bucket=state.chosen_side_edge_bucket,
        bucket_policy_dimension=state.bucket_policy_dimension,
        bucket_policy_bucket=state.bucket_policy_bucket,
        bucket_policy_side=state.bucket_policy_side,
        block_reason=state.block_reason,
        trade_intent=state.trade_intent,
    )
