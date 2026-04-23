from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.live.kalshi.jsonl_logger import JsonlEventLogger

if TYPE_CHECKING:
    from src.live.kalshi.execution import KalshiExecutionEngine, KalshiExecutionUpdate
    from src.live.kalshi.signal_risk import KalshiSignalRiskEngine

AlertCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SessionRiskGovernorConfig:
    max_session_drawdown_dollars: float = 15.0
    max_session_loss_dollars: float = 25.0
    max_consecutive_losses: int = 8
    max_positions_opened_per_hour: int = 24
    settle_instead_of_close_tau_minutes: float = 2.0
    log_dir: Path = Path("output/live/kalshi/risk_governor")

    def __post_init__(self) -> None:
        if self.max_session_drawdown_dollars <= 0:
            raise ValueError("max_session_drawdown_dollars must be positive")
        if self.max_session_loss_dollars <= 0:
            raise ValueError("max_session_loss_dollars must be positive")
        if self.max_consecutive_losses <= 0:
            raise ValueError("max_consecutive_losses must be positive")
        if self.max_positions_opened_per_hour <= 0:
            raise ValueError("max_positions_opened_per_hour must be positive")
        if self.settle_instead_of_close_tau_minutes < 0:
            raise ValueError("settle_instead_of_close_tau_minutes must be non-negative")


@dataclass(frozen=True)
class SessionRiskGovernorState:
    event_time: datetime
    running_realized_pnl_dollars: float = 0.0
    session_peak_pnl_dollars: float = 0.0
    session_drawdown_dollars: float = 0.0
    consecutive_losses: int = 0
    positions_opened_last_hour: int = 0
    halted: bool = False
    halt_reason: str | None = None
    halt_message: str | None = None
    halted_at: datetime | None = None
    resumed_at: datetime | None = None


@dataclass(frozen=True)
class SessionRiskGovernorHaltRecord:
    event_time: datetime
    reason: str
    message: str
    state: SessionRiskGovernorState
    cancelled_gtc_decision_ids: tuple[str, ...]
    flatten_actions: tuple[dict[str, Any], ...]


class SessionRiskGovernor:
    BLOCK_REASON = "session_risk_governor_halted"

    def __init__(self, config: SessionRiskGovernorConfig | None = None, *, environment: str = "production"):
        self.config = config or SessionRiskGovernorConfig()
        self._logger = JsonlEventLogger(self.config.log_dir, environment)
        self._signal_engine: KalshiSignalRiskEngine | None = None
        self._execution_engine: KalshiExecutionEngine | None = None
        self._state = SessionRiskGovernorState(event_time=utc_now())
        self._last_halt_record: SessionRiskGovernorHaltRecord | None = None
        self._positions_opened_timestamps: deque[datetime] = deque()
        self._observed_status_by_decision: dict[str, str] = {}
        self._absolute_cumulative_realized_pnl_dollars = 0.0
        self._session_pnl_baseline_dollars = 0.0
        self._state_lock = asyncio.Lock()
        self._halt_lock = asyncio.Lock()
        self._alert_callbacks: list[AlertCallback] = []

    def attach(self, *, signal_engine: KalshiSignalRiskEngine, execution_engine: KalshiExecutionEngine) -> None:
        self._signal_engine = signal_engine
        self._execution_engine = execution_engine
        signal_engine.register_external_blocker(self.block_reason_for_ticker)
        execution_engine.subscribe(self.handle_execution_update)

    async def close(self) -> None:
        if self._signal_engine is not None:
            self._signal_engine.unregister_external_blocker(self.block_reason_for_ticker)
        close = getattr(self._logger, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    def subscribe_alerts(self, callback: AlertCallback) -> None:
        self._alert_callbacks.append(callback)

    def get_state(self) -> SessionRiskGovernorState:
        return self._state

    def get_last_halt_record(self) -> SessionRiskGovernorHaltRecord | None:
        return self._last_halt_record

    def block_reason_for_ticker(self, _ticker: str) -> str | None:
        if self._state.halted:
            return self.BLOCK_REASON
        return None

    async def manual_resume(self, *, reason: str = "manual_resume") -> SessionRiskGovernorState:
        resumed_at = utc_now()
        async with self._halt_lock:
            async with self._state_lock:
                self._positions_opened_timestamps.clear()
                self._session_pnl_baseline_dollars = self._absolute_cumulative_realized_pnl_dollars
                self._state = SessionRiskGovernorState(
                    event_time=resumed_at,
                    running_realized_pnl_dollars=0.0,
                    session_peak_pnl_dollars=0.0,
                    session_drawdown_dollars=0.0,
                    consecutive_losses=0,
                    positions_opened_last_hour=0,
                    halted=False,
                    halt_reason=None,
                    halt_message=None,
                    halted_at=None,
                    resumed_at=resumed_at,
                )
            await self._logger.write(
                "session_risk_resumed",
                {
                    "reason": reason,
                    "running_realized_pnl_dollars": 0.0,
                    "absolute_cumulative_realized_pnl_dollars": self._absolute_cumulative_realized_pnl_dollars,
                },
                event_time=resumed_at,
            )
            if self._signal_engine is not None:
                await self._signal_engine.reevaluate_all_tickers()
        return self._state

    async def handle_execution_update(self, update: KalshiExecutionUpdate) -> None:
        trip_reason: str | None = None
        async with self._state_lock:
            previous_status = self._observed_status_by_decision.get(update.decision_id)
            self._observed_status_by_decision[update.decision_id] = update.status

            if (
                update.status in {"partially_filled", "filled"}
                and previous_status not in {"partially_filled", "filled", "settled"}
                and update.filled_contracts > 0
            ):
                self._record_position_opened(update.event_time)

            if update.status == "settled" and previous_status != "settled":
                self._apply_realized_pnl_update(update)

            self._state = SessionRiskGovernorState(
                event_time=update.event_time,
                running_realized_pnl_dollars=self._state.running_realized_pnl_dollars,
                session_peak_pnl_dollars=self._state.session_peak_pnl_dollars,
                session_drawdown_dollars=self._state.session_drawdown_dollars,
                consecutive_losses=self._state.consecutive_losses,
                positions_opened_last_hour=self._positions_opened_last_hour(update.event_time),
                halted=self._state.halted,
                halt_reason=self._state.halt_reason,
                halt_message=self._state.halt_message,
                halted_at=self._state.halted_at,
                resumed_at=self._state.resumed_at,
            )
            if not self._state.halted:
                trip_reason = self._trip_reason()

        if trip_reason is not None:
            await self._trip(trip_reason, update.event_time)

    def _record_position_opened(self, event_time: datetime) -> None:
        self._positions_opened_timestamps.append(event_time)
        self._prune_position_window(event_time)

    def _prune_position_window(self, event_time: datetime) -> None:
        cutoff = event_time - timedelta(hours=1)
        while self._positions_opened_timestamps and self._positions_opened_timestamps[0] < cutoff:
            self._positions_opened_timestamps.popleft()

    def _positions_opened_last_hour(self, event_time: datetime) -> int:
        self._prune_position_window(event_time)
        return len(self._positions_opened_timestamps)

    def _apply_realized_pnl_update(self, update: KalshiExecutionUpdate) -> None:
        realized_pnl = 0.0 if update.realized_pnl_dollars is None else update.realized_pnl_dollars
        if update.cumulative_realized_pnl_dollars is not None:
            self._absolute_cumulative_realized_pnl_dollars = update.cumulative_realized_pnl_dollars
        else:
            self._absolute_cumulative_realized_pnl_dollars += realized_pnl
        running_realized_pnl_dollars = (
            self._absolute_cumulative_realized_pnl_dollars - self._session_pnl_baseline_dollars
        )
        session_peak_pnl_dollars = max(self._state.session_peak_pnl_dollars, running_realized_pnl_dollars)
        session_drawdown_dollars = max(0.0, session_peak_pnl_dollars - running_realized_pnl_dollars)
        consecutive_losses = 0 if realized_pnl >= 0 else self._state.consecutive_losses + 1
        self._state = SessionRiskGovernorState(
            event_time=update.event_time,
            running_realized_pnl_dollars=running_realized_pnl_dollars,
            session_peak_pnl_dollars=session_peak_pnl_dollars,
            session_drawdown_dollars=session_drawdown_dollars,
            consecutive_losses=consecutive_losses,
            positions_opened_last_hour=self._positions_opened_last_hour(update.event_time),
            halted=self._state.halted,
            halt_reason=self._state.halt_reason,
            halt_message=self._state.halt_message,
            halted_at=self._state.halted_at,
            resumed_at=self._state.resumed_at,
        )

    def _trip_reason(self) -> str | None:
        if self._state.session_drawdown_dollars >= self.config.max_session_drawdown_dollars:
            return "max_session_drawdown_dollars"
        if self._state.running_realized_pnl_dollars <= -self.config.max_session_loss_dollars:
            return "max_session_loss_dollars"
        if self._state.consecutive_losses >= self.config.max_consecutive_losses:
            return "max_consecutive_losses"
        if self._state.positions_opened_last_hour >= self.config.max_positions_opened_per_hour:
            return "max_positions_opened_per_hour"
        return None

    def _trip_message(self, reason: str) -> str:
        if reason == "max_session_drawdown_dollars":
            return (
                f"Session halted: drawdown ${self._state.session_drawdown_dollars:.2f} "
                f">= ${self.config.max_session_drawdown_dollars:.2f}"
            )
        if reason == "max_session_loss_dollars":
            return (
                f"Session halted: running PnL ${self._state.running_realized_pnl_dollars:.2f} "
                f"<= -${self.config.max_session_loss_dollars:.2f}"
            )
        if reason == "max_consecutive_losses":
            return (
                f"Session halted: consecutive losses {self._state.consecutive_losses} "
                f">= {self.config.max_consecutive_losses}"
            )
        return (
            f"Session halted: positions opened in last hour {self._state.positions_opened_last_hour} "
            f">= {self.config.max_positions_opened_per_hour}"
        )

    async def _trip(self, reason: str, event_time: datetime) -> None:
        async with self._halt_lock:
            async with self._state_lock:
                if self._state.halted:
                    return
                message = self._trip_message(reason)
                self._state = SessionRiskGovernorState(
                    event_time=event_time,
                    running_realized_pnl_dollars=self._state.running_realized_pnl_dollars,
                    session_peak_pnl_dollars=self._state.session_peak_pnl_dollars,
                    session_drawdown_dollars=self._state.session_drawdown_dollars,
                    consecutive_losses=self._state.consecutive_losses,
                    positions_opened_last_hour=self._state.positions_opened_last_hour,
                    halted=True,
                    halt_reason=reason,
                    halt_message=message,
                    halted_at=event_time,
                    resumed_at=self._state.resumed_at,
                )
                state_snapshot = self._state

            cancelled_gtc_decision_ids: tuple[str, ...] = ()
            flatten_actions: tuple[dict[str, Any], ...] = ()
            if self._execution_engine is not None:
                cancelled_gtc_decision_ids = tuple(
                    await self._execution_engine.cancel_open_gtc_orders(reason=f"{self.BLOCK_REASON}:{reason}")
                )
                flatten_actions = tuple(
                    await self._execution_engine.flatten_open_positions_if_rational(
                        reason=f"{self.BLOCK_REASON}:{reason}",
                        settle_instead_of_close_tau_minutes=self.config.settle_instead_of_close_tau_minutes,
                    )
                )

            halt_record = SessionRiskGovernorHaltRecord(
                event_time=event_time,
                reason=reason,
                message=state_snapshot.halt_message or self._trip_message(reason),
                state=state_snapshot,
                cancelled_gtc_decision_ids=cancelled_gtc_decision_ids,
                flatten_actions=flatten_actions,
            )
            self._last_halt_record = halt_record
            payload = {
                "reason": halt_record.reason,
                "message": halt_record.message,
                "running_realized_pnl_dollars": halt_record.state.running_realized_pnl_dollars,
                "session_peak_pnl_dollars": halt_record.state.session_peak_pnl_dollars,
                "session_drawdown_dollars": halt_record.state.session_drawdown_dollars,
                "consecutive_losses": halt_record.state.consecutive_losses,
                "positions_opened_last_hour": halt_record.state.positions_opened_last_hour,
                "cancelled_gtc_decision_ids": list(cancelled_gtc_decision_ids),
                "flatten_actions": list(flatten_actions),
            }
            await self._logger.write("session_risk_alert", payload, event_time=event_time)
            await self._logger.write("session_risk_halt", payload, event_time=event_time)
            await self._emit_alert(payload)
            if self._signal_engine is not None:
                await self._signal_engine.reevaluate_all_tickers()

    async def _emit_alert(self, payload: dict[str, Any]) -> None:
        for callback in list(self._alert_callbacks):
            result = callback(payload)
            if inspect.isawaitable(result):
                await result
