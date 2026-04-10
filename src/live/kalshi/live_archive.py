from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import zstandard as zstd

from src.live.kalshi.collector import KalshiMarketDataCollector
from src.live.kalshi.execution import KalshiExecutionUpdate
from src.live.kalshi.feature_engine import KalshiFeatureStateEngine
from src.live.kalshi.features import KalshiFeatureUpdate
from src.live.kalshi.layering import KalshiLayeringDecision
from src.live.kalshi.regime import evaluate_kxbtc15m_regime_for_state
from src.live.kalshi.scorer import KalshiLightGBMScoreUpdate
from src.live.kalshi.signal_risk import KalshiSignalDecisionUpdate
from src.live.kalshi.types import KalshiRawStreamEvent, KalshiTickerUpdate

ARCHIVE_STAGE_LAYERS = ("market_events", "feature_rows", "model_outputs", "strategy_events")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _partition_parts(event_time: datetime) -> tuple[str, str]:
    normalized = event_time.astimezone(UTC)
    return normalized.strftime("%Y-%m-%d"), normalized.strftime("%H")


def _market_event_id(update: KalshiTickerUpdate) -> str:
    if update.event_id:
        return update.event_id
    return f"market:{update.ticker}:{update.source}:{update.event_time.isoformat()}"


def _feature_row_id(update: KalshiFeatureUpdate) -> str:
    if update.event_id:
        return f"feature:{update.event_id}"
    return f"feature:{update.ticker}:{update.source}:{update.event_time.isoformat()}"


def _model_output_id(model_label: str, score_update: KalshiLightGBMScoreUpdate) -> str:
    if score_update.event_id:
        return f"model_output:{model_label}:{score_update.event_id}"
    return f"model_output:{model_label}:{score_update.ticker}:{score_update.event_time.isoformat()}"


def _append_jsonl_zst(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, default=_json_default) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        with zstd.ZstdCompressor(level=3).stream_writer(handle, closefd=False) as writer:
            writer.write(payload)


def _read_jsonl_zst(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        with zstd.ZstdDecompressor().stream_reader(handle) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for raw_line in text_reader:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


@dataclass(frozen=True)
class KalshiLiveArchiveConfig:
    archive_root: Path
    run_name: str
    environment: str
    compact_on_shutdown: bool = True


@dataclass(frozen=True)
class KalshiLiveArchiveRuntime:
    label: str
    family: str
    model_file: Path
    scorer: Any
    signal_engine: Any
    execution_engine: Any
    layering_engine: Any | None = None
    calibration_enabled: bool = False


class KalshiLiveArchiveManager:
    def __init__(
        self,
        collector: KalshiMarketDataCollector,
        feature_engine: KalshiFeatureStateEngine,
        runtimes: list[KalshiLiveArchiveRuntime],
        config: KalshiLiveArchiveConfig,
    ):
        self.collector = collector
        self.feature_engine = feature_engine
        self.runtimes = list(runtimes)
        self.config = config
        self._session_id = uuid.uuid4().hex[:12]
        self._write_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._raw_queue: asyncio.Queue[KalshiRawStreamEvent] | None = None
        self._market_queue: asyncio.Queue[KalshiTickerUpdate] | None = None
        self._feature_queue: asyncio.Queue[KalshiFeatureUpdate] | None = None
        self._runtime_queues: list[tuple[KalshiLiveArchiveRuntime, asyncio.Queue[Any], str]] = []
        self._last_compacted_hour_key: str | None = None

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop_event = asyncio.Event()
        self._raw_queue = self.collector.subscribe_raw_stream_queue()
        self._market_queue = self.collector.subscribe_queue()
        self._feature_queue = self.feature_engine.subscribe_queue()
        self._runtime_queues = []
        for runtime in self.runtimes:
            self._runtime_queues.extend(
                [
                    (runtime, runtime.scorer.subscribe_queue(), "model_outputs"),
                    (runtime, runtime.signal_engine.subscribe_queue(), "signal"),
                    (runtime, runtime.execution_engine.subscribe_queue(), "execution"),
                ]
            )
            if runtime.layering_engine is not None:
                self._runtime_queues.append((runtime, runtime.layering_engine.subscribe_queue(), "layering"))
        await self.compact_all_staging()
        self._tasks = [
            asyncio.create_task(self._consume_raw_loop(), name="kalshi-live-archive-raw"),
            asyncio.create_task(self._consume_market_loop(), name="kalshi-live-archive-market"),
            asyncio.create_task(self._consume_feature_loop(), name="kalshi-live-archive-feature"),
        ]
        for runtime, queue, queue_kind in self._runtime_queues:
            self._tasks.append(
                asyncio.create_task(
                    self._consume_runtime_loop(runtime, queue, queue_kind),
                    name=f"kalshi-live-archive-{runtime.label}-{queue_kind}",
                )
            )

    async def stop(self) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        if self.config.compact_on_shutdown:
            try:
                await self.compact_all_staging()
            except Exception as exc:  # pragma: no cover - defensive live path
                print(f"[live-archive] compaction on shutdown failed: {exc!r}")

    async def _consume_raw_loop(self) -> None:
        if self._raw_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                event = await self._raw_queue.get()
                await self._write_raw_stream_event(event)
        except asyncio.CancelledError:
            raise

    async def _consume_market_loop(self) -> None:
        if self._market_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                update = await self._market_queue.get()
                await self._write_market_event(update)
        except asyncio.CancelledError:
            raise

    async def _consume_feature_loop(self) -> None:
        if self._feature_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                update = await self._feature_queue.get()
                await self._write_feature_row(update)
        except asyncio.CancelledError:
            raise

    async def _consume_runtime_loop(
        self,
        runtime: KalshiLiveArchiveRuntime,
        queue: asyncio.Queue[Any],
        queue_kind: str,
    ) -> None:
        try:
            while not self._stop_event.is_set():
                update = await queue.get()
                if queue_kind == "model_outputs":
                    await self._write_model_output(runtime, update)
                elif queue_kind == "signal":
                    await self._write_signal_decision_event(runtime, update)
                elif queue_kind == "execution":
                    await self._write_execution_event(runtime, update)
                elif queue_kind == "layering":
                    await self._write_layering_event(runtime, update)
        except asyncio.CancelledError:
            raise

    async def _write_raw_stream_event(self, event: KalshiRawStreamEvent) -> None:
        event_time = event.received_at
        date_part, hour_part = _partition_parts(event_time)
        path = (
            self.config.archive_root
            / "raw_ws"
            / f"environment={self.config.environment}"
            / f"date={date_part}"
            / f"hour={hour_part}"
            / f"{event.channel}-{self._session_id}.jsonl.zst"
        )
        row = {
            "raw_event_id": event.raw_event_id,
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "channel": event.channel,
            "received_at": _iso(event.received_at),
            "exchange_event_time": _iso(event.exchange_event_time),
            "market_ticker": event.market_ticker,
            "session_id": event.session_id,
            "message_index": event.message_index,
            "payload": event.payload,
        }
        async with self._write_lock:
            _append_jsonl_zst(path, row)

    async def _write_market_event(self, update: KalshiTickerUpdate) -> None:
        market = self.collector.get_market(update.ticker)
        event_time = update.received_at or update.event_time
        row = {
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "market_event_id": _market_event_id(update),
            "raw_event_id": update.raw_event_id,
            "ticker": update.ticker,
            "event_ticker": market.event_ticker if market is not None else None,
            "source": update.source,
            "event_time": _iso(update.event_time),
            "received_at": _iso(update.received_at),
            "exchange_event_time": _iso(update.event_time),
            "is_open": update.is_open,
            "result": market.result if market is not None else None,
            "last_yes_price_cents": update.last_yes_price_cents,
            "last_price_cents": update.last_price_cents,
            "yes_bid_cents": update.yes_bid_cents,
            "yes_ask_cents": update.yes_ask_cents,
            "no_bid_cents": update.no_bid_cents,
            "no_ask_cents": update.no_ask_cents,
            "quote_mid_prob": (
                (update.yes_bid_cents + update.yes_ask_cents) / 200.0
                if update.yes_bid_cents is not None and update.yes_ask_cents is not None
                else (
                    (200.0 - update.no_bid_cents - update.no_ask_cents) / 200.0
                    if update.no_bid_cents is not None and update.no_ask_cents is not None
                    else None
                )
            ),
            "quote_spread_cents": (
                update.yes_ask_cents - update.yes_bid_cents
                if update.yes_bid_cents is not None and update.yes_ask_cents is not None
                else (
                    update.no_ask_cents - update.no_bid_cents
                    if update.no_bid_cents is not None and update.no_ask_cents is not None
                    else None
                )
            ),
            "buy_yes_price_cents": (
                update.yes_ask_cents
                if update.yes_ask_cents is not None
                else (100 - update.no_bid_cents if update.no_bid_cents is not None else None)
            ),
            "buy_no_price_cents": (
                update.no_ask_cents
                if update.no_ask_cents is not None
                else (100 - update.yes_bid_cents if update.yes_bid_cents is not None else None)
            ),
            "quote_age_seconds": (
                max(0.0, (update.event_time - update.ticker_update_time).total_seconds())
                if update.ticker_update_time is not None
                else None
            ),
            "trade_id": update.trade_id,
            "count": update.count,
            "taker_side": update.taker_side,
            "volume": update.volume,
            "open_interest": update.open_interest,
            "dollar_volume": update.dollar_volume,
            "dollar_open_interest": update.dollar_open_interest,
            "open_time": _iso(update.open_time),
            "close_time": _iso(update.close_time),
        }
        await self._write_stage_row("market_events", event_time, row)

    async def _write_feature_row(self, update: KalshiFeatureUpdate) -> None:
        event_time = update.received_at or update.event_time
        regime = evaluate_kxbtc15m_regime_for_state(update)
        row = {
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "feature_row_id": _feature_row_id(update),
            "market_event_id": update.event_id,
            "raw_event_id": update.raw_event_id,
            "ticker": update.ticker,
            "event_time": _iso(update.event_time),
            "received_at": _iso(update.received_at),
            "source": update.source,
            "market_prob": update.market_prob,
            "previous_market_prob": update.previous_market_prob,
            "tau_minutes": update.tau_minutes,
            "trade_yes_prob": update.trade_yes_prob,
            "quote_mid_prob": update.quote_mid_prob,
            "quote_spread_cents": update.quote_spread_cents,
            "buy_yes_price_cents": update.buy_yes_price_cents,
            "buy_no_price_cents": update.buy_no_price_cents,
            "quote_age_seconds": update.quote_age_seconds,
            "last_to_mid_gap": update.last_to_mid_gap,
            "last_yes_price_cents": update.last_yes_price_cents,
            "last_price_cents": update.last_price_cents,
            "yes_bid_cents": update.yes_bid_cents,
            "yes_ask_cents": update.yes_ask_cents,
            "no_bid_cents": update.no_bid_cents,
            "no_ask_cents": update.no_ask_cents,
            "schema_version": "kalshi_feature_row_v1",
            "in_training_window": update.in_training_window,
            "is_scoreable": update.is_scoreable,
            "regime_label": regime.regime_label,
            "bearish_vote_count": regime.bearish_vote_count,
            "bullish_vote_count": regime.bullish_vote_count,
            "regime_price_momentum_bearish": regime.price_momentum_bearish,
            "regime_signed_flow_bearish": regime.signed_flow_bearish,
            "regime_yes_share_bearish": regime.yes_share_bearish,
            "regime_price_momentum_bullish": regime.price_momentum_bullish,
            "regime_signed_flow_bullish": regime.signed_flow_bullish,
            "regime_yes_share_bullish": regime.yes_share_bullish,
        }
        row.update(update.feature_lookup())
        await self._write_stage_row("feature_rows", event_time, row)

    async def _write_model_output(
        self,
        runtime: KalshiLiveArchiveRuntime,
        update: KalshiLightGBMScoreUpdate,
    ) -> None:
        event_time = update.received_at or update.event_time
        regime = evaluate_kxbtc15m_regime_for_state(update)
        row = {
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "model_output_id": _model_output_id(runtime.label, update),
            "feature_row_id": f"feature:{update.event_id}" if update.event_id else "",
            "market_event_id": update.event_id,
            "raw_event_id": update.raw_event_id,
            "ticker": update.ticker,
            "event_time": _iso(update.event_time),
            "received_at": _iso(update.received_at),
            "model_label": runtime.label,
            "model_family": runtime.family,
            "model_file": str(runtime.model_file),
            "calibration_enabled": runtime.calibration_enabled,
            "predicted_yes_probability": update.predicted_yes_probability,
            "predicted_no_probability": 1.0 - update.predicted_yes_probability,
            "market_prob": update.market_prob,
            "tau_minutes": update.tau_minutes,
            "model_edge": update.model_edge,
            "last_yes_price_cents": update.last_yes_price_cents,
            "last_price_cents": update.last_price_cents,
            "yes_bid_cents": update.yes_bid_cents,
            "yes_ask_cents": update.yes_ask_cents,
            "no_bid_cents": update.no_bid_cents,
            "no_ask_cents": update.no_ask_cents,
            "buy_yes_price_cents": update.buy_yes_price_cents,
            "buy_no_price_cents": update.buy_no_price_cents,
            "quote_mid_prob": update.quote_mid_prob,
            "quote_spread_cents": update.quote_spread_cents,
            "quote_age_seconds": update.quote_age_seconds,
            "trade_yes_prob": update.trade_yes_prob,
            "last_to_mid_gap": update.last_to_mid_gap,
            "source": update.source,
            "regime_label": regime.regime_label,
            "bearish_vote_count": regime.bearish_vote_count,
            "bullish_vote_count": regime.bullish_vote_count,
            "regime_price_momentum_bearish": regime.price_momentum_bearish,
            "regime_signed_flow_bearish": regime.signed_flow_bearish,
            "regime_yes_share_bearish": regime.yes_share_bearish,
            "regime_price_momentum_bullish": regime.price_momentum_bullish,
            "regime_signed_flow_bullish": regime.signed_flow_bullish,
            "regime_yes_share_bullish": regime.yes_share_bullish,
        }
        await self._write_stage_row("model_outputs", event_time, row)

    async def _write_signal_decision_event(
        self,
        runtime: KalshiLiveArchiveRuntime,
        update: KalshiSignalDecisionUpdate,
    ) -> None:
        decision_id = None if update.trade_intent is None else update.trade_intent.decision_id
        event_kind = "signal_approved" if update.approved else "signal_blocked"
        row = {
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "model_label": runtime.label,
            "model_family": runtime.family,
            "strategy_event_id": (
                f"strategy:{runtime.label}:signal:{decision_id}"
                if decision_id is not None
                else f"strategy:{runtime.label}:signal:{update.ticker}:{event_kind}:{update.event_time.isoformat()}"
            ),
            "event_kind": event_kind,
            "event_time": _iso(update.event_time),
            "ticker": update.ticker,
            "side": update.side,
            "decision_id": decision_id,
            "approved": update.approved,
            "status": "approved" if update.approved else "blocked",
            "reason": update.block_reason,
            "reference_price_cents": update.reference_price_cents,
            "limit_price_cents": update.max_acceptable_entry_price_cents,
            "predicted_yes_probability": update.predicted_yes_probability,
            "predicted_no_probability": update.predicted_no_probability,
            "feature_basis_market_prob": update.feature_basis_market_prob,
            "raw_model_edge": update.raw_model_edge,
            "post_cost_edge": update.post_cost_edge,
            "yes_post_cost_edge": update.yes_post_cost_edge,
            "no_post_cost_edge": update.no_post_cost_edge,
            "tau_minutes": update.tau_minutes,
            "last_yes_price_cents": update.last_yes_price_cents,
            "yes_bid_cents": update.yes_bid_cents,
            "yes_ask_cents": update.yes_ask_cents,
            "buy_yes_price_cents": update.buy_yes_price_cents,
            "buy_no_price_cents": update.buy_no_price_cents,
            "quote_mid_prob": update.quote_mid_prob,
            "quote_spread_cents": update.quote_spread_cents,
            "quote_age_seconds": update.quote_age_seconds,
            "regime_label": update.regime_label,
            "bearish_vote_count": update.bearish_vote_count,
            "bullish_vote_count": update.bullish_vote_count,
            "regime_price_momentum_bearish": update.regime_price_momentum_bearish,
            "regime_signed_flow_bearish": update.regime_signed_flow_bearish,
            "regime_yes_share_bearish": update.regime_yes_share_bearish,
            "regime_price_momentum_bullish": update.regime_price_momentum_bullish,
            "regime_signed_flow_bullish": update.regime_signed_flow_bullish,
            "regime_yes_share_bullish": update.regime_yes_share_bullish,
            "tau_bucket": update.tau_bucket,
            "price_bucket": update.price_bucket,
            "probability_bucket": update.chosen_side_probability_bucket,
            "edge_bucket": update.chosen_side_edge_bucket,
            "bucket_policy_dimension": update.bucket_policy_dimension,
            "bucket_policy_bucket": update.bucket_policy_bucket,
            "bucket_policy_side": update.bucket_policy_side,
            "contracts": None if update.trade_intent is None else update.trade_intent.contracts,
            "estimated_entry_cost_dollars": (
                None if update.trade_intent is None else update.trade_intent.estimated_entry_cost_dollars
            ),
            "estimated_fees_dollars": (
                None if update.trade_intent is None else update.trade_intent.estimated_fees_dollars
            ),
            "estimated_cash_required_dollars": (
                None if update.trade_intent is None else update.trade_intent.estimated_cash_required_dollars
            ),
            "thesis_id": None if update.trade_intent is None else update.trade_intent.thesis_id,
            "tranche_index": None if update.trade_intent is None else update.trade_intent.tranche_index,
            "tranche_window": None if update.trade_intent is None else update.trade_intent.tranche_window,
            "tranche_reason": None if update.trade_intent is None else update.trade_intent.tranche_reason,
            "lifecycle_state": None if update.trade_intent is None else update.trade_intent.lifecycle_state,
            "total_thesis_budget_dollars": (
                None if update.trade_intent is None else update.trade_intent.total_thesis_budget_dollars
            ),
            "payout_if_yes_dollars": None if update.trade_intent is None else update.trade_intent.payout_if_yes_dollars,
            "payout_if_no_dollars": None if update.trade_intent is None else update.trade_intent.payout_if_no_dollars,
            "expected_value_dollars": None if update.trade_intent is None else update.trade_intent.expected_value_dollars,
            "worst_case_loss_dollars": (
                None if update.trade_intent is None else update.trade_intent.worst_case_loss_dollars
            ),
            "execution_mode": runtime.execution_engine.config.mode.value,
            "subaccount": runtime.execution_engine.config.subaccount,
        }
        await self._write_stage_row("strategy_events", update.event_time, row)

    async def _write_execution_event(
        self,
        runtime: KalshiLiveArchiveRuntime,
        update: KalshiExecutionUpdate,
    ) -> None:
        row = {
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "model_label": runtime.label,
            "model_family": runtime.family,
            "strategy_event_id": (
                f"strategy:{runtime.label}:execution:{update.decision_id}:{update.status}:{update.event_time.isoformat()}"
            ),
            "event_kind": f"execution_{update.status}",
            "event_time": _iso(update.event_time),
            "ticker": update.ticker,
            "side": update.side,
            "decision_id": update.decision_id,
            "client_order_id": update.client_order_id,
            "order_id": update.order_id,
            "approved": None,
            "status": update.status,
            "reason": update.message,
            "reference_price_cents": update.reference_price_cents,
            "limit_price_cents": update.limit_price_cents,
            "contracts": update.contracts,
            "filled_contracts": update.filled_contracts,
            "remaining_contracts": update.remaining_contracts,
            "fill_price_cents": update.fill_price_cents,
            "entry_cost_dollars": update.entry_cost_dollars,
            "fees_dollars": update.fees_dollars,
            "cash_required_dollars": update.cash_required_dollars,
            "available_cash_dollars": update.available_cash_dollars,
            "realized_pnl_dollars": update.realized_pnl_dollars,
            "cumulative_realized_pnl_dollars": update.cumulative_realized_pnl_dollars,
            "settlement_result": update.settlement_result,
            "thesis_id": update.thesis_id,
            "tranche_index": update.tranche_index,
            "tranche_window": update.tranche_window,
            "tranche_reason": update.tranche_reason,
            "lifecycle_state": update.lifecycle_state,
            "total_thesis_budget_dollars": update.total_thesis_budget_dollars,
            "payout_if_yes_dollars": update.payout_if_yes_dollars,
            "payout_if_no_dollars": update.payout_if_no_dollars,
            "expected_value_dollars": update.expected_value_dollars,
            "worst_case_loss_dollars": update.worst_case_loss_dollars,
            "execution_mode": update.mode.value,
            "subaccount": runtime.execution_engine.config.subaccount,
        }
        await self._write_stage_row("strategy_events", update.event_time, row)

    async def _write_layering_event(
        self,
        runtime: KalshiLiveArchiveRuntime,
        update: KalshiLayeringDecision,
    ) -> None:
        row = {
            "run_name": self.config.run_name,
            "environment": self.config.environment,
            "model_label": runtime.label,
            "model_family": runtime.family,
            "strategy_event_id": (
                f"strategy:{runtime.label}:layering:{update.ticker}:{update.action}:{update.event_time.isoformat()}"
            ),
            "event_kind": f"layering_{update.action}",
            "event_time": _iso(update.event_time),
            "ticker": update.ticker,
            "side": update.side,
            "decision_id": None,
            "approved": None,
            "status": update.status,
            "reason": update.message,
            "reference_price_cents": update.entry_price_cents,
            "limit_price_cents": update.entry_price_cents,
            "contracts": update.contracts,
            "filled_contracts": None,
            "remaining_contracts": None,
            "fill_price_cents": None,
            "entry_cost_dollars": None,
            "fees_dollars": None,
            "cash_required_dollars": None,
            "available_cash_dollars": None,
            "realized_pnl_dollars": None,
            "cumulative_realized_pnl_dollars": None,
            "settlement_result": None,
            "thesis_id": update.thesis_id,
            "tranche_index": update.tranche_index,
            "tranche_window": update.decision_window,
            "tranche_reason": update.action,
            "lifecycle_state": update.lifecycle_state,
            "total_thesis_budget_dollars": update.total_thesis_budget_dollars,
            "payout_if_yes_dollars": update.payout_if_yes_dollars,
            "payout_if_no_dollars": update.payout_if_no_dollars,
            "expected_value_dollars": update.expected_value_dollars,
            "worst_case_loss_dollars": update.worst_case_loss_dollars,
            "execution_mode": runtime.execution_engine.config.mode.value,
            "subaccount": runtime.execution_engine.config.subaccount,
        }
        await self._write_stage_row("strategy_events", update.event_time, row)

    async def _write_stage_row(self, layer: str, event_time: datetime, row: dict[str, Any]) -> None:
        await self._maybe_compact_rolled_hours(event_time)
        date_part, hour_part = _partition_parts(event_time)
        path = (
            self.config.archive_root
            / f"{layer}_staging"
            / f"environment={self.config.environment}"
            / f"date={date_part}"
            / f"hour={hour_part}"
            / f"part-{self._session_id}.jsonl.zst"
        )
        async with self._write_lock:
            _append_jsonl_zst(path, row)

    async def _maybe_compact_rolled_hours(self, event_time: datetime) -> None:
        hour_key = event_time.astimezone(UTC).strftime("%Y-%m-%dT%H")
        if self._last_compacted_hour_key is None:
            self._last_compacted_hour_key = hour_key
            return
        if hour_key == self._last_compacted_hour_key:
            return
        self._last_compacted_hour_key = hour_key
        try:
            await self.compact_all_staging(exclude_hour_key=hour_key)
        except Exception as exc:  # pragma: no cover - defensive live path
            print(f"[live-archive] hourly compaction failed: {exc!r}")

    async def compact_all_staging(self, *, exclude_hour_key: str | None = None) -> None:
        async with self._write_lock:
            for layer in ARCHIVE_STAGE_LAYERS:
                stage_root = self.config.archive_root / f"{layer}_staging"
                if not stage_root.exists():
                    continue
                for hour_dir in sorted(stage_root.rglob("hour=*")):
                    date_dir = hour_dir.parent
                    date_name = date_dir.name.split("=", 1)[-1]
                    hour_name = hour_dir.name.split("=", 1)[-1]
                    hour_key = f"{date_name}T{hour_name}"
                    if exclude_hour_key is not None and hour_key == exclude_hour_key:
                        continue
                    self._compact_partition_dir(layer, date_dir.parent.name, date_name, hour_name, hour_dir)

    def _compact_partition_dir(
        self,
        layer: str,
        environment_name: str,
        date_part: str,
        hour_part: str,
        hour_dir: Path,
    ) -> None:
        stage_files = sorted(hour_dir.glob("*.jsonl.zst"))
        if not stage_files:
            return
        rows: list[dict[str, Any]] = []
        for stage_file in stage_files:
            rows.extend(_read_jsonl_zst(stage_file))
        if not rows:
            for stage_file in stage_files:
                stage_file.unlink(missing_ok=True)
            return
        final_dir = (
            self.config.archive_root
            / layer
            / environment_name
            / f"date={date_part}"
            / f"hour={hour_part}"
        )
        final_dir.mkdir(parents=True, exist_ok=True)
        session_id = getattr(self, "_session_id", "repair")
        output_path = final_dir / f"part-{session_id}-{uuid.uuid4().hex[:8]}.parquet"
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        for stage_file in stage_files:
            stage_file.unlink(missing_ok=True)


async def repair_live_archive(archive_root: Path, *, run_name: str | None = None, environment: str = "production") -> None:
    del run_name
    config = KalshiLiveArchiveConfig(
        archive_root=archive_root,
        run_name="repair",
        environment=environment,
        compact_on_shutdown=True,
    )
    manager = KalshiLiveArchiveManager.__new__(KalshiLiveArchiveManager)
    manager.config = config
    manager._write_lock = asyncio.Lock()
    manager._last_compacted_hour_key = None
    await KalshiLiveArchiveManager.compact_all_staging(manager)
