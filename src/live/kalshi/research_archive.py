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

from src.live.kalshi.bucket_policy import (
    label_edge_bucket,
    label_price_bucket,
    label_probability_bucket,
    label_tau_bucket,
)
from src.live.kalshi.collector import KalshiMarketDataCollector
from src.live.kalshi.feature_engine import KalshiFeatureStateEngine
from src.live.kalshi.features import KalshiFeatureUpdate
from src.live.kalshi.regime import evaluate_kxbtc15m_regime_for_state
from src.live.kalshi.research.engine import KalshiResearchSampleUpdate, KalshiResearchSettlementUpdate, KalshiResearchSummaryUpdate
from src.live.kalshi.scorer import KalshiLightGBMScoreUpdate
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


def _sample_strategy_event_id(model_label: str, update: KalshiResearchSampleUpdate) -> str:
    if update.sample is not None:
        return f"strategy:{model_label}:{update.status}:{update.sample.sample_id}"
    if update.feature_row_id:
        return f"strategy:{model_label}:{update.status}:{update.feature_row_id}"
    return f"strategy:{model_label}:{update.status}:{update.ticker}:{update.event_time.isoformat()}"


def _settlement_strategy_event_id(model_label: str, update: KalshiResearchSettlementUpdate) -> str:
    return f"strategy:{model_label}:settled:{update.sample_id}"


def _summary_strategy_event_id(model_label: str, update: KalshiResearchSummaryUpdate) -> str:
    return f"strategy:{model_label}:summary:{update.event_time.isoformat()}"


def _safe_model_output_id(model_label: str, feature_row_id: str, event_time: datetime, ticker: str) -> str:
    if feature_row_id:
        return f"model_output:{model_label}:{feature_row_id}"
    return f"model_output:{model_label}:{ticker}:{event_time.isoformat()}"


def _build_stage_strategy_row(
    *,
    run_name: str,
    environment: str,
    model_label: str,
    model_family: str,
    event_kind: str,
    strategy_event_id: str,
    event_time: datetime,
    ticker: str | None,
    side: str | None,
    sample_id: str | None,
    feature_row_id: str,
    model_output_id: str,
    reason: str | None,
    settlement_result: str | None,
    is_win: bool | None,
    realized_pnl_dollars: float | None,
    cumulative_realized_pnl_dollars: float | None,
    reference_price_cents: int | None,
    chosen_side_probability: float | None,
    chosen_post_cost_edge: float | None,
    tau_bucket: str | None,
    price_bucket: str | None,
    probability_bucket: str | None,
    edge_bucket: str | None,
    regime_label: str | None = None,
    bearish_vote_count: int | None = None,
    bullish_vote_count: int | None = None,
    regime_price_momentum_bearish: bool | None = None,
    regime_signed_flow_bearish: bool | None = None,
    regime_yes_share_bearish: bool | None = None,
    regime_price_momentum_bullish: bool | None = None,
    regime_signed_flow_bullish: bool | None = None,
    regime_yes_share_bullish: bool | None = None,
    bucket_policy_dimension: str | None = None,
    bucket_policy_bucket: str | None = None,
    bucket_policy_side: str | None = None,
    open_sample_count: int | None = None,
    settled_sample_count: int | None = None,
    win_count: int | None = None,
    loss_count: int | None = None,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "environment": environment,
        "model_label": model_label,
        "model_family": model_family,
        "strategy_event_id": strategy_event_id,
        "event_kind": event_kind,
        "event_time": _iso(event_time),
        "ticker": ticker,
        "side": side,
        "sample_id": sample_id,
        "feature_row_id": feature_row_id,
        "model_output_id": model_output_id,
        "reason": reason,
        "settlement_result": settlement_result,
        "is_win": is_win,
        "realized_pnl_dollars": realized_pnl_dollars,
        "cumulative_realized_pnl_dollars": cumulative_realized_pnl_dollars,
        "reference_price_cents": reference_price_cents,
        "chosen_side_probability": chosen_side_probability,
        "chosen_post_cost_edge": chosen_post_cost_edge,
        "tau_bucket": tau_bucket,
        "price_bucket": price_bucket,
        "probability_bucket": probability_bucket,
        "edge_bucket": edge_bucket,
        "regime_label": regime_label,
        "bearish_vote_count": bearish_vote_count,
        "bullish_vote_count": bullish_vote_count,
        "regime_price_momentum_bearish": regime_price_momentum_bearish,
        "regime_signed_flow_bearish": regime_signed_flow_bearish,
        "regime_yes_share_bearish": regime_yes_share_bearish,
        "regime_price_momentum_bullish": regime_price_momentum_bullish,
        "regime_signed_flow_bullish": regime_signed_flow_bullish,
        "regime_yes_share_bullish": regime_yes_share_bullish,
        "bucket_policy_dimension": bucket_policy_dimension,
        "bucket_policy_bucket": bucket_policy_bucket,
        "bucket_policy_side": bucket_policy_side,
        "open_sample_count": open_sample_count,
        "settled_sample_count": settled_sample_count,
        "win_count": win_count,
        "loss_count": loss_count,
    }


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
class KalshiResearchArchiveConfig:
    archive_root: Path
    run_name: str
    environment: str
    compact_on_shutdown: bool = True


@dataclass(frozen=True)
class KalshiResearchArchiveRuntime:
    label: str
    family: str
    model_file: Path
    scorer: Any
    sampler: Any
    ledger: Any
    calibration_enabled: bool = False


class KalshiResearchArchiveManager:
    def __init__(
        self,
        collector: KalshiMarketDataCollector,
        feature_engine: KalshiFeatureStateEngine,
        runtimes: list[KalshiResearchArchiveRuntime],
        config: KalshiResearchArchiveConfig,
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
        self._runtime_queues: list[tuple[KalshiResearchArchiveRuntime, asyncio.Queue[Any], str]] = []
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
                    (runtime, runtime.sampler.subscribe_queue(), "strategy_sample"),
                    (runtime, runtime.ledger.subscribe_queue(), "strategy_settlement"),
                    (runtime, runtime.ledger.subscribe_summary_queue(), "strategy_summary"),
                ]
            )
        await self.compact_all_staging()
        self._tasks = [
            asyncio.create_task(self._consume_raw_loop(), name="kalshi-research-archive-raw"),
            asyncio.create_task(self._consume_market_loop(), name="kalshi-research-archive-market"),
            asyncio.create_task(self._consume_feature_loop(), name="kalshi-research-archive-feature"),
        ]
        for runtime, queue, queue_kind in self._runtime_queues:
            self._tasks.append(
                asyncio.create_task(
                    self._consume_runtime_loop(runtime, queue, queue_kind),
                    name=f"kalshi-research-archive-{runtime.label}-{queue_kind}",
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
                print(f"[archive] compaction on shutdown failed: {exc!r}")

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
        runtime: KalshiResearchArchiveRuntime,
        queue: asyncio.Queue[Any],
        queue_kind: str,
    ) -> None:
        try:
            while not self._stop_event.is_set():
                update = await queue.get()
                if queue_kind == "model_outputs":
                    await self._write_model_output(runtime, update)
                elif queue_kind == "strategy_sample":
                    await self._write_strategy_sample_event(runtime, update)
                elif queue_kind == "strategy_settlement":
                    await self._write_strategy_settlement_event(runtime, update)
                elif queue_kind == "strategy_summary":
                    await self._write_strategy_summary_event(runtime, update)
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
            "yes_bid_size": update.yes_bid_size,
            "yes_ask_size": update.yes_ask_size,
            "no_bid_size": update.no_bid_size,
            "no_ask_size": update.no_ask_size,
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
            "buy_yes_size": (
                update.yes_ask_size
                if update.yes_ask_size is not None
                else update.no_bid_size
            ),
            "buy_no_price_cents": (
                update.no_ask_cents
                if update.no_ask_cents is not None
                else (100 - update.yes_bid_cents if update.yes_bid_cents is not None else None)
            ),
            "buy_no_size": (
                update.no_ask_size
                if update.no_ask_size is not None
                else update.yes_bid_size
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
            "schema_version": "kalshi_feature_row_v2",
            "in_training_window": update.in_training_window,
            "is_scoreable": update.is_scoreable,
            "btc_spot_source": update.btc_spot_source,
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
        runtime: KalshiResearchArchiveRuntime,
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

    async def _write_strategy_sample_event(
        self,
        runtime: KalshiResearchArchiveRuntime,
        update: KalshiResearchSampleUpdate,
    ) -> None:
        sample = update.sample
        chosen_probability = sample.chosen_side_probability if sample is not None else None
        chosen_edge_cents = None if update.chosen_post_cost_edge is None else update.chosen_post_cost_edge * 100.0
        reference_price_cents = update.reference_price_cents
        price_bucket = sample.price_bucket if sample is not None else label_price_bucket(reference_price_cents)
        probability_bucket = (
            sample.chosen_side_probability_bucket
            if sample is not None
            else label_probability_bucket(chosen_probability)
        )
        edge_bucket = sample.chosen_side_edge_bucket if sample is not None else label_edge_bucket(chosen_edge_cents)
        row = _build_stage_strategy_row(
            run_name=self.config.run_name,
            environment=self.config.environment,
            model_label=runtime.label,
            model_family=runtime.family,
            event_kind=update.status,
            strategy_event_id=_sample_strategy_event_id(runtime.label, update),
            event_time=update.event_time,
            ticker=update.ticker,
            side=update.side,
            sample_id=sample.sample_id if sample is not None else None,
            feature_row_id=sample.feature_row_id if sample is not None else update.feature_row_id,
            model_output_id=_safe_model_output_id(
                runtime.label,
                sample.feature_row_id if sample is not None else update.feature_row_id,
                update.event_time,
                update.ticker,
            ),
            reason=update.reason,
            settlement_result=None,
            is_win=None,
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=None,
            reference_price_cents=reference_price_cents,
            chosen_side_probability=chosen_probability,
            chosen_post_cost_edge=chosen_edge_cents,
            tau_bucket=sample.tau_bucket if sample is not None else label_tau_bucket(update.tau_minutes),
            price_bucket=price_bucket,
            probability_bucket=probability_bucket,
            edge_bucket=edge_bucket,
            regime_label=update.regime_label,
            bearish_vote_count=update.bearish_vote_count,
            bullish_vote_count=update.bullish_vote_count,
            regime_price_momentum_bearish=update.regime_price_momentum_bearish,
            regime_signed_flow_bearish=update.regime_signed_flow_bearish,
            regime_yes_share_bearish=update.regime_yes_share_bearish,
            regime_price_momentum_bullish=update.regime_price_momentum_bullish,
            regime_signed_flow_bullish=update.regime_signed_flow_bullish,
            regime_yes_share_bullish=update.regime_yes_share_bullish,
            bucket_policy_dimension=update.bucket_policy_dimension,
            bucket_policy_bucket=update.bucket_policy_bucket,
            bucket_policy_side=update.bucket_policy_side,
        )
        row.update(
            {
                "predicted_yes_probability": update.predicted_yes_probability,
                "predicted_no_probability": update.predicted_no_probability,
                "feature_basis_market_prob": update.feature_basis_market_prob,
                "raw_model_edge": update.raw_model_edge,
                "yes_post_cost_edge": (
                    update.yes_post_cost_edge * 100.0 if update.yes_post_cost_edge is not None else None
                ),
                "no_post_cost_edge": (
                    update.no_post_cost_edge * 100.0 if update.no_post_cost_edge is not None else None
                ),
                "quote_mid_prob": update.quote_mid_prob,
                "quote_spread_cents": update.quote_spread_cents,
                "quote_age_seconds": update.quote_age_seconds,
                "yes_bid_cents": update.yes_bid_cents,
                "yes_ask_cents": update.yes_ask_cents,
                "no_bid_cents": update.no_bid_cents,
                "no_ask_cents": update.no_ask_cents,
                "buy_yes_price_cents": update.buy_yes_price_cents,
                "buy_no_price_cents": update.buy_no_price_cents,
                "market_event_id": update.market_event_id,
                "raw_event_id": update.raw_event_id,
                "received_at": _iso(update.received_at),
                "source": update.source,
            }
        )
        await self._write_stage_row("strategy_events", update.received_at or update.event_time, row)

    async def _write_strategy_settlement_event(
        self,
        runtime: KalshiResearchArchiveRuntime,
        update: KalshiResearchSettlementUpdate,
    ) -> None:
        sample = update.sample
        row = _build_stage_strategy_row(
            run_name=self.config.run_name,
            environment=self.config.environment,
            model_label=runtime.label,
            model_family=runtime.family,
            event_kind="settled",
            strategy_event_id=_settlement_strategy_event_id(runtime.label, update),
            event_time=update.event_time,
            ticker=update.ticker,
            side=update.side,
            sample_id=update.sample_id,
            feature_row_id=sample.feature_row_id,
            model_output_id=_safe_model_output_id(runtime.label, sample.feature_row_id, update.event_time, update.ticker),
            reason=None,
            settlement_result=update.settlement_result,
            is_win=update.is_win,
            realized_pnl_dollars=update.realized_pnl_dollars,
            cumulative_realized_pnl_dollars=update.cumulative_realized_pnl_dollars,
            reference_price_cents=sample.reference_price_cents,
            chosen_side_probability=sample.chosen_side_probability,
            chosen_post_cost_edge=sample.chosen_post_cost_edge * 100.0,
            tau_bucket=sample.tau_bucket,
            price_bucket=sample.price_bucket,
            probability_bucket=sample.chosen_side_probability_bucket,
            edge_bucket=sample.chosen_side_edge_bucket,
            regime_label=sample.regime_label,
            bearish_vote_count=sample.bearish_vote_count,
            bullish_vote_count=sample.bullish_vote_count,
            regime_price_momentum_bearish=sample.regime_price_momentum_bearish,
            regime_signed_flow_bearish=sample.regime_signed_flow_bearish,
            regime_yes_share_bearish=sample.regime_yes_share_bearish,
            regime_price_momentum_bullish=sample.regime_price_momentum_bullish,
            regime_signed_flow_bullish=sample.regime_signed_flow_bullish,
            regime_yes_share_bullish=sample.regime_yes_share_bullish,
        )
        row.update(
            {
                "contracts": sample.contracts,
                "cash_required_dollars": sample.estimated_cash_required_dollars,
                "market_event_id": sample.market_event_id,
                "raw_event_id": sample.raw_event_id,
                "received_at": _iso(sample.received_at),
                "source": sample.source,
            }
        )
        await self._write_stage_row("strategy_events", update.event_time, row)

    async def _write_strategy_summary_event(
        self,
        runtime: KalshiResearchArchiveRuntime,
        update: KalshiResearchSummaryUpdate,
    ) -> None:
        row = _build_stage_strategy_row(
            run_name=self.config.run_name,
            environment=self.config.environment,
            model_label=runtime.label,
            model_family=runtime.family,
            event_kind="summary_snapshot",
            strategy_event_id=_summary_strategy_event_id(runtime.label, update),
            event_time=update.event_time,
            ticker=None,
            side=None,
            sample_id=None,
            feature_row_id="",
            model_output_id="",
            reason=None,
            settlement_result=None,
            is_win=None,
            realized_pnl_dollars=None,
            cumulative_realized_pnl_dollars=update.cumulative_realized_pnl_dollars,
            reference_price_cents=None,
            chosen_side_probability=None,
            chosen_post_cost_edge=None,
            tau_bucket=None,
            price_bucket=None,
            probability_bucket=None,
            edge_bucket=None,
            open_sample_count=update.open_sample_count,
            settled_sample_count=update.settled_sample_count,
            win_count=update.win_count,
            loss_count=update.loss_count,
        )
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
            print(f"[archive] hourly compaction failed: {exc!r}")

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
        output_path = final_dir / f"part-{self._session_id}-{uuid.uuid4().hex[:8]}.parquet"
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        for stage_file in stage_files:
            stage_file.unlink(missing_ok=True)


async def repair_research_archive(archive_root: Path, *, run_name: str | None = None, environment: str = "demo") -> None:
    manager = KalshiResearchArchiveManager.__new__(KalshiResearchArchiveManager)
    manager.config = KalshiResearchArchiveConfig(
        archive_root=archive_root,
        run_name=run_name or archive_root.parent.name,
        environment=environment,
        compact_on_shutdown=True,
    )
    manager._session_id = uuid.uuid4().hex[:12]
    manager._write_lock = asyncio.Lock()
    await KalshiResearchArchiveManager.compact_all_staging(manager)
