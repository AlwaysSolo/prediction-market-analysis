from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.indexers.kalshi.models import parse_datetime
from src.live.kalshi.bucket_policy import (
    DEFAULT_BANNED_NO_PRICE_BUCKETS,
    DEFAULT_BANNED_YES_PRICE_BUCKETS,
    DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
    DEFAULT_BANNED_YES_TAU_BUCKETS,
    EDGE_BUCKETS,
    PRICE_BUCKETS,
    PROBABILITY_BUCKETS,
    TAU_BUCKETS,
    KalshiBucketPolicyEvaluation,
    KalshiChosenSideBuckets,
    bucket_policy_fields,
    build_chosen_side_buckets,
    evaluate_bucket_ban_policy,
    parse_bucket_csv,
)
from src.live.kalshi.collector import KalshiMarketDataCollector
from src.live.kalshi.config import KalshiEnvironment
from src.live.kalshi.regime import KalshiRegimeEvaluation, evaluate_kxbtc15m_regime_for_state
from src.live.kalshi.scorer import KalshiLightGBMScoreState, KalshiLightGBMScoreUpdate
from src.live.kalshi.signal_risk import (
    KalshiSignalRiskConfig,
    calculate_cost_metrics,
    find_max_acceptable_entry_price_cents,
)
from src.live.kalshi.types import KalshiTickerUpdate

SampleCallback = Callable[["KalshiResearchSampleUpdate"], Awaitable[None] | None]
SettlementCallback = Callable[["KalshiResearchSettlementUpdate"], Awaitable[None] | None]
SummaryCallback = Callable[["KalshiResearchSummaryUpdate"], Awaitable[None] | None]
RECOVERY_RECONCILE_INTERVAL_SECONDS = 15.0


def utc_now() -> datetime:
    return datetime.now(UTC)


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


def _env_var_names(environment: KalshiEnvironment, suffix: str) -> tuple[str, str]:
    env_prefix = "DEMO" if environment is KalshiEnvironment.DEMO else "PROD"
    return (f"KALSHI_{env_prefix}_RESEARCH_{suffix}", f"KALSHI_RESEARCH_{suffix}")


def _resolve_env_value(environment: KalshiEnvironment, suffix: str) -> str | None:
    for env_name in _env_var_names(environment, suffix):
        value = os.getenv(env_name)
        if value is not None and value != "":
            return value
    return None


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")

def _regime_fields(regime: KalshiRegimeEvaluation) -> dict[str, Any]:
    return {
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


@dataclass(frozen=True)
class KalshiResearchSamplerConfig:
    min_edge_cents: float = 2.0
    min_tau_minutes: float = 2.0
    max_tau_minutes: float = 14.0
    apply_regime_hard_gate: bool = False
    price_band_min_cents: int = 20
    price_band_max_cents: int = 80
    quote_max_age_seconds: float = 3.0
    contracts_per_sample: int = 1
    slippage_pct: float = 1.0
    enable_bucket_ban_policy: bool = True
    banned_yes_tau_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_YES_TAU_BUCKETS)
    banned_yes_price_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_YES_PRICE_BUCKETS)
    banned_yes_probability_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_YES_PROBABILITY_BUCKETS)
    banned_no_price_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_NO_PRICE_BUCKETS)

    def __post_init__(self) -> None:
        if self.min_edge_cents < 0:
            raise ValueError("min_edge_cents must be non-negative")
        if self.min_tau_minutes < 0:
            raise ValueError("min_tau_minutes must be non-negative")
        if self.max_tau_minutes < self.min_tau_minutes:
            raise ValueError("max_tau_minutes must be >= min_tau_minutes")
        if not (0 <= self.price_band_min_cents <= 99):
            raise ValueError("price_band_min_cents must be between 0 and 99")
        if not (1 <= self.price_band_max_cents <= 100):
            raise ValueError("price_band_max_cents must be between 1 and 100")
        if self.price_band_min_cents > self.price_band_max_cents:
            raise ValueError("price_band_min_cents must be <= price_band_max_cents")
        if self.quote_max_age_seconds <= 0:
            raise ValueError("quote_max_age_seconds must be positive")
        if self.contracts_per_sample <= 0:
            raise ValueError("contracts_per_sample must be positive")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")

    @property
    def min_edge(self) -> float:
        return self.min_edge_cents / 100.0

    @property
    def slippage(self) -> float:
        return self.slippage_pct / 100.0

    @classmethod
    def from_env(cls, environment: KalshiEnvironment) -> KalshiResearchSamplerConfig:
        return cls(
            min_edge_cents=float(_resolve_env_value(environment, "MIN_EDGE_CENTS") or 2.0),
            min_tau_minutes=float(_resolve_env_value(environment, "MIN_TAU_MINUTES") or 2.0),
            max_tau_minutes=float(_resolve_env_value(environment, "MAX_TAU_MINUTES") or 14.0),
            apply_regime_hard_gate=_parse_bool(_resolve_env_value(environment, "APPLY_REGIME_HARD_GATE") or "true"),
            price_band_min_cents=int(_resolve_env_value(environment, "PRICE_BAND_MIN_CENTS") or 20),
            price_band_max_cents=int(_resolve_env_value(environment, "PRICE_BAND_MAX_CENTS") or 80),
            quote_max_age_seconds=float(_resolve_env_value(environment, "QUOTE_MAX_AGE_SECONDS") or 3.0),
            contracts_per_sample=int(_resolve_env_value(environment, "CONTRACTS_PER_SAMPLE") or 1),
            slippage_pct=float(_resolve_env_value(environment, "SLIPPAGE_PCT") or 1.0),
            enable_bucket_ban_policy=_parse_bool(_resolve_env_value(environment, "ENABLE_BUCKET_BAN_POLICY") or "true"),
            banned_yes_tau_buckets=parse_bucket_csv(
                _resolve_env_value(environment, "BANNED_YES_TAU_BUCKETS"),
                default=DEFAULT_BANNED_YES_TAU_BUCKETS,
            ),
            banned_yes_price_buckets=parse_bucket_csv(
                _resolve_env_value(environment, "BANNED_YES_PRICE_BUCKETS"),
                default=DEFAULT_BANNED_YES_PRICE_BUCKETS,
            ),
            banned_yes_probability_buckets=parse_bucket_csv(
                _resolve_env_value(environment, "BANNED_YES_PROBABILITY_BUCKETS"),
                default=DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
            ),
            banned_no_price_buckets=parse_bucket_csv(
                _resolve_env_value(environment, "BANNED_NO_PRICE_BUCKETS"),
                default=DEFAULT_BANNED_NO_PRICE_BUCKETS,
            ),
        )

    def to_signal_risk_config(self) -> KalshiSignalRiskConfig:
        return KalshiSignalRiskConfig(
            edge_threshold_cents=self.min_edge_cents,
            min_tau_minutes=self.min_tau_minutes,
            max_tau_minutes=self.max_tau_minutes,
            apply_regime_hard_gate=self.apply_regime_hard_gate,
            price_band_min_cents=self.price_band_min_cents,
            price_band_max_cents=self.price_band_max_cents,
            quote_max_age_seconds=self.quote_max_age_seconds,
            contracts_per_order=self.contracts_per_sample,
            slippage_pct=self.slippage_pct,
            enable_bucket_ban_policy=self.enable_bucket_ban_policy,
            banned_yes_tau_buckets=self.banned_yes_tau_buckets,
            banned_yes_price_buckets=self.banned_yes_price_buckets,
            banned_yes_probability_buckets=self.banned_yes_probability_buckets,
            banned_no_price_buckets=self.banned_no_price_buckets,
        )


@dataclass(frozen=True)
class KalshiResearchSample:
    sample_id: str
    ticker: str
    side: str
    recorded_at: datetime
    contracts: int
    reference_price_cents: int
    max_acceptable_entry_price_cents: int | None
    predicted_yes_probability: float
    predicted_no_probability: float
    chosen_side_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    chosen_post_cost_edge: float
    last_yes_price_cents: int | None
    last_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
    no_bid_cents: int | None
    no_ask_cents: int | None
    buy_yes_price_cents: int | None
    buy_no_price_cents: int | None
    quote_mid_prob: float | None
    quote_spread_cents: int | None
    quote_age_seconds: float | None
    regime_label: str
    bearish_vote_count: int
    bullish_vote_count: int
    regime_price_momentum_bearish: bool
    regime_signed_flow_bearish: bool
    regime_yes_share_bearish: bool
    regime_price_momentum_bullish: bool
    regime_signed_flow_bullish: bool
    regime_yes_share_bullish: bool
    tau_minutes: float
    tau_bucket: str
    price_bucket: str
    chosen_side_probability_bucket: str
    chosen_side_edge_bucket: str
    estimated_entry_cost_dollars: float
    estimated_fees_dollars: float
    estimated_cash_required_dollars: float
    market_event_id: str = ""
    feature_row_id: str = ""
    raw_event_id: str | None = None
    received_at: datetime | None = None
    source: str = "snapshot"


@dataclass(frozen=True)
class KalshiResearchSampleUpdate:
    ticker: str
    event_time: datetime
    status: str
    reason: str | None
    side: str | None
    predicted_yes_probability: float
    predicted_no_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    chosen_post_cost_edge: float | None
    tau_minutes: float
    reference_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
    no_bid_cents: int | None
    no_ask_cents: int | None
    buy_yes_price_cents: int | None
    buy_no_price_cents: int | None
    quote_mid_prob: float | None
    quote_spread_cents: int | None
    quote_age_seconds: float | None
    regime_label: str
    bearish_vote_count: int
    bullish_vote_count: int
    regime_price_momentum_bearish: bool
    regime_signed_flow_bearish: bool
    regime_yes_share_bearish: bool
    regime_price_momentum_bullish: bool
    regime_signed_flow_bullish: bool
    regime_yes_share_bullish: bool
    tau_bucket: str | None
    price_bucket: str | None
    chosen_side_probability_bucket: str | None
    chosen_side_edge_bucket: str | None
    bucket_policy_dimension: str | None
    bucket_policy_bucket: str | None
    bucket_policy_side: str | None
    sample: KalshiResearchSample | None
    market_event_id: str = ""
    feature_row_id: str = ""
    raw_event_id: str | None = None
    received_at: datetime | None = None
    source: str = "snapshot"


@dataclass(frozen=True)
class KalshiResearchSettlementUpdate:
    sample_id: str
    ticker: str
    event_time: datetime
    side: str
    settlement_result: str
    is_win: bool
    realized_pnl_dollars: float
    cumulative_realized_pnl_dollars: float
    sample: KalshiResearchSample


@dataclass(frozen=True)
class KalshiResearchSummaryUpdate:
    event_time: datetime
    open_sample_count: int
    settled_sample_count: int
    win_count: int
    loss_count: int
    cumulative_realized_pnl_dollars: float


@dataclass(frozen=True)
class _ResearchSideEvaluation:
    side: str
    entry_price_cents: int
    max_acceptable_entry_price_cents: int | None
    post_cost_edge: float
    entry_cost_dollars: float
    fees_dollars: float
    cash_required_dollars: float


class KalshiResearchSampler:
    def __init__(
        self,
        scorer: Any,
        config: KalshiResearchSamplerConfig | None = None,
        *,
        log_dir: Path | None = None,
    ):
        self.scorer = scorer
        self.config = config or KalshiResearchSamplerConfig()
        collector_config = self.scorer.feature_engine.collector.config
        self._logger = JsonlEventLogger(
            log_dir or (collector_config.log_dir.parent / "research"),
            collector_config.environment.value,
        )
        self._score_queue: asyncio.Queue[KalshiLightGBMScoreUpdate] | None = None
        self._callbacks: list[SampleCallback] = []
        self._queues: list[asyncio.Queue[KalshiResearchSampleUpdate]] = []
        self._sample_queues: list[asyncio.Queue[KalshiResearchSample]] = []
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._seen_signatures: dict[str, set[tuple[str, str, str, str, str]]] = defaultdict(set)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._score_queue is None:
            self._score_queue = self.scorer.subscribe_queue()
        await self._logger.write(
            "research_started",
            {
                "min_edge_cents": self.config.min_edge_cents,
                "min_tau_minutes": self.config.min_tau_minutes,
                "max_tau_minutes": self.config.max_tau_minutes,
                "apply_regime_hard_gate": self.config.apply_regime_hard_gate,
                "price_band_min_cents": self.config.price_band_min_cents,
                "price_band_max_cents": self.config.price_band_max_cents,
                "quote_max_age_seconds": self.config.quote_max_age_seconds,
                "contracts_per_sample": self.config.contracts_per_sample,
                "slippage_pct": self.config.slippage_pct,
                "enable_bucket_ban_policy": self.config.enable_bucket_ban_policy,
                "banned_yes_tau_buckets": sorted(self.config.banned_yes_tau_buckets),
                "banned_yes_price_buckets": sorted(self.config.banned_yes_price_buckets),
                "banned_yes_probability_buckets": sorted(self.config.banned_yes_probability_buckets),
                "banned_no_price_buckets": sorted(self.config.banned_no_price_buckets),
            },
        )
        await self._bootstrap_from_scorer()
        self._task = asyncio.create_task(self._consume_loop(), name="kalshi-research-sampler")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        await self._logger.write("research_stopped", {})

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def subscribe(self, callback: SampleCallback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiResearchSampleUpdate]:
        queue: asyncio.Queue[KalshiResearchSampleUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def subscribe_sample_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiResearchSample]:
        queue: asyncio.Queue[KalshiResearchSample] = asyncio.Queue(maxsize=maxsize)
        self._sample_queues.append(queue)
        return queue

    async def _bootstrap_from_scorer(self) -> None:
        for score_state in self.scorer.snapshot_states().values():
            await self._handle_score_state(score_state)

    async def _consume_loop(self) -> None:
        if self._score_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                update = await self._score_queue.get()
                await self._handle_score_state(update)
        except asyncio.CancelledError:
            raise

    async def _handle_score_state(
        self,
        score_state: KalshiLightGBMScoreState | KalshiLightGBMScoreUpdate,
    ) -> None:
        regime = evaluate_kxbtc15m_regime_for_state(score_state)
        quote_reason = self._quote_rejection_reason(score_state)
        if quote_reason is not None:
            await self._publish_skipped(score_state, quote_reason, None, None, regime)
            return
        if not self._tau_allowed(score_state.tau_minutes):
            await self._publish_skipped(score_state, "outside_tau_window", None, None, regime)
            return

        yes_evaluation = None
        no_evaluation = None
        if score_state.buy_yes_price_cents is not None:
            yes_evaluation = self._evaluate_side_candidate(
                score_state,
                side="YES",
                entry_price_cents=score_state.buy_yes_price_cents,
            )
        if score_state.buy_no_price_cents is not None:
            no_evaluation = self._evaluate_side_candidate(
                score_state,
                side="NO",
                entry_price_cents=score_state.buy_no_price_cents,
            )

        chosen_evaluation = self._select_side_evaluation(
            yes_evaluation=yes_evaluation,
            no_evaluation=no_evaluation,
        )
        if chosen_evaluation is None:
            await self._publish_skipped(
                score_state,
                self._rejection_reason(score_state, yes_evaluation=yes_evaluation, no_evaluation=no_evaluation),
                yes_evaluation,
                no_evaluation,
                regime,
            )
            return

        predicted_no_probability = 1.0 - score_state.predicted_yes_probability
        chosen_probability = (
            score_state.predicted_yes_probability
            if chosen_evaluation.side == "YES"
            else predicted_no_probability
        )
        chosen_buckets = build_chosen_side_buckets(
            side=chosen_evaluation.side,
            tau_minutes=score_state.tau_minutes,
            entry_price_cents=chosen_evaluation.entry_price_cents,
            predicted_yes_probability=score_state.predicted_yes_probability,
            chosen_edge_cents=chosen_evaluation.post_cost_edge * 100.0,
        )
        signature = (
            chosen_evaluation.side,
            chosen_buckets.tau_bucket,
            chosen_buckets.price_bucket,
            chosen_buckets.chosen_side_probability_bucket,
            chosen_buckets.chosen_side_edge_bucket,
        )
        if signature in self._seen_signatures[score_state.ticker]:
            await self._publish_skipped(
                score_state,
                "duplicate_signature",
                yes_evaluation,
                no_evaluation,
                regime,
                chosen_buckets=chosen_buckets,
            )
            return
        bucket_policy = evaluate_bucket_ban_policy(
            enabled=self.config.enable_bucket_ban_policy,
            side=chosen_evaluation.side,
            buckets=chosen_buckets,
            banned_yes_tau_buckets=self.config.banned_yes_tau_buckets,
            banned_yes_price_buckets=self.config.banned_yes_price_buckets,
            banned_yes_probability_buckets=self.config.banned_yes_probability_buckets,
            banned_no_price_buckets=self.config.banned_no_price_buckets,
        )
        if bucket_policy.is_blocked:
            await self._publish_skipped(
                score_state,
                "blocked_by_bucket_policy",
                yes_evaluation,
                no_evaluation,
                regime,
                chosen_buckets=chosen_buckets,
                bucket_policy=bucket_policy,
            )
            return
        if self.config.apply_regime_hard_gate and chosen_evaluation.side == "YES" and regime.regime_label == "downtrend":
            await self._publish_skipped(
                score_state,
                "blocked_by_regime_downtrend",
                yes_evaluation,
                no_evaluation,
                regime,
                chosen_buckets=chosen_buckets,
            )
            return
        self._seen_signatures[score_state.ticker].add(signature)

        sample = KalshiResearchSample(
            sample_id=str(uuid.uuid4()),
            ticker=score_state.ticker,
            side=chosen_evaluation.side,
            recorded_at=score_state.event_time,
            contracts=self.config.contracts_per_sample,
            reference_price_cents=chosen_evaluation.entry_price_cents,
            max_acceptable_entry_price_cents=chosen_evaluation.max_acceptable_entry_price_cents,
            predicted_yes_probability=score_state.predicted_yes_probability,
            predicted_no_probability=predicted_no_probability,
            chosen_side_probability=chosen_probability,
            feature_basis_market_prob=score_state.market_prob,
            raw_model_edge=score_state.model_edge,
            yes_post_cost_edge=None if yes_evaluation is None else yes_evaluation.post_cost_edge,
            no_post_cost_edge=None if no_evaluation is None else no_evaluation.post_cost_edge,
            chosen_post_cost_edge=chosen_evaluation.post_cost_edge,
            last_yes_price_cents=score_state.last_yes_price_cents,
            last_price_cents=score_state.last_price_cents,
            yes_bid_cents=score_state.yes_bid_cents,
            yes_ask_cents=score_state.yes_ask_cents,
            no_bid_cents=score_state.no_bid_cents,
            no_ask_cents=score_state.no_ask_cents,
            buy_yes_price_cents=score_state.buy_yes_price_cents,
            buy_no_price_cents=score_state.buy_no_price_cents,
            quote_mid_prob=score_state.quote_mid_prob,
            quote_spread_cents=score_state.quote_spread_cents,
            quote_age_seconds=score_state.quote_age_seconds,
            regime_label=regime.regime_label,
            bearish_vote_count=regime.bearish_vote_count,
            bullish_vote_count=regime.bullish_vote_count,
            regime_price_momentum_bearish=regime.price_momentum_bearish,
            regime_signed_flow_bearish=regime.signed_flow_bearish,
            regime_yes_share_bearish=regime.yes_share_bearish,
            regime_price_momentum_bullish=regime.price_momentum_bullish,
            regime_signed_flow_bullish=regime.signed_flow_bullish,
            regime_yes_share_bullish=regime.yes_share_bullish,
            tau_minutes=score_state.tau_minutes,
            tau_bucket=chosen_buckets.tau_bucket,
            price_bucket=chosen_buckets.price_bucket,
            chosen_side_probability_bucket=chosen_buckets.chosen_side_probability_bucket,
            chosen_side_edge_bucket=chosen_buckets.chosen_side_edge_bucket,
            estimated_entry_cost_dollars=chosen_evaluation.entry_cost_dollars,
            estimated_fees_dollars=chosen_evaluation.fees_dollars,
            estimated_cash_required_dollars=chosen_evaluation.cash_required_dollars,
            market_event_id=score_state.event_id,
            feature_row_id=f"feature:{score_state.event_id}" if score_state.event_id else "",
            raw_event_id=score_state.raw_event_id,
            received_at=score_state.received_at,
            source=score_state.source,
        )
        update = KalshiResearchSampleUpdate(
            ticker=score_state.ticker,
            event_time=score_state.event_time,
            status="recorded",
            reason=None,
            side=sample.side,
            predicted_yes_probability=sample.predicted_yes_probability,
            predicted_no_probability=sample.predicted_no_probability,
            feature_basis_market_prob=sample.feature_basis_market_prob,
            raw_model_edge=sample.raw_model_edge,
            yes_post_cost_edge=sample.yes_post_cost_edge,
            no_post_cost_edge=sample.no_post_cost_edge,
            chosen_post_cost_edge=sample.chosen_post_cost_edge,
            tau_minutes=sample.tau_minutes,
            reference_price_cents=sample.reference_price_cents,
            yes_bid_cents=sample.yes_bid_cents,
            yes_ask_cents=sample.yes_ask_cents,
            no_bid_cents=sample.no_bid_cents,
            no_ask_cents=sample.no_ask_cents,
            buy_yes_price_cents=sample.buy_yes_price_cents,
            buy_no_price_cents=sample.buy_no_price_cents,
            quote_mid_prob=sample.quote_mid_prob,
            quote_spread_cents=sample.quote_spread_cents,
            quote_age_seconds=sample.quote_age_seconds,
            regime_label=sample.regime_label,
            bearish_vote_count=sample.bearish_vote_count,
            bullish_vote_count=sample.bullish_vote_count,
            regime_price_momentum_bearish=sample.regime_price_momentum_bearish,
            regime_signed_flow_bearish=sample.regime_signed_flow_bearish,
            regime_yes_share_bearish=sample.regime_yes_share_bearish,
            regime_price_momentum_bullish=sample.regime_price_momentum_bullish,
            regime_signed_flow_bullish=sample.regime_signed_flow_bullish,
            regime_yes_share_bullish=sample.regime_yes_share_bullish,
            tau_bucket=sample.tau_bucket,
            price_bucket=sample.price_bucket,
            chosen_side_probability_bucket=sample.chosen_side_probability_bucket,
            chosen_side_edge_bucket=sample.chosen_side_edge_bucket,
            bucket_policy_dimension=None,
            bucket_policy_bucket=None,
            bucket_policy_side=None,
            sample=sample,
            market_event_id=sample.market_event_id,
            feature_row_id=sample.feature_row_id,
            raw_event_id=sample.raw_event_id,
            received_at=sample.received_at,
            source=sample.source,
        )
        await self._logger.write("research_sample_recorded", _sample_payload(sample), score_state.event_time)
        await self._publish_update(update)
        for queue in self._sample_queues:
            await queue.put(sample)

    async def _publish_skipped(
        self,
        score_state: KalshiLightGBMScoreState | KalshiLightGBMScoreUpdate,
        reason: str,
        yes_evaluation: _ResearchSideEvaluation | None,
        no_evaluation: _ResearchSideEvaluation | None,
        regime: KalshiRegimeEvaluation,
        *,
        chosen_buckets: KalshiChosenSideBuckets | None = None,
        bucket_policy: KalshiBucketPolicyEvaluation | None = None,
    ) -> None:
        predicted_no_probability = 1.0 - score_state.predicted_yes_probability
        chosen_edge = None
        side = None
        chosen = self._select_side_evaluation(yes_evaluation=yes_evaluation, no_evaluation=no_evaluation)
        if chosen is not None:
            chosen_edge = chosen.post_cost_edge
            side = chosen.side
        reference_price_cents = None if chosen is None else chosen.entry_price_cents
        if chosen_buckets is None:
            chosen_buckets = build_chosen_side_buckets(
                side=side,
                tau_minutes=score_state.tau_minutes,
                entry_price_cents=reference_price_cents,
                predicted_yes_probability=score_state.predicted_yes_probability,
                chosen_edge_cents=None if chosen_edge is None else chosen_edge * 100.0,
            )
        update = KalshiResearchSampleUpdate(
            ticker=score_state.ticker,
            event_time=score_state.event_time,
            status="skipped",
            reason=reason,
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            predicted_no_probability=predicted_no_probability,
            feature_basis_market_prob=score_state.market_prob,
            raw_model_edge=score_state.model_edge,
            yes_post_cost_edge=None if yes_evaluation is None else yes_evaluation.post_cost_edge,
            no_post_cost_edge=None if no_evaluation is None else no_evaluation.post_cost_edge,
            chosen_post_cost_edge=chosen_edge,
            tau_minutes=score_state.tau_minutes,
            reference_price_cents=reference_price_cents,
            yes_bid_cents=score_state.yes_bid_cents,
            yes_ask_cents=score_state.yes_ask_cents,
            no_bid_cents=score_state.no_bid_cents,
            no_ask_cents=score_state.no_ask_cents,
            buy_yes_price_cents=score_state.buy_yes_price_cents,
            buy_no_price_cents=score_state.buy_no_price_cents,
            quote_mid_prob=score_state.quote_mid_prob,
            quote_spread_cents=score_state.quote_spread_cents,
            quote_age_seconds=score_state.quote_age_seconds,
            regime_label=regime.regime_label,
            bearish_vote_count=regime.bearish_vote_count,
            bullish_vote_count=regime.bullish_vote_count,
            regime_price_momentum_bearish=regime.price_momentum_bearish,
            regime_signed_flow_bearish=regime.signed_flow_bearish,
            regime_yes_share_bearish=regime.yes_share_bearish,
            regime_price_momentum_bullish=regime.price_momentum_bullish,
            regime_signed_flow_bullish=regime.signed_flow_bullish,
            regime_yes_share_bullish=regime.yes_share_bullish,
            tau_bucket=chosen_buckets.tau_bucket,
            price_bucket=chosen_buckets.price_bucket,
            chosen_side_probability_bucket=chosen_buckets.chosen_side_probability_bucket,
            chosen_side_edge_bucket=chosen_buckets.chosen_side_edge_bucket,
            bucket_policy_dimension=None if bucket_policy is None else bucket_policy.blocked_dimension,
            bucket_policy_bucket=None if bucket_policy is None else bucket_policy.blocked_bucket,
            bucket_policy_side=None if bucket_policy is None else bucket_policy.blocked_side,
            sample=None,
            market_event_id=score_state.event_id,
            feature_row_id=f"feature:{score_state.event_id}" if score_state.event_id else "",
            raw_event_id=score_state.raw_event_id,
            received_at=score_state.received_at,
            source=score_state.source,
        )
        await self._logger.write(
            "research_sample_skipped",
            {
                "ticker": score_state.ticker,
                "reason": reason,
                "side": side,
                "predicted_yes_probability": score_state.predicted_yes_probability,
                "predicted_no_probability": predicted_no_probability,
                "feature_basis_market_prob": score_state.market_prob,
                "raw_model_edge": score_state.model_edge,
                "yes_post_cost_edge": None if yes_evaluation is None else yes_evaluation.post_cost_edge,
                "no_post_cost_edge": None if no_evaluation is None else no_evaluation.post_cost_edge,
                "chosen_post_cost_edge": chosen_edge,
                "tau_minutes": score_state.tau_minutes,
                "reference_price_cents": reference_price_cents,
                "yes_bid_cents": score_state.yes_bid_cents,
                "yes_ask_cents": score_state.yes_ask_cents,
                "no_bid_cents": score_state.no_bid_cents,
                "no_ask_cents": score_state.no_ask_cents,
                "buy_yes_price_cents": score_state.buy_yes_price_cents,
                "buy_no_price_cents": score_state.buy_no_price_cents,
                "quote_mid_prob": score_state.quote_mid_prob,
                "quote_spread_cents": score_state.quote_spread_cents,
                "quote_age_seconds": score_state.quote_age_seconds,
                "tau_bucket": chosen_buckets.tau_bucket,
                "price_bucket": chosen_buckets.price_bucket,
                "chosen_side_probability_bucket": chosen_buckets.chosen_side_probability_bucket,
                "chosen_side_edge_bucket": chosen_buckets.chosen_side_edge_bucket,
                **_regime_fields(regime),
                **bucket_policy_fields(bucket_policy),
                "market_event_id": score_state.event_id,
                "feature_row_id": f"feature:{score_state.event_id}" if score_state.event_id else "",
                "raw_event_id": score_state.raw_event_id,
                "received_at": score_state.received_at.isoformat() if score_state.received_at else None,
                "source": score_state.source,
            },
            score_state.event_time,
        )
        await self._publish_update(update)

    async def _publish_update(self, update: KalshiResearchSampleUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)

    def _quote_rejection_reason(
        self,
        score_state: KalshiLightGBMScoreState | KalshiLightGBMScoreUpdate,
    ) -> str | None:
        if (
            score_state.yes_bid_cents is None
            or score_state.yes_ask_cents is None
            or score_state.buy_yes_price_cents is None
            or score_state.buy_no_price_cents is None
        ):
            return "missing_quote"
        if score_state.yes_bid_cents >= score_state.yes_ask_cents:
            return "crossed_quote"
        if score_state.quote_age_seconds is None:
            return "missing_quote"
        if score_state.quote_age_seconds > self.config.quote_max_age_seconds:
            return "stale_quote"
        return None

    def _tau_allowed(self, tau_minutes: float) -> bool:
        return self.config.min_tau_minutes <= tau_minutes <= self.config.max_tau_minutes

    def _evaluate_side_candidate(
        self,
        score_state: KalshiLightGBMScoreState | KalshiLightGBMScoreUpdate,
        *,
        side: str,
        entry_price_cents: int,
    ) -> _ResearchSideEvaluation | None:
        if entry_price_cents < self.config.price_band_min_cents or entry_price_cents > self.config.price_band_max_cents:
            return None
        max_acceptable_entry_price_cents = find_max_acceptable_entry_price_cents(
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            config=self.config.to_signal_risk_config(),
            contracts=self.config.contracts_per_sample,
        )
        post_cost_edge, entry_cost_dollars, fees_dollars, cash_required_dollars = calculate_cost_metrics(
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            displayed_entry_price_cents=entry_price_cents,
            contracts=self.config.contracts_per_sample,
            slippage=self.config.slippage,
        )
        if (
            max_acceptable_entry_price_cents is None
            or entry_price_cents > max_acceptable_entry_price_cents
            or post_cost_edge + 1e-12 < self.config.min_edge
        ):
            return None
        return _ResearchSideEvaluation(
            side=side,
            entry_price_cents=entry_price_cents,
            max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
            post_cost_edge=post_cost_edge,
            entry_cost_dollars=entry_cost_dollars,
            fees_dollars=fees_dollars,
            cash_required_dollars=cash_required_dollars,
        )

    def _select_side_evaluation(
        self,
        *,
        yes_evaluation: _ResearchSideEvaluation | None,
        no_evaluation: _ResearchSideEvaluation | None,
    ) -> _ResearchSideEvaluation | None:
        evaluations = [evaluation for evaluation in (yes_evaluation, no_evaluation) if evaluation is not None]
        if not evaluations:
            return None
        evaluations.sort(key=lambda item: (item.post_cost_edge, -item.entry_price_cents), reverse=True)
        return evaluations[0]

    def _rejection_reason(
        self,
        score_state: KalshiLightGBMScoreState | KalshiLightGBMScoreUpdate,
        *,
        yes_evaluation: _ResearchSideEvaluation | None,
        no_evaluation: _ResearchSideEvaluation | None,
    ) -> str:
        if (
            score_state.buy_yes_price_cents is not None
            and score_state.buy_no_price_cents is not None
            and (
                (
                    score_state.buy_yes_price_cents < self.config.price_band_min_cents
                    or score_state.buy_yes_price_cents > self.config.price_band_max_cents
                )
                and (
                    score_state.buy_no_price_cents < self.config.price_band_min_cents
                    or score_state.buy_no_price_cents > self.config.price_band_max_cents
                )
            )
        ):
            return "outside_price_band"
        if yes_evaluation is None and no_evaluation is None:
            return "insufficient_post_cost_edge"
        return "unknown"


class KalshiResearchLedger:
    def __init__(
        self,
        collector: KalshiMarketDataCollector,
        sampler: KalshiResearchSampler,
        *,
        log_dir: Path | None = None,
    ):
        self.collector = collector
        self.sampler = sampler
        self._logger = JsonlEventLogger(
            log_dir or (self.collector.config.log_dir.parent / "research"),
            self.collector.config.environment.value,
        )
        self._sample_queue = self.sampler.subscribe_sample_queue()
        self._collector_queue = self.collector.subscribe_queue()
        self._callbacks: list[SettlementCallback] = []
        self._queues: list[asyncio.Queue[KalshiResearchSettlementUpdate]] = []
        self._summary_callbacks: list[SummaryCallback] = []
        self._summary_queues: list[asyncio.Queue[KalshiResearchSummaryUpdate]] = []
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._open_samples: dict[str, KalshiResearchSample] = {}
        self._sample_ids_by_ticker: dict[str, set[str]] = defaultdict(set)
        self._settled_sample_count = 0
        self._win_count = 0
        self._loss_count = 0
        self._cumulative_realized_pnl_dollars = 0.0

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        await self._recover_logged_state()
        self._tasks = [
            asyncio.create_task(self._consume_sample_loop(), name="kalshi-research-ledger-samples"),
            asyncio.create_task(self._consume_collector_loop(), name="kalshi-research-ledger-collector"),
            asyncio.create_task(self._reconcile_open_samples_loop(), name="kalshi-research-ledger-reconcile"),
        ]
        await self._reconcile_open_samples()
        self._ready_event.set()

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

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def subscribe(self, callback: SettlementCallback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiResearchSettlementUpdate]:
        queue: asyncio.Queue[KalshiResearchSettlementUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def subscribe_summary(self, callback: SummaryCallback) -> None:
        self._summary_callbacks.append(callback)

    def subscribe_summary_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiResearchSummaryUpdate]:
        queue: asyncio.Queue[KalshiResearchSummaryUpdate] = asyncio.Queue(maxsize=maxsize)
        self._summary_queues.append(queue)
        return queue

    async def _consume_sample_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                sample = await self._sample_queue.get()
                await self._handle_recorded_sample(sample)
        except asyncio.CancelledError:
            raise

    async def _consume_collector_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                update = await self._collector_queue.get()
                await self._handle_collector_update(update)
        except asyncio.CancelledError:
            raise

    async def _reconcile_open_samples_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(RECOVERY_RECONCILE_INTERVAL_SECONDS)
                if self._stop_event.is_set():
                    break
                await self._reconcile_open_samples()
        except asyncio.CancelledError:
            raise

    async def _handle_recorded_sample(self, sample: KalshiResearchSample) -> None:
        self._open_samples[sample.sample_id] = sample
        self._sample_ids_by_ticker[sample.ticker].add(sample.sample_id)
        await self._write_summary_snapshot(sample.recorded_at)
        await self._maybe_settle_ticker(sample.ticker, sample.recorded_at)

    async def _handle_collector_update(self, update: KalshiTickerUpdate) -> None:
        await self._maybe_settle_ticker(update.ticker, update.event_time)

    async def _recover_logged_state(self) -> None:
        environment_dir = self._logger.base_dir / self._logger.environment
        recovered_open_samples: dict[str, KalshiResearchSample] = {}
        recovered_by_ticker: dict[str, set[str]] = defaultdict(set)
        settled_sample_count = 0
        win_count = 0
        loss_count = 0
        cumulative_realized_pnl_dollars = 0.0

        if not environment_dir.exists():
            self._open_samples = {}
            self._sample_ids_by_ticker = defaultdict(set)
            self._settled_sample_count = 0
            self._win_count = 0
            self._loss_count = 0
            self._cumulative_realized_pnl_dollars = 0.0
            return

        for path in sorted(environment_dir.rglob("events.jsonl")):
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = event.get("payload", {})
                event_type = event.get("event_type")
                if event_type == "research_sample_recorded":
                    sample = _sample_from_event_payload(payload, logged_at=event.get("logged_at"))
                    recovered_open_samples[sample.sample_id] = sample
                    recovered_by_ticker[sample.ticker].add(sample.sample_id)
                    continue
                if event_type != "research_sample_settled":
                    continue

                sample_id = str(payload.get("sample_id") or "")
                ticker = str(payload.get("ticker") or "")
                sample = recovered_open_samples.pop(sample_id, None)
                if sample is not None:
                    recovered_by_ticker[sample.ticker].discard(sample_id)
                    if not recovered_by_ticker[sample.ticker]:
                        recovered_by_ticker.pop(sample.ticker, None)
                elif ticker:
                    recovered_by_ticker[ticker].discard(sample_id)
                    if not recovered_by_ticker[ticker]:
                        recovered_by_ticker.pop(ticker, None)

                settled_sample_count += 1
                is_win = bool(payload.get("is_win"))
                if is_win:
                    win_count += 1
                else:
                    loss_count += 1
                cumulative_realized_pnl_dollars += float(payload.get("realized_pnl_dollars", 0.0) or 0.0)

        self._open_samples = recovered_open_samples
        self._sample_ids_by_ticker = defaultdict(set, recovered_by_ticker)
        self._settled_sample_count = settled_sample_count
        self._win_count = win_count
        self._loss_count = loss_count
        self._cumulative_realized_pnl_dollars = cumulative_realized_pnl_dollars

    async def _reconcile_open_samples(self) -> None:
        if not self._sample_ids_by_ticker:
            return
        for ticker in tuple(self._sample_ids_by_ticker):
            market = self.collector.get_market(ticker)
            if market is None or market.result.strip().upper() not in {"YES", "NO"}:
                await self.collector.refresh_market_snapshot(ticker)
            await self._maybe_settle_ticker(ticker, utc_now())

    async def _maybe_settle_ticker(self, ticker: str, event_time: datetime) -> None:
        market = self.collector.get_market(ticker)
        if market is None:
            return
        settlement_result = market.result.strip().upper()
        if settlement_result not in {"YES", "NO"}:
            return
        sample_ids = tuple(self._sample_ids_by_ticker.get(ticker, ()))
        for sample_id in sample_ids:
            sample = self._open_samples.pop(sample_id, None)
            if sample is None:
                continue
            self._sample_ids_by_ticker[ticker].discard(sample_id)
            await self._settle_sample(sample, settlement_result=settlement_result, event_time=event_time)
        if not self._sample_ids_by_ticker[ticker]:
            self._sample_ids_by_ticker.pop(ticker, None)

    async def _settle_sample(
        self,
        sample: KalshiResearchSample,
        *,
        settlement_result: str,
        event_time: datetime,
    ) -> None:
        payout_dollars = float(sample.contracts) if settlement_result == sample.side.upper() else 0.0
        realized_pnl_dollars = payout_dollars - sample.estimated_cash_required_dollars
        self._cumulative_realized_pnl_dollars += realized_pnl_dollars
        self._settled_sample_count += 1
        is_win = realized_pnl_dollars > 0.0
        if is_win:
            self._win_count += 1
        else:
            self._loss_count += 1
        update = KalshiResearchSettlementUpdate(
            sample_id=sample.sample_id,
            ticker=sample.ticker,
            event_time=event_time,
            side=sample.side,
            settlement_result=settlement_result,
            is_win=is_win,
            realized_pnl_dollars=realized_pnl_dollars,
            cumulative_realized_pnl_dollars=self._cumulative_realized_pnl_dollars,
            sample=sample,
        )
        await self._logger.write(
            "research_sample_settled",
            {
                "sample_id": sample.sample_id,
                "ticker": sample.ticker,
                "side": sample.side,
                "settlement_result": settlement_result,
                "is_win": is_win,
                "contracts": sample.contracts,
                "cash_required_dollars": sample.estimated_cash_required_dollars,
                "realized_pnl_dollars": realized_pnl_dollars,
                "cumulative_realized_pnl_dollars": self._cumulative_realized_pnl_dollars,
            },
            event_time,
        )
        await self._write_summary_snapshot(event_time)
        await self._publish_update(update)

    async def _write_summary_snapshot(self, event_time: datetime) -> None:
        update = KalshiResearchSummaryUpdate(
            event_time=event_time,
            open_sample_count=len(self._open_samples),
            settled_sample_count=self._settled_sample_count,
            win_count=self._win_count,
            loss_count=self._loss_count,
            cumulative_realized_pnl_dollars=self._cumulative_realized_pnl_dollars,
        )
        await self._logger.write(
            "research_summary_snapshot",
            {
                "open_sample_count": update.open_sample_count,
                "settled_sample_count": update.settled_sample_count,
                "win_count": update.win_count,
                "loss_count": update.loss_count,
                "cumulative_realized_pnl_dollars": update.cumulative_realized_pnl_dollars,
            },
            event_time,
        )
        for callback in self._summary_callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._summary_queues:
            await queue.put(update)

    async def _publish_update(self, update: KalshiResearchSettlementUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)


def _sample_payload(sample: KalshiResearchSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "ticker": sample.ticker,
        "side": sample.side,
        "contracts": sample.contracts,
        "reference_price_cents": sample.reference_price_cents,
        "max_acceptable_entry_price_cents": sample.max_acceptable_entry_price_cents,
        "predicted_yes_probability": sample.predicted_yes_probability,
        "predicted_no_probability": sample.predicted_no_probability,
        "chosen_side_probability": sample.chosen_side_probability,
        "feature_basis_market_prob": sample.feature_basis_market_prob,
        "raw_model_edge": sample.raw_model_edge,
        "yes_post_cost_edge": sample.yes_post_cost_edge,
        "no_post_cost_edge": sample.no_post_cost_edge,
        "chosen_post_cost_edge": sample.chosen_post_cost_edge,
        "last_yes_price_cents": sample.last_yes_price_cents,
        "last_price_cents": sample.last_price_cents,
        "yes_bid_cents": sample.yes_bid_cents,
        "yes_ask_cents": sample.yes_ask_cents,
        "no_bid_cents": sample.no_bid_cents,
        "no_ask_cents": sample.no_ask_cents,
        "buy_yes_price_cents": sample.buy_yes_price_cents,
        "buy_no_price_cents": sample.buy_no_price_cents,
        "quote_mid_prob": sample.quote_mid_prob,
        "quote_spread_cents": sample.quote_spread_cents,
        "quote_age_seconds": sample.quote_age_seconds,
        "regime_label": sample.regime_label,
        "bearish_vote_count": sample.bearish_vote_count,
        "bullish_vote_count": sample.bullish_vote_count,
        "regime_price_momentum_bearish": sample.regime_price_momentum_bearish,
        "regime_signed_flow_bearish": sample.regime_signed_flow_bearish,
        "regime_yes_share_bearish": sample.regime_yes_share_bearish,
        "regime_price_momentum_bullish": sample.regime_price_momentum_bullish,
        "regime_signed_flow_bullish": sample.regime_signed_flow_bullish,
        "regime_yes_share_bullish": sample.regime_yes_share_bullish,
        "tau_minutes": sample.tau_minutes,
        "tau_bucket": sample.tau_bucket,
        "price_bucket": sample.price_bucket,
        "chosen_side_probability_bucket": sample.chosen_side_probability_bucket,
        "chosen_side_edge_bucket": sample.chosen_side_edge_bucket,
        "estimated_entry_cost_dollars": sample.estimated_entry_cost_dollars,
        "estimated_fees_dollars": sample.estimated_fees_dollars,
        "estimated_cash_required_dollars": sample.estimated_cash_required_dollars,
        "market_event_id": sample.market_event_id,
        "feature_row_id": sample.feature_row_id,
        "raw_event_id": sample.raw_event_id,
        "received_at": sample.received_at.isoformat() if sample.received_at else None,
        "source": sample.source,
    }


def _sample_from_event_payload(payload: dict[str, Any], *, logged_at: str | None) -> KalshiResearchSample:
    recorded_at = parse_datetime(logged_at) if logged_at else utc_now()
    return KalshiResearchSample(
        sample_id=str(payload["sample_id"]),
        ticker=str(payload["ticker"]),
        side=str(payload["side"]),
        recorded_at=recorded_at,
        contracts=int(payload["contracts"]),
        reference_price_cents=int(payload["reference_price_cents"]),
        max_acceptable_entry_price_cents=(
            int(payload["max_acceptable_entry_price_cents"])
            if payload.get("max_acceptable_entry_price_cents") is not None
            else None
        ),
        predicted_yes_probability=float(payload["predicted_yes_probability"]),
        predicted_no_probability=float(payload["predicted_no_probability"]),
        chosen_side_probability=float(payload["chosen_side_probability"]),
        feature_basis_market_prob=float(payload["feature_basis_market_prob"]),
        raw_model_edge=float(payload["raw_model_edge"]),
        yes_post_cost_edge=(
            float(payload["yes_post_cost_edge"]) if payload.get("yes_post_cost_edge") is not None else None
        ),
        no_post_cost_edge=(
            float(payload["no_post_cost_edge"]) if payload.get("no_post_cost_edge") is not None else None
        ),
        chosen_post_cost_edge=float(payload["chosen_post_cost_edge"]),
        last_yes_price_cents=(
            int(payload["last_yes_price_cents"]) if payload.get("last_yes_price_cents") is not None else None
        ),
        last_price_cents=int(payload["last_price_cents"]) if payload.get("last_price_cents") is not None else None,
        yes_bid_cents=int(payload["yes_bid_cents"]) if payload.get("yes_bid_cents") is not None else None,
        yes_ask_cents=int(payload["yes_ask_cents"]) if payload.get("yes_ask_cents") is not None else None,
        no_bid_cents=int(payload["no_bid_cents"]) if payload.get("no_bid_cents") is not None else None,
        no_ask_cents=int(payload["no_ask_cents"]) if payload.get("no_ask_cents") is not None else None,
        buy_yes_price_cents=(
            int(payload["buy_yes_price_cents"]) if payload.get("buy_yes_price_cents") is not None else None
        ),
        buy_no_price_cents=(
            int(payload["buy_no_price_cents"]) if payload.get("buy_no_price_cents") is not None else None
        ),
        quote_mid_prob=float(payload["quote_mid_prob"]) if payload.get("quote_mid_prob") is not None else None,
        quote_spread_cents=(
            int(payload["quote_spread_cents"]) if payload.get("quote_spread_cents") is not None else None
        ),
        quote_age_seconds=(
            float(payload["quote_age_seconds"]) if payload.get("quote_age_seconds") is not None else None
        ),
        regime_label=str(payload.get("regime_label") or "neutral"),
        bearish_vote_count=int(payload.get("bearish_vote_count") or 0),
        bullish_vote_count=int(payload.get("bullish_vote_count") or 0),
        regime_price_momentum_bearish=bool(payload.get("regime_price_momentum_bearish")),
        regime_signed_flow_bearish=bool(payload.get("regime_signed_flow_bearish")),
        regime_yes_share_bearish=bool(payload.get("regime_yes_share_bearish")),
        regime_price_momentum_bullish=bool(payload.get("regime_price_momentum_bullish")),
        regime_signed_flow_bullish=bool(payload.get("regime_signed_flow_bullish")),
        regime_yes_share_bullish=bool(payload.get("regime_yes_share_bullish")),
        tau_minutes=float(payload["tau_minutes"]),
        tau_bucket=str(payload["tau_bucket"]),
        price_bucket=str(payload["price_bucket"]),
        chosen_side_probability_bucket=str(payload["chosen_side_probability_bucket"]),
        chosen_side_edge_bucket=str(payload["chosen_side_edge_bucket"]),
        estimated_entry_cost_dollars=float(payload["estimated_entry_cost_dollars"]),
        estimated_fees_dollars=float(payload["estimated_fees_dollars"]),
        estimated_cash_required_dollars=float(payload["estimated_cash_required_dollars"]),
        market_event_id=str(payload.get("market_event_id") or ""),
        feature_row_id=str(payload.get("feature_row_id") or ""),
        raw_event_id=str(payload.get("raw_event_id")) if payload.get("raw_event_id") is not None else None,
        received_at=parse_datetime(str(payload["received_at"])) if payload.get("received_at") else None,
        source=str(payload.get("source") or "snapshot"),
    )
