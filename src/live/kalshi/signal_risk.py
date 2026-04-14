from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.live.kalshi.bucket_policy import (
    DEFAULT_BANNED_COMBO_BUCKETS,
    DEFAULT_BANNED_NO_PRICE_BUCKETS,
    DEFAULT_BANNED_YES_PRICE_BUCKETS,
    DEFAULT_BANNED_YES_PROBABILITY_BUCKETS,
    DEFAULT_BANNED_YES_TAU_BUCKETS,
    KalshiBucketPolicyEvaluation,
    bucket_policy_fields,
    build_chosen_side_buckets,
    evaluate_bucket_ban_policy,
    evaluate_combo_ban_policy,
    parse_bucket_csv,
)
from src.live.kalshi.config import KalshiEnvironment
from src.live.kalshi.jsonl_logger import JsonlEventLogger
from src.live.kalshi.regime import (
    KalshiRegimeEvaluation,
    evaluate_kxbtc15m_regime_for_state,
    evaluate_kxbtc15m_structural_regime_for_state,
)
from src.live.kalshi.scorer import (
    KalshiLightGBMScorer,
    KalshiLightGBMScoreState,
    KalshiLightGBMScoreUpdate,
)

Callback = Callable[["KalshiSignalDecisionUpdate"], Awaitable[None] | None]
TradeIntentCallback = Callable[["KalshiTradeIntent"], Awaitable[None] | None]
StackingSignature = tuple[str | None, str | None, str | None, str | None, str | None]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _env_var_names(environment: KalshiEnvironment, suffix: str) -> tuple[str, str]:
    env_prefix = "DEMO" if environment is KalshiEnvironment.DEMO else "PROD"
    return (f"KALSHI_{env_prefix}_SIGNAL_{suffix}", f"KALSHI_SIGNAL_{suffix}")


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class KalshiSignalRiskConfig:
    edge_threshold_cents: float = 4.0
    maintain_edge_cents: float = 1.0
    min_tau_minutes: float = 2.0
    max_tau_minutes: float = 14.0
    apply_regime_hard_gate: bool = False
    invert_model_signal: bool = False
    allow_stacking: bool = False
    starting_cash_dollars: float = 10.0
    contracts_per_order: int = 1
    capital_pct_per_order: float | None = None
    kelly_fraction_multiplier: float | None = None
    kelly_fraction_cap_pct: float | None = None
    reserve_cash_pct: float = 30.0
    slippage_pct: float = 1.0
    price_band_min_cents: int = 20
    price_band_max_cents: int = 80
    quote_max_age_seconds: float = 3.0
    quote_consistency_tolerance_cents: int = 1
    enable_low_liquidity_chop_gate: bool = False
    low_liquidity_min_trade_count_300s: float = 3.0
    low_liquidity_min_contracts_sum_300s: float = 10.0
    low_liquidity_min_abs_signed_contracts_sum_300s: float = 3.0
    low_liquidity_price_volatility_300s_threshold: float = 0.01
    structural_regime_refresh_seconds: float = 30.0
    structural_regime_flip_confirmations: int = 2
    reservation_ttl_seconds: float = 5.0
    trade_cooldown_seconds: float = 0.0
    enable_bucket_ban_policy: bool = True
    banned_yes_tau_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_YES_TAU_BUCKETS)
    banned_yes_price_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_YES_PRICE_BUCKETS)
    banned_yes_probability_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_YES_PROBABILITY_BUCKETS)
    banned_no_price_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_NO_PRICE_BUCKETS)
    enable_combo_ban_policy: bool = False
    banned_combo_buckets: frozenset[str] = field(default_factory=lambda: DEFAULT_BANNED_COMBO_BUCKETS)
    blocked_regime_labels: frozenset[str] = field(default_factory=frozenset)
    auto_reserve_trade_intents: bool = True

    def __post_init__(self) -> None:
        if self.edge_threshold_cents < 0:
            raise ValueError("edge_threshold_cents must be non-negative")
        if self.maintain_edge_cents < 0:
            raise ValueError("maintain_edge_cents must be non-negative")
        if self.maintain_edge_cents > self.edge_threshold_cents:
            raise ValueError("maintain_edge_cents must be <= edge_threshold_cents")
        if self.min_tau_minutes < 0:
            raise ValueError("min_tau_minutes must be non-negative")
        if self.max_tau_minutes < self.min_tau_minutes:
            raise ValueError("max_tau_minutes must be >= min_tau_minutes")
        if self.starting_cash_dollars <= 0:
            raise ValueError("starting_cash_dollars must be positive")
        if self.contracts_per_order <= 0:
            raise ValueError("contracts_per_order must be positive")
        if self.capital_pct_per_order is not None and self.capital_pct_per_order <= 0:
            raise ValueError("capital_pct_per_order must be positive when provided")
        if self.kelly_fraction_multiplier is not None and self.kelly_fraction_multiplier < 0:
            raise ValueError("kelly_fraction_multiplier must be non-negative when provided")
        if self.kelly_fraction_cap_pct is not None and self.kelly_fraction_cap_pct <= 0:
            raise ValueError("kelly_fraction_cap_pct must be positive when provided")
        if not (0 <= self.reserve_cash_pct < 100):
            raise ValueError("reserve_cash_pct must be between 0 and 100")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        if not (0 <= self.price_band_min_cents <= 99):
            raise ValueError("price_band_min_cents must be between 0 and 99")
        if not (1 <= self.price_band_max_cents <= 100):
            raise ValueError("price_band_max_cents must be between 1 and 100")
        if self.price_band_min_cents > self.price_band_max_cents:
            raise ValueError("price_band_min_cents must be <= price_band_max_cents")
        if self.quote_max_age_seconds <= 0:
            raise ValueError("quote_max_age_seconds must be positive")
        if self.quote_consistency_tolerance_cents < 0:
            raise ValueError("quote_consistency_tolerance_cents must be non-negative")
        if self.low_liquidity_min_trade_count_300s < 0:
            raise ValueError("low_liquidity_min_trade_count_300s must be non-negative")
        if self.low_liquidity_min_contracts_sum_300s < 0:
            raise ValueError("low_liquidity_min_contracts_sum_300s must be non-negative")
        if self.low_liquidity_min_abs_signed_contracts_sum_300s < 0:
            raise ValueError("low_liquidity_min_abs_signed_contracts_sum_300s must be non-negative")
        if self.low_liquidity_price_volatility_300s_threshold < 0:
            raise ValueError("low_liquidity_price_volatility_300s_threshold must be non-negative")
        if self.structural_regime_refresh_seconds <= 0:
            raise ValueError("structural_regime_refresh_seconds must be positive")
        if self.structural_regime_flip_confirmations <= 0:
            raise ValueError("structural_regime_flip_confirmations must be positive")
        if self.reservation_ttl_seconds <= 0:
            raise ValueError("reservation_ttl_seconds must be positive")
        if self.trade_cooldown_seconds < 0:
            raise ValueError("trade_cooldown_seconds must be non-negative")

    @property
    def edge_threshold(self) -> float:
        return self.edge_threshold_cents / 100.0

    @property
    def maintain_edge_threshold(self) -> float:
        return self.maintain_edge_cents / 100.0

    @property
    def slippage(self) -> float:
        return self.slippage_pct / 100.0

    @classmethod
    def from_env(cls, environment: KalshiEnvironment) -> KalshiSignalRiskConfig:
        edge_threshold_cents = float(_resolve_env_value(environment, "EDGE_THRESHOLD_CENTS") or 4.0)
        maintain_edge_value = _resolve_env_value(environment, "MAINTAIN_EDGE_CENTS")
        return cls(
            edge_threshold_cents=edge_threshold_cents,
            maintain_edge_cents=(
                float(maintain_edge_value)
                if maintain_edge_value is not None
                else min(1.0, edge_threshold_cents)
            ),
            min_tau_minutes=float(_resolve_env_value(environment, "MIN_TAU_MINUTES") or 2.0),
            max_tau_minutes=float(_resolve_env_value(environment, "MAX_TAU_MINUTES") or 14.0),
            apply_regime_hard_gate=_parse_bool(_resolve_env_value(environment, "APPLY_REGIME_HARD_GATE") or "true"),
            invert_model_signal=_parse_bool(_resolve_env_value(environment, "INVERT_MODEL_SIGNAL") or "false"),
            allow_stacking=_parse_bool(_resolve_env_value(environment, "ALLOW_STACKING") or "false"),
            starting_cash_dollars=float(_resolve_env_value(environment, "STARTING_CASH_DOLLARS") or 10.0),
            contracts_per_order=int(_resolve_env_value(environment, "CONTRACTS_PER_ORDER") or 1),
            capital_pct_per_order=(
                float(_resolve_env_value(environment, "CAPITAL_PCT_PER_ORDER"))
                if _resolve_env_value(environment, "CAPITAL_PCT_PER_ORDER") is not None
                else None
            ),
            kelly_fraction_multiplier=(
                float(_resolve_env_value(environment, "KELLY_FRACTION_MULTIPLIER"))
                if _resolve_env_value(environment, "KELLY_FRACTION_MULTIPLIER") is not None
                else None
            ),
            kelly_fraction_cap_pct=(
                float(_resolve_env_value(environment, "KELLY_FRACTION_CAP_PCT"))
                if _resolve_env_value(environment, "KELLY_FRACTION_CAP_PCT") is not None
                else None
            ),
            reserve_cash_pct=float(_resolve_env_value(environment, "RESERVE_CASH_PCT") or 30.0),
            slippage_pct=float(_resolve_env_value(environment, "SLIPPAGE_PCT") or 1.0),
            price_band_min_cents=int(_resolve_env_value(environment, "PRICE_BAND_MIN_CENTS") or 20),
            price_band_max_cents=int(_resolve_env_value(environment, "PRICE_BAND_MAX_CENTS") or 80),
            quote_max_age_seconds=float(_resolve_env_value(environment, "QUOTE_MAX_AGE_SECONDS") or 3.0),
            quote_consistency_tolerance_cents=int(
                _resolve_env_value(environment, "QUOTE_CONSISTENCY_TOLERANCE_CENTS") or 1
            ),
            enable_low_liquidity_chop_gate=_parse_bool(
                _resolve_env_value(environment, "ENABLE_LOW_LIQUIDITY_CHOP_GATE") or "false"
            ),
            low_liquidity_min_trade_count_300s=float(
                _resolve_env_value(environment, "LOW_LIQUIDITY_MIN_TRADE_COUNT_300S") or 3.0
            ),
            low_liquidity_min_contracts_sum_300s=float(
                _resolve_env_value(environment, "LOW_LIQUIDITY_MIN_CONTRACTS_SUM_300S") or 10.0
            ),
            low_liquidity_min_abs_signed_contracts_sum_300s=float(
                _resolve_env_value(environment, "LOW_LIQUIDITY_MIN_ABS_SIGNED_CONTRACTS_SUM_300S") or 3.0
            ),
            low_liquidity_price_volatility_300s_threshold=float(
                _resolve_env_value(environment, "LOW_LIQUIDITY_PRICE_VOLATILITY_300S_THRESHOLD") or 0.01
            ),
            structural_regime_refresh_seconds=float(
                _resolve_env_value(environment, "STRUCTURAL_REGIME_REFRESH_SECONDS") or 30.0
            ),
            structural_regime_flip_confirmations=int(
                _resolve_env_value(environment, "STRUCTURAL_REGIME_FLIP_CONFIRMATIONS") or 2
            ),
            reservation_ttl_seconds=float(_resolve_env_value(environment, "RESERVATION_TTL_SECONDS") or 5.0),
            trade_cooldown_seconds=float(_resolve_env_value(environment, "TRADE_COOLDOWN_SECONDS") or 0.0),
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
            enable_combo_ban_policy=_parse_bool(_resolve_env_value(environment, "ENABLE_COMBO_BAN_POLICY") or "false"),
            banned_combo_buckets=parse_bucket_csv(
                _resolve_env_value(environment, "BANNED_COMBO_BUCKETS"),
                default=DEFAULT_BANNED_COMBO_BUCKETS,
            ),
            blocked_regime_labels=parse_bucket_csv(
                _resolve_env_value(environment, "BLOCKED_REGIME_LABELS"),
                default=frozenset(),
            ),
            auto_reserve_trade_intents=_parse_bool(
                _resolve_env_value(environment, "AUTO_RESERVE_TRADE_INTENTS") or "true"
            ),
        )


@dataclass(frozen=True)
class KalshiPortfolioPosition:
    ticker: str
    side: str
    contracts: int
    entry_cost_dollars: float = 0.0
    fees_dollars: float = 0.0
    cash_required_dollars: float = 0.0
    decision_id: str | None = None


@dataclass(frozen=True)
class KellySizingMetrics:
    side_probability: float
    per_contract_cash_required_dollars: float
    per_contract_win_profit_dollars: float
    payoff_ratio: float
    raw_fraction_of_equity: float
    scaled_fraction_of_equity: float
    capped_fraction_of_equity: float


@dataclass(frozen=True)
class KalshiPortfolioSnapshot:
    event_time: datetime
    available_cash_dollars: float
    deployed_capital_dollars: float = 0.0
    open_positions: tuple[KalshiPortfolioPosition, ...] = ()


@dataclass(frozen=True)
class KalshiExecutionFeedback:
    decision_id: str
    status: str
    event_time: datetime
    filled_contracts: int | None = None
    filled_price_cents: int | None = None
    cash_delta_dollars: float | None = None


@dataclass(frozen=True)
class KalshiTradeIntent:
    decision_id: str
    ticker: str
    side: str
    contracts: int
    reference_price_cents: int
    max_acceptable_entry_price_cents: int
    predicted_yes_probability: float
    predicted_no_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float
    post_cost_edge: float
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    last_yes_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
    buy_yes_price_cents: int | None
    buy_no_price_cents: int | None
    quote_mid_prob: float | None
    quote_spread_cents: int | None
    quote_age_seconds: float | None
    estimated_entry_cost_dollars: float
    estimated_fees_dollars: float
    estimated_cash_required_dollars: float
    generated_at: datetime
    stacking_signature: StackingSignature | None = None
    thesis_id: str | None = None
    tranche_index: int | None = None
    tranche_window: str | None = None
    tranche_reason: str | None = None
    lifecycle_state: str | None = None
    total_thesis_budget_dollars: float | None = None
    payout_if_yes_dollars: float | None = None
    payout_if_no_dollars: float | None = None
    expected_value_dollars: float | None = None
    worst_case_loss_dollars: float | None = None
    target_id: str | None = None
    attempt_index: int | None = None
    desired_contracts: int | None = None
    remaining_contracts_before_submit: int | None = None
    hard_max_price_cents: int | None = None
    retry_reason: str | None = None
    was_first_attempt: bool | None = None


@dataclass(frozen=True)
class KalshiSignalDecisionState:
    ticker: str
    event_time: datetime
    approved: bool
    side: str | None
    predicted_yes_probability: float
    predicted_no_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float | None
    post_cost_edge: float | None
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    tau_minutes: float
    reference_price_cents: int | None
    max_acceptable_entry_price_cents: int | None
    last_yes_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
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
    block_reason: str | None
    trade_intent: KalshiTradeIntent | None


@dataclass(frozen=True)
class KalshiSignalDecisionUpdate:
    ticker: str
    event_time: datetime
    approved: bool
    side: str | None
    predicted_yes_probability: float
    predicted_no_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float | None
    post_cost_edge: float | None
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    tau_minutes: float
    reference_price_cents: int | None
    max_acceptable_entry_price_cents: int | None
    last_yes_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
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
    block_reason: str | None
    trade_intent: KalshiTradeIntent | None


@dataclass(frozen=True)
class KalshiSignalPortfolioState:
    event_time: datetime
    baseline_available_cash_dollars: float
    baseline_deployed_capital_dollars: float
    available_cash_dollars: float
    deployed_capital_dollars: float
    equity_dollars: float
    reserved_cash_dollars: float
    open_positions: tuple[KalshiPortfolioPosition, ...]
    pending_reservations: tuple[KalshiPortfolioPosition, ...]


@dataclass
class _PendingReservation:
    decision_id: str
    ticker: str
    side: str
    contracts: int
    entry_cost_dollars: float
    fees_dollars: float
    cash_required_dollars: float
    created_at: datetime
    expires_at: datetime | None
    is_acknowledged: bool = False
    stacking_signature: StackingSignature | None = None

    def as_position(self) -> KalshiPortfolioPosition:
        return KalshiPortfolioPosition(
            ticker=self.ticker,
            side=self.side,
            contracts=self.contracts,
            entry_cost_dollars=self.entry_cost_dollars,
            fees_dollars=self.fees_dollars,
            cash_required_dollars=self.cash_required_dollars,
            decision_id=self.decision_id,
        )


@dataclass(frozen=True)
class _ExpiredReservation:
    reservation: _PendingReservation
    expired_at: datetime


@dataclass
class _StructuralRegimeMemory:
    current_label: str | None = None
    pending_label: str | None = None
    pending_count: int = 0
    last_refreshed_at: datetime | None = None
    last_evaluation: KalshiRegimeEvaluation | None = None


@dataclass(frozen=True)
class _DecisionCandidate:
    ticker: str
    event_time: datetime
    approved: bool
    side: str | None
    predicted_yes_probability: float
    predicted_no_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float | None
    post_cost_edge: float | None
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    tau_minutes: float
    reference_price_cents: int | None
    max_acceptable_entry_price_cents: int | None
    last_yes_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
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
    block_reason: str | None
    contracts: int
    entry_cost_dollars: float
    fees_dollars: float
    cash_required_dollars: float


def decision_update_from_state(state: KalshiSignalDecisionState) -> KalshiSignalDecisionUpdate:
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


def apply_adverse_slippage(entry_price: float, slippage: float) -> float:
    if slippage <= 0:
        return entry_price
    return min(0.999999, entry_price * (1.0 + slippage))


def calculate_kalshi_fee_dollars(entry_price: float, contracts: int) -> float:
    raw_fee = 0.07 * contracts * entry_price * (1.0 - entry_price)
    return float(np.ceil(raw_fee * 100.0) / 100.0)


def calculate_realized_cash_metrics(*, entry_price_cents: int, contracts: int) -> tuple[float, float, float]:
    entry_price = entry_price_cents / 100.0
    entry_cost = entry_price * contracts
    fees = calculate_kalshi_fee_dollars(entry_price, contracts)
    return entry_cost, fees, entry_cost + fees


def calculate_cost_metrics(
    *,
    side: str,
    predicted_yes_probability: float,
    displayed_entry_price_cents: int,
    contracts: int,
    slippage: float,
) -> tuple[float, float, float, float]:
    displayed_entry_price = displayed_entry_price_cents / 100.0
    slipped_entry_price = apply_adverse_slippage(displayed_entry_price, slippage)
    fees = calculate_kalshi_fee_dollars(slipped_entry_price, contracts)
    entry_cost = slipped_entry_price * contracts
    per_contract_fee = fees / contracts
    side_probability = predicted_yes_probability if side == "YES" else 1.0 - predicted_yes_probability
    post_cost_edge = side_probability - (slipped_entry_price + per_contract_fee)
    return post_cost_edge, entry_cost, fees, entry_cost + fees


def calculate_kelly_sizing_metrics(
    *,
    side: str,
    predicted_yes_probability: float,
    displayed_entry_price_cents: int,
    slippage: float,
    fraction_multiplier: float = 1.0,
    fraction_cap: float | None = None,
) -> KellySizingMetrics:
    if fraction_multiplier < 0:
        raise ValueError("fraction_multiplier must be non-negative")
    if fraction_cap is not None and fraction_cap < 0:
        raise ValueError("fraction_cap must be non-negative when provided")

    side_probability = predicted_yes_probability if side == "YES" else 1.0 - predicted_yes_probability
    _edge, _entry_cost, _fees, cash_required = calculate_cost_metrics(
        side=side,
        predicted_yes_probability=predicted_yes_probability,
        displayed_entry_price_cents=displayed_entry_price_cents,
        contracts=1,
        slippage=slippage,
    )
    win_profit = max(0.0, 1.0 - cash_required)
    raw_fraction = 0.0
    payoff_ratio = 0.0
    if cash_required > 0 and win_profit > 0:
        payoff_ratio = win_profit / cash_required
        raw_fraction = max(0.0, ((payoff_ratio * side_probability) - (1.0 - side_probability)) / payoff_ratio)
    scaled_fraction = max(0.0, raw_fraction * fraction_multiplier)
    capped_fraction = scaled_fraction if fraction_cap is None else min(scaled_fraction, fraction_cap)
    return KellySizingMetrics(
        side_probability=float(side_probability),
        per_contract_cash_required_dollars=float(cash_required),
        per_contract_win_profit_dollars=float(win_profit),
        payoff_ratio=float(payoff_ratio),
        raw_fraction_of_equity=float(raw_fraction),
        scaled_fraction_of_equity=float(scaled_fraction),
        capped_fraction_of_equity=float(capped_fraction),
    )


def find_max_acceptable_entry_price_cents(
    *,
    side: str,
    predicted_yes_probability: float,
    config: KalshiSignalRiskConfig,
    contracts: int | None = None,
    edge_threshold_cents: float | None = None,
) -> int | None:
    contracts_to_use = contracts if contracts is not None else config.contracts_per_order
    required_edge = config.edge_threshold if edge_threshold_cents is None else (edge_threshold_cents / 100.0)
    for price_cents in range(config.price_band_max_cents, config.price_band_min_cents - 1, -1):
        post_cost_edge, _entry_cost, _fees, _cash_required = calculate_cost_metrics(
            side=side,
            predicted_yes_probability=predicted_yes_probability,
            displayed_entry_price_cents=price_cents,
            contracts=contracts_to_use,
            slippage=config.slippage,
        )
        if post_cost_edge + 1e-12 >= required_edge:
            return price_cents
    return None


@dataclass(frozen=True)
class _SideEvaluation:
    side: str
    entry_price_cents: int
    max_acceptable_entry_price_cents: int | None
    post_cost_edge: float
    contracts: int
    entry_cost_dollars: float
    fees_dollars: float
    cash_required_dollars: float


class KalshiSignalRiskEngine:
    _EXPIRED_RESERVATION_GRACE_SECONDS = 60.0
    _CLAIM_RESERVATION_GRACE_SECONDS = 120.0

    def __init__(
        self,
        scorer: KalshiLightGBMScorer,
        config: KalshiSignalRiskConfig | None = None,
        *,
        log_dir: Path | None = None,
    ):
        self.scorer = scorer
        self.config = config or KalshiSignalRiskConfig()
        collector_config = self.scorer.feature_engine.collector.config
        self._logger = JsonlEventLogger(log_dir or (collector_config.log_dir.parent / "signal"), collector_config.environment.value)
        self._score_queue: asyncio.Queue[KalshiLightGBMScoreUpdate] | None = None
        self._states: dict[str, KalshiSignalDecisionState] = {}
        self._latest_scores: dict[str, KalshiLightGBMScoreState] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiSignalDecisionUpdate]] = []
        self._intent_queues: list[asyncio.Queue[KalshiTradeIntent]] = []
        self._intent_callbacks: list[TradeIntentCallback] = []
        self._task: asyncio.Task[Any] | None = None
        self._reservation_task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._baseline_snapshot = KalshiPortfolioSnapshot(
            event_time=utc_now(),
            available_cash_dollars=self.config.starting_cash_dollars,
            deployed_capital_dollars=0.0,
            open_positions=(),
        )
        self._pending_reservations: dict[str, _PendingReservation] = {}
        self._expired_reservations: dict[str, _ExpiredReservation] = {}
        self._local_open_positions: dict[str, KalshiPortfolioPosition] = {}
        self._active_stacking_signatures: dict[str, dict[StackingSignature, int]] = defaultdict(dict)
        self._completed_stacking_signatures: dict[str, set[StackingSignature]] = defaultdict(set)
        self._structural_regime_memory: dict[str, _StructuralRegimeMemory] = {}
        self._last_trade_opened_at: datetime | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._score_queue is None:
            self._score_queue = self.scorer.subscribe_queue()

        await self._logger.write(
            "signal_started",
            {
                "edge_threshold_cents": self.config.edge_threshold_cents,
                "maintain_edge_cents": self.config.maintain_edge_cents,
                "min_tau_minutes": self.config.min_tau_minutes,
                "max_tau_minutes": self.config.max_tau_minutes,
                "invert_model_signal": self.config.invert_model_signal,
                "allow_stacking": self.config.allow_stacking,
                "starting_cash_dollars": self.config.starting_cash_dollars,
                "contracts_per_order": self.config.contracts_per_order,
                "capital_pct_per_order": self.config.capital_pct_per_order,
                "kelly_fraction_multiplier": self.config.kelly_fraction_multiplier,
                "kelly_fraction_cap_pct": self.config.kelly_fraction_cap_pct,
                "price_band_min_cents": self.config.price_band_min_cents,
                "price_band_max_cents": self.config.price_band_max_cents,
                "quote_max_age_seconds": self.config.quote_max_age_seconds,
                "enable_low_liquidity_chop_gate": self.config.enable_low_liquidity_chop_gate,
                "low_liquidity_min_trade_count_300s": self.config.low_liquidity_min_trade_count_300s,
                "low_liquidity_min_contracts_sum_300s": self.config.low_liquidity_min_contracts_sum_300s,
                "low_liquidity_min_abs_signed_contracts_sum_300s": self.config.low_liquidity_min_abs_signed_contracts_sum_300s,
                "low_liquidity_price_volatility_300s_threshold": self.config.low_liquidity_price_volatility_300s_threshold,
                "structural_regime_refresh_seconds": self.config.structural_regime_refresh_seconds,
                "structural_regime_flip_confirmations": self.config.structural_regime_flip_confirmations,
                "trade_cooldown_seconds": self.config.trade_cooldown_seconds,
                "enable_bucket_ban_policy": self.config.enable_bucket_ban_policy,
                "auto_reserve_trade_intents": self.config.auto_reserve_trade_intents,
                "banned_yes_tau_buckets": sorted(self.config.banned_yes_tau_buckets),
                "banned_yes_price_buckets": sorted(self.config.banned_yes_price_buckets),
                "banned_yes_probability_buckets": sorted(self.config.banned_yes_probability_buckets),
                "banned_no_price_buckets": sorted(self.config.banned_no_price_buckets),
            },
        )
        await self._bootstrap_from_scorer()
        self._task = asyncio.create_task(self._consume_loop(), name="kalshi-signal-risk-engine")
        self._reservation_task = asyncio.create_task(
            self._reservation_cleanup_loop(),
            name="kalshi-signal-risk-reservations",
        )
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        for task in (self._task, self._reservation_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._reservation_task = None
        await self._logger.write("signal_stopped", {})
        close = getattr(self._logger, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def get_state(self, ticker: str) -> KalshiSignalDecisionState | None:
        return self._states.get(ticker)

    def snapshot_states(self) -> dict[str, KalshiSignalDecisionState]:
        return dict(self._states)

    def update_pending_reservation_fill_pricing(
        self,
        decision_id: str,
        *,
        fill_price_cents: int,
        contracts: int | None = None,
        entry_cost_dollars: float | None = None,
        fees_dollars: float | None = None,
        cash_required_dollars: float | None = None,
    ) -> None:
        reservation, _reservation_source = self._lookup_reservation(decision_id)
        if reservation is None:
            raise KeyError(f"Unknown reservation decision id: {decision_id}")
        contracts_to_use = reservation.contracts if contracts is None else max(0, contracts)
        if entry_cost_dollars is None or fees_dollars is None or cash_required_dollars is None:
            entry_cost_dollars, fees_dollars, cash_required_dollars = calculate_realized_cash_metrics(
                entry_price_cents=fill_price_cents,
                contracts=contracts_to_use,
            )
        reservation.contracts = contracts_to_use
        reservation.entry_cost_dollars = entry_cost_dollars
        reservation.fees_dollars = fees_dollars
        reservation.cash_required_dollars = cash_required_dollars

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiSignalDecisionUpdate]:
        queue: asyncio.Queue[KalshiSignalDecisionUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    def subscribe_trade_intent_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiTradeIntent]:
        queue: asyncio.Queue[KalshiTradeIntent] = asyncio.Queue(maxsize=maxsize)
        self._intent_queues.append(queue)
        return queue

    def subscribe_trade_intent_callback(self, callback: TradeIntentCallback) -> None:
        self._intent_callbacks.append(callback)

    def unsubscribe_trade_intent_callback(self, callback: TradeIntentCallback) -> None:
        try:
            self._intent_callbacks.remove(callback)
        except ValueError:
            pass

    def get_portfolio_state(self) -> KalshiSignalPortfolioState:
        available_cash, deployed_capital, equity, reserved_cash = self._portfolio_metrics()
        open_positions = tuple(self._combined_open_positions())
        pending_reservations = tuple(
            reservation.as_position()
            for reservation in sorted(self._pending_reservations.values(), key=lambda item: item.created_at)
        )
        return KalshiSignalPortfolioState(
            event_time=utc_now(),
            baseline_available_cash_dollars=self._baseline_snapshot.available_cash_dollars,
            baseline_deployed_capital_dollars=self._baseline_snapshot.deployed_capital_dollars,
            available_cash_dollars=available_cash,
            deployed_capital_dollars=deployed_capital,
            equity_dollars=equity,
            reserved_cash_dollars=reserved_cash,
            open_positions=open_positions,
            pending_reservations=pending_reservations,
        )

    def claim_trade_intent_reservation(self, intent: KalshiTradeIntent) -> None:
        reservation, reservation_source = self._lookup_reservation(intent.decision_id)
        extend_until = utc_now() + timedelta(seconds=self._CLAIM_RESERVATION_GRACE_SECONDS)
        if reservation is None:
            reservation = _PendingReservation(
                decision_id=intent.decision_id,
                ticker=intent.ticker,
                side=intent.side,
                contracts=intent.contracts,
                entry_cost_dollars=intent.estimated_entry_cost_dollars,
                fees_dollars=intent.estimated_fees_dollars,
                cash_required_dollars=intent.estimated_cash_required_dollars,
                created_at=intent.generated_at,
                expires_at=extend_until,
                is_acknowledged=False,
                stacking_signature=intent.stacking_signature,
            )
            self._pending_reservations[intent.decision_id] = reservation
            self._mark_stacking_signature_active(reservation)
            return

        if reservation_source == "expired":
            self._expired_reservations.pop(intent.decision_id, None)
            self._pending_reservations[intent.decision_id] = reservation
            self._mark_stacking_signature_active(reservation)

        reservation.entry_cost_dollars = intent.estimated_entry_cost_dollars
        reservation.fees_dollars = intent.estimated_fees_dollars
        reservation.cash_required_dollars = intent.estimated_cash_required_dollars
        if reservation.stacking_signature is None and intent.stacking_signature is not None:
            reservation.stacking_signature = intent.stacking_signature
            self._mark_stacking_signature_active(reservation)
        if reservation.expires_at is None or reservation.expires_at < extend_until:
            reservation.expires_at = extend_until

    def resolve_manual_trade_contracts(
        self,
        *,
        decision_state: KalshiSignalDecisionState | KalshiSignalDecisionUpdate,
        side: str,
        entry_price_cents: int,
        cash_budget_dollars: float | None = None,
    ) -> int:
        if entry_price_cents <= 0:
            return 0
        if cash_budget_dollars is not None:
            if cash_budget_dollars <= 0:
                return 0
            return self._max_contracts_for_cash_budget(
                side=side,
                predicted_yes_probability=decision_state.predicted_yes_probability,
                entry_price_cents=entry_price_cents,
                cash_budget=cash_budget_dollars,
            )
        return self._resolve_contracts(
            side=side,
            predicted_yes_probability=decision_state.predicted_yes_probability,
            entry_price_cents=entry_price_cents,
        )

    def reserve_manual_trade_intent(
        self,
        *,
        decision_state: KalshiSignalDecisionState | KalshiSignalDecisionUpdate,
        side: str,
        entry_price_cents: int,
        cash_budget_dollars: float | None = None,
        contracts: int | None = None,
        allow_ticker_lock_bypass: bool = False,
        ignore_trade_cooldown: bool = True,
        stacking_signature: StackingSignature | None = None,
        thesis_id: str | None = None,
        tranche_index: int | None = None,
        tranche_window: str | None = None,
        tranche_reason: str | None = None,
        lifecycle_state: str | None = None,
        total_thesis_budget_dollars: float | None = None,
        payout_if_yes_dollars: float | None = None,
        payout_if_no_dollars: float | None = None,
        expected_value_dollars: float | None = None,
        worst_case_loss_dollars: float | None = None,
        allow_unapproved_retry: bool = False,
        ignore_post_cost_edge_threshold: bool = False,
        max_acceptable_entry_price_cents_override: int | None = None,
        target_id: str | None = None,
        attempt_index: int | None = None,
        desired_contracts: int | None = None,
        remaining_contracts_before_submit: int | None = None,
        hard_max_price_cents: int | None = None,
        retry_reason: str | None = None,
        was_first_attempt: bool | None = None,
    ) -> KalshiTradeIntent | None:
        if not allow_unapproved_retry and (not decision_state.approved or decision_state.side is None):
            return None
        normalized_side = side.upper()
        if normalized_side not in {"YES", "NO"}:
            raise ValueError(f"Unsupported manual trade side: {side}")
        if entry_price_cents < self.config.price_band_min_cents or entry_price_cents > self.config.price_band_max_cents:
            return None
        if not allow_ticker_lock_bypass and not self.config.allow_stacking and self._ticker_is_locked(decision_state.ticker):
            return None
        if not ignore_trade_cooldown and self._cooldown_is_active():
            return None

        resolved_contracts = contracts
        if resolved_contracts is None:
            resolved_contracts = self.resolve_manual_trade_contracts(
                decision_state=decision_state,
                side=normalized_side,
                entry_price_cents=entry_price_cents,
                cash_budget_dollars=cash_budget_dollars,
            )
        if resolved_contracts is None or resolved_contracts <= 0:
            return None

        max_acceptable_entry_price_cents = (
            max_acceptable_entry_price_cents_override
            if max_acceptable_entry_price_cents_override is not None
            else find_max_acceptable_entry_price_cents(
                side=normalized_side,
                predicted_yes_probability=decision_state.predicted_yes_probability,
                config=self.config,
                contracts=resolved_contracts,
            )
        )
        post_cost_edge, entry_cost_dollars, fees_dollars, cash_required_dollars = calculate_cost_metrics(
            side=normalized_side,
            predicted_yes_probability=decision_state.predicted_yes_probability,
            displayed_entry_price_cents=entry_price_cents,
            contracts=resolved_contracts,
            slippage=self.config.slippage,
        )
        if (
            max_acceptable_entry_price_cents is None
            or entry_price_cents > max_acceptable_entry_price_cents
            or (
                not ignore_post_cost_edge_threshold
                and post_cost_edge + 1e-12 < self.config.edge_threshold
            )
        ):
            return None
        if cash_budget_dollars is not None and cash_required_dollars > cash_budget_dollars + 1e-12:
            return None

        available_cash, _deployed_capital, equity, _reserved_cash = self._portfolio_metrics()
        reserve_cash = (self.config.reserve_cash_pct / 100.0) * equity
        deployable_cash = max(0.0, available_cash - reserve_cash)
        if cash_required_dollars > deployable_cash + 1e-12:
            return None

        candidate = _DecisionCandidate(
            ticker=decision_state.ticker,
            event_time=decision_state.event_time,
            approved=True,
            side=normalized_side,
            predicted_yes_probability=decision_state.predicted_yes_probability,
            predicted_no_probability=decision_state.predicted_no_probability,
            feature_basis_market_prob=decision_state.feature_basis_market_prob,
            raw_model_edge=decision_state.raw_model_edge,
            post_cost_edge=post_cost_edge,
            yes_post_cost_edge=(
                post_cost_edge if normalized_side == "YES" else decision_state.yes_post_cost_edge
            ),
            no_post_cost_edge=(
                post_cost_edge if normalized_side == "NO" else decision_state.no_post_cost_edge
            ),
            tau_minutes=decision_state.tau_minutes,
            reference_price_cents=entry_price_cents,
            max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
            last_yes_price_cents=decision_state.last_yes_price_cents,
            yes_bid_cents=decision_state.yes_bid_cents,
            yes_ask_cents=decision_state.yes_ask_cents,
            buy_yes_price_cents=decision_state.buy_yes_price_cents,
            buy_no_price_cents=decision_state.buy_no_price_cents,
            quote_mid_prob=decision_state.quote_mid_prob,
            quote_spread_cents=decision_state.quote_spread_cents,
            quote_age_seconds=decision_state.quote_age_seconds,
            regime_label=decision_state.regime_label,
            bearish_vote_count=decision_state.bearish_vote_count,
            bullish_vote_count=decision_state.bullish_vote_count,
            regime_price_momentum_bearish=decision_state.regime_price_momentum_bearish,
            regime_signed_flow_bearish=decision_state.regime_signed_flow_bearish,
            regime_yes_share_bearish=decision_state.regime_yes_share_bearish,
            regime_price_momentum_bullish=decision_state.regime_price_momentum_bullish,
            regime_signed_flow_bullish=decision_state.regime_signed_flow_bullish,
            regime_yes_share_bullish=decision_state.regime_yes_share_bullish,
            tau_bucket=decision_state.tau_bucket,
            price_bucket=decision_state.price_bucket,
            chosen_side_probability_bucket=decision_state.chosen_side_probability_bucket,
            chosen_side_edge_bucket=decision_state.chosen_side_edge_bucket,
            bucket_policy_dimension=decision_state.bucket_policy_dimension,
            bucket_policy_bucket=decision_state.bucket_policy_bucket,
            bucket_policy_side=decision_state.bucket_policy_side,
            block_reason=None,
            contracts=resolved_contracts,
            entry_cost_dollars=entry_cost_dollars,
            fees_dollars=fees_dollars,
            cash_required_dollars=cash_required_dollars,
        )
        intent = self._reserve_trade_intent(candidate, stacking_signature=stacking_signature)
        return replace(
            intent,
            thesis_id=thesis_id,
            tranche_index=tranche_index,
            tranche_window=tranche_window,
            tranche_reason=tranche_reason,
            lifecycle_state=lifecycle_state,
            total_thesis_budget_dollars=total_thesis_budget_dollars,
            payout_if_yes_dollars=payout_if_yes_dollars,
            payout_if_no_dollars=payout_if_no_dollars,
            expected_value_dollars=expected_value_dollars,
            worst_case_loss_dollars=worst_case_loss_dollars,
            target_id=target_id,
            attempt_index=attempt_index,
            desired_contracts=desired_contracts,
            remaining_contracts_before_submit=remaining_contracts_before_submit,
            hard_max_price_cents=(
                max_acceptable_entry_price_cents if hard_max_price_cents is None else hard_max_price_cents
            ),
            retry_reason=retry_reason,
            was_first_attempt=was_first_attempt,
        )

    async def apply_portfolio_snapshot(self, snapshot: KalshiPortfolioSnapshot) -> None:
        expired_tickers = self._expire_reservations()
        self._baseline_snapshot = snapshot
        # Authoritative snapshots replace the shadow book's open-position baseline.
        self._local_open_positions.clear()
        affected_tickers = set(self._latest_scores) | set(expired_tickers)
        await self._reevaluate_tickers(affected_tickers)

    async def apply_execution_feedback(self, feedback: KalshiExecutionFeedback) -> None:
        expired_tickers = self._expire_reservations()
        affected_tickers = set(expired_tickers)

        if feedback.status not in {"accepted", "rejected", "cancelled", "filled", "released"}:
            raise ValueError(f"Unsupported execution feedback status: {feedback.status}")

        reservation, reservation_source = self._lookup_reservation(feedback.decision_id)
        local_open_position = self._local_open_positions.get(feedback.decision_id)

        if feedback.status == "accepted":
            if reservation is None:
                raise KeyError(f"Unknown reservation decision id: {feedback.decision_id}")
            if reservation_source == "expired":
                self._expired_reservations.pop(feedback.decision_id, None)
                self._pending_reservations[feedback.decision_id] = reservation
                self._mark_stacking_signature_active(reservation)
            reservation.is_acknowledged = True
            reservation.expires_at = None
            affected_tickers.add(reservation.ticker)
            self._states.pop(reservation.ticker, None)
        elif feedback.status in {"rejected", "cancelled"}:
            if reservation is None:
                raise KeyError(f"Unknown reservation decision id: {feedback.decision_id}")
            if reservation_source == "expired":
                self._expired_reservations.pop(feedback.decision_id, None)
            else:
                self._pending_reservations.pop(feedback.decision_id, None)
            self._release_stacking_signature_active(reservation)
            affected_tickers.add(reservation.ticker)
            self._states.pop(reservation.ticker, None)
        elif feedback.status == "filled":
            if reservation is None:
                raise KeyError(f"Unknown reservation decision id: {feedback.decision_id}")
            if reservation_source == "expired":
                self._expired_reservations.pop(feedback.decision_id, None)
            else:
                self._pending_reservations.pop(feedback.decision_id, None)
            self._release_stacking_signature_active(reservation)
            self._mark_stacking_signature_completed(reservation)
            self._local_open_positions[feedback.decision_id] = KalshiPortfolioPosition(
                ticker=reservation.ticker,
                side=reservation.side,
                contracts=feedback.filled_contracts or reservation.contracts,
                entry_cost_dollars=reservation.entry_cost_dollars,
                fees_dollars=reservation.fees_dollars,
                cash_required_dollars=reservation.cash_required_dollars,
                decision_id=reservation.decision_id,
            )
            affected_tickers.add(reservation.ticker)
            self._states.pop(reservation.ticker, None)
        else:
            if reservation is not None:
                if reservation_source == "expired":
                    self._expired_reservations.pop(feedback.decision_id, None)
                else:
                    self._pending_reservations.pop(feedback.decision_id, None)
                self._release_stacking_signature_active(reservation)
                affected_tickers.add(reservation.ticker)
                self._states.pop(reservation.ticker, None)
            elif local_open_position is not None:
                self._local_open_positions.pop(feedback.decision_id, None)
                affected_tickers.add(local_open_position.ticker)
                self._states.pop(local_open_position.ticker, None)
                if feedback.cash_delta_dollars is not None:
                    self._latest_scores.pop(local_open_position.ticker, None)
            else:
                raise KeyError(f"Unknown decision id: {feedback.decision_id}")

            if feedback.cash_delta_dollars is not None:
                self._baseline_snapshot = KalshiPortfolioSnapshot(
                    event_time=feedback.event_time,
                    available_cash_dollars=self._baseline_snapshot.available_cash_dollars + feedback.cash_delta_dollars,
                    deployed_capital_dollars=self._baseline_snapshot.deployed_capital_dollars,
                    open_positions=self._baseline_snapshot.open_positions,
                )

        await self._reevaluate_tickers(affected_tickers)

    async def _bootstrap_from_scorer(self) -> None:
        for score_state in self.scorer.snapshot_states().values():
            self._latest_scores[score_state.ticker] = score_state
        await self._reevaluate_tickers(self._latest_scores)

    async def _consume_loop(self) -> None:
        if self._score_queue is None:
            return

        try:
            while not self._stop_event.is_set():
                score_update = await self._score_queue.get()
                self._expire_reservations()
                self._latest_scores[score_update.ticker] = KalshiLightGBMScoreState(**score_update.__dict__)
                await self._reevaluate_tickers((score_update.ticker,))
        except asyncio.CancelledError:
            raise

    async def _reservation_cleanup_loop(self) -> None:
        interval_seconds = min(1.0, max(0.1, self.config.reservation_ttl_seconds / 2.0))
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(interval_seconds)
                self._expire_reservations()
        except asyncio.CancelledError:
            raise

    def _expire_reservations(self) -> set[str]:
        now = utc_now()
        expired_tickers: set[str] = set()
        self._prune_expired_reservations(now)
        expired_ids = [
            decision_id
            for decision_id, reservation in self._pending_reservations.items()
            if reservation.expires_at is not None and reservation.expires_at <= now
        ]
        for decision_id in expired_ids:
            reservation = self._pending_reservations.pop(decision_id, None)
            if reservation is None:
                continue
            self._release_stacking_signature_active(reservation)
            self._expired_reservations[decision_id] = _ExpiredReservation(reservation=reservation, expired_at=now)
            expired_tickers.add(reservation.ticker)
            self._states.pop(reservation.ticker, None)
        return expired_tickers

    def _lookup_reservation(self, decision_id: str) -> tuple[_PendingReservation | None, str | None]:
        reservation = self._pending_reservations.get(decision_id)
        if reservation is not None:
            return reservation, "pending"
        expired = self._expired_reservations.get(decision_id)
        if expired is None:
            return None, None
        return expired.reservation, "expired"

    def _prune_expired_reservations(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._EXPIRED_RESERVATION_GRACE_SECONDS)
        stale_ids = [
            decision_id
            for decision_id, expired in self._expired_reservations.items()
            if expired.expired_at <= cutoff
        ]
        for decision_id in stale_ids:
            self._expired_reservations.pop(decision_id, None)

    async def _reevaluate_tickers(self, tickers: Iterable[str] | dict[str, Any]) -> None:
        for ticker in sorted(set(tickers)):
            score_state = self._latest_scores.get(ticker)
            if score_state is None:
                continue
            await self._evaluate_score_state(score_state)

    async def _evaluate_score_state(self, score_state: KalshiLightGBMScoreState) -> None:
        candidate = self._build_candidate(score_state)
        current_state = self._states.get(score_state.ticker)
        if current_state is not None and self._decision_signature(current_state) == self._candidate_signature(candidate):
            return

        if candidate.approved:
            trade_intent = self._reserve_trade_intent(candidate) if self.config.auto_reserve_trade_intents else None
            next_state = KalshiSignalDecisionState(
                ticker=candidate.ticker,
                event_time=candidate.event_time,
                approved=True,
                side=candidate.side,
                predicted_yes_probability=candidate.predicted_yes_probability,
                predicted_no_probability=candidate.predicted_no_probability,
                feature_basis_market_prob=candidate.feature_basis_market_prob,
                raw_model_edge=candidate.raw_model_edge,
                post_cost_edge=candidate.post_cost_edge,
                yes_post_cost_edge=candidate.yes_post_cost_edge,
                no_post_cost_edge=candidate.no_post_cost_edge,
                tau_minutes=candidate.tau_minutes,
                reference_price_cents=candidate.reference_price_cents,
                max_acceptable_entry_price_cents=candidate.max_acceptable_entry_price_cents,
                last_yes_price_cents=candidate.last_yes_price_cents,
                yes_bid_cents=candidate.yes_bid_cents,
                yes_ask_cents=candidate.yes_ask_cents,
                buy_yes_price_cents=candidate.buy_yes_price_cents,
                buy_no_price_cents=candidate.buy_no_price_cents,
                quote_mid_prob=candidate.quote_mid_prob,
                quote_spread_cents=candidate.quote_spread_cents,
                quote_age_seconds=candidate.quote_age_seconds,
                regime_label=candidate.regime_label,
                bearish_vote_count=candidate.bearish_vote_count,
                bullish_vote_count=candidate.bullish_vote_count,
                regime_price_momentum_bearish=candidate.regime_price_momentum_bearish,
                regime_signed_flow_bearish=candidate.regime_signed_flow_bearish,
                regime_yes_share_bearish=candidate.regime_yes_share_bearish,
                regime_price_momentum_bullish=candidate.regime_price_momentum_bullish,
                regime_signed_flow_bullish=candidate.regime_signed_flow_bullish,
                regime_yes_share_bullish=candidate.regime_yes_share_bullish,
                tau_bucket=candidate.tau_bucket,
                price_bucket=candidate.price_bucket,
                chosen_side_probability_bucket=candidate.chosen_side_probability_bucket,
                chosen_side_edge_bucket=candidate.chosen_side_edge_bucket,
                bucket_policy_dimension=candidate.bucket_policy_dimension,
                bucket_policy_bucket=candidate.bucket_policy_bucket,
                bucket_policy_side=candidate.bucket_policy_side,
                block_reason=None,
                trade_intent=trade_intent,
            )
        else:
            next_state = KalshiSignalDecisionState(
                ticker=candidate.ticker,
                event_time=candidate.event_time,
                approved=False,
                side=candidate.side,
                predicted_yes_probability=candidate.predicted_yes_probability,
                predicted_no_probability=candidate.predicted_no_probability,
                feature_basis_market_prob=candidate.feature_basis_market_prob,
                raw_model_edge=candidate.raw_model_edge,
                post_cost_edge=candidate.post_cost_edge,
                yes_post_cost_edge=candidate.yes_post_cost_edge,
                no_post_cost_edge=candidate.no_post_cost_edge,
                tau_minutes=candidate.tau_minutes,
                reference_price_cents=candidate.reference_price_cents,
                max_acceptable_entry_price_cents=candidate.max_acceptable_entry_price_cents,
                last_yes_price_cents=candidate.last_yes_price_cents,
                yes_bid_cents=candidate.yes_bid_cents,
                yes_ask_cents=candidate.yes_ask_cents,
                buy_yes_price_cents=candidate.buy_yes_price_cents,
                buy_no_price_cents=candidate.buy_no_price_cents,
                quote_mid_prob=candidate.quote_mid_prob,
                quote_spread_cents=candidate.quote_spread_cents,
                quote_age_seconds=candidate.quote_age_seconds,
                regime_label=candidate.regime_label,
                bearish_vote_count=candidate.bearish_vote_count,
                bullish_vote_count=candidate.bullish_vote_count,
                regime_price_momentum_bearish=candidate.regime_price_momentum_bearish,
                regime_signed_flow_bearish=candidate.regime_signed_flow_bearish,
                regime_yes_share_bearish=candidate.regime_yes_share_bearish,
                regime_price_momentum_bullish=candidate.regime_price_momentum_bullish,
                regime_signed_flow_bullish=candidate.regime_signed_flow_bullish,
                regime_yes_share_bullish=candidate.regime_yes_share_bullish,
                tau_bucket=candidate.tau_bucket,
                price_bucket=candidate.price_bucket,
                chosen_side_probability_bucket=candidate.chosen_side_probability_bucket,
                chosen_side_edge_bucket=candidate.chosen_side_edge_bucket,
                bucket_policy_dimension=candidate.bucket_policy_dimension,
                bucket_policy_bucket=candidate.bucket_policy_bucket,
                bucket_policy_side=candidate.bucket_policy_side,
                block_reason=candidate.block_reason,
                trade_intent=None,
            )

        self._states[next_state.ticker] = next_state
        await self._publish_update(decision_update_from_state(next_state))

    def _feature_state_for_score(self, score_state: KalshiLightGBMScoreState) -> Any | None:
        feature_engine = getattr(self.scorer, "feature_engine", None)
        get_state = getattr(feature_engine, "get_state", None)
        if not callable(get_state):
            return None
        try:
            return get_state(score_state.ticker)
        except Exception:
            return None

    def _resolve_regime(
        self,
        score_state: KalshiLightGBMScoreState,
        *,
        feature_state: Any | None = None,
    ) -> KalshiRegimeEvaluation:
        resolved_feature_state = feature_state if feature_state is not None else self._feature_state_for_score(score_state)
        if resolved_feature_state is None:
            return evaluate_kxbtc15m_regime_for_state(score_state)
        raw_structural = evaluate_kxbtc15m_structural_regime_for_state(resolved_feature_state)
        if raw_structural.regime_label == "unknown":
            return evaluate_kxbtc15m_regime_for_state(score_state)
        return self._stable_structural_regime(
            ticker=score_state.ticker,
            event_time=score_state.event_time,
            raw_regime=raw_structural,
        )

    def _stable_structural_regime(
        self,
        *,
        ticker: str,
        event_time: datetime,
        raw_regime: KalshiRegimeEvaluation,
    ) -> KalshiRegimeEvaluation:
        memory = self._structural_regime_memory.setdefault(ticker, _StructuralRegimeMemory())
        if (
            memory.last_refreshed_at is not None
            and memory.last_evaluation is not None
            and (event_time - memory.last_refreshed_at).total_seconds() < self.config.structural_regime_refresh_seconds
        ):
            return memory.last_evaluation

        raw_label = raw_regime.regime_label
        resolved_label = raw_label
        if memory.current_label is None or memory.current_label == raw_label:
            memory.current_label = raw_label
            memory.pending_label = None
            memory.pending_count = 0
            resolved_label = raw_label
        else:
            if memory.pending_label == raw_label:
                memory.pending_count += 1
            else:
                memory.pending_label = raw_label
                memory.pending_count = 1
            if memory.pending_count >= self.config.structural_regime_flip_confirmations:
                memory.current_label = raw_label
                memory.pending_label = None
                memory.pending_count = 0
            resolved_label = memory.current_label

        resolved = replace(raw_regime, regime_label=resolved_label)
        memory.last_refreshed_at = event_time
        memory.last_evaluation = resolved
        return resolved

    def _low_liquidity_chop_reason(
        self,
        *,
        feature_state: Any | None,
    ) -> str | None:
        if not self.config.enable_low_liquidity_chop_gate or feature_state is None:
            return None
        trade_count_300s = _optional_float(getattr(feature_state, "trade_count_300s", None))
        contracts_sum_300s = _optional_float(getattr(feature_state, "contracts_sum_300s", None))
        signed_contracts_sum_300s = _optional_float(getattr(feature_state, "signed_contracts_sum_300s", None))
        price_volatility_300s = _optional_float(getattr(feature_state, "price_volatility_300s", None))

        if (
            trade_count_300s is None
            and contracts_sum_300s is None
            and signed_contracts_sum_300s is None
            and price_volatility_300s is None
        ):
            return None

        low_activity = (
            trade_count_300s is not None
            and contracts_sum_300s is not None
            and trade_count_300s < self.config.low_liquidity_min_trade_count_300s
            and contracts_sum_300s < self.config.low_liquidity_min_contracts_sum_300s
        )
        choppy_low_liquidity = (
            contracts_sum_300s is not None
            and price_volatility_300s is not None
            and contracts_sum_300s < self.config.low_liquidity_min_contracts_sum_300s
            and price_volatility_300s > self.config.low_liquidity_price_volatility_300s_threshold
        )
        weak_directional_flow = (
            signed_contracts_sum_300s is not None
            and price_volatility_300s is not None
            and abs(signed_contracts_sum_300s) < self.config.low_liquidity_min_abs_signed_contracts_sum_300s
            and price_volatility_300s > self.config.low_liquidity_price_volatility_300s_threshold
        )
        if low_activity or choppy_low_liquidity or weak_directional_flow:
            return "low_liquidity_chop"
        return None

    def _build_candidate(self, score_state: KalshiLightGBMScoreState) -> _DecisionCandidate:
        predicted_yes_probability = score_state.predicted_yes_probability
        predicted_no_probability = 1.0 - predicted_yes_probability
        raw_model_edge = predicted_yes_probability - score_state.market_prob
        feature_state = self._feature_state_for_score(score_state)
        regime = self._resolve_regime(score_state, feature_state=feature_state)
        buy_yes_price_cents = score_state.buy_yes_price_cents
        buy_no_price_cents = score_state.buy_no_price_cents

        yes_post_cost_edge: float | None = None
        no_post_cost_edge: float | None = None
        if buy_yes_price_cents is not None:
            yes_post_cost_edge, _yes_entry_cost, _yes_fees, _yes_cash = calculate_cost_metrics(
                side="YES",
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=buy_yes_price_cents,
                contracts=1,
                slippage=self.config.slippage,
            )
        if buy_no_price_cents is not None:
            no_post_cost_edge, _no_entry_cost, _no_fees, _no_cash = calculate_cost_metrics(
                side="NO",
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=buy_no_price_cents,
                contracts=1,
                slippage=self.config.slippage,
            )

        if score_state.tau_minutes < self.config.min_tau_minutes or score_state.tau_minutes > self.config.max_tau_minutes:
            return self._blocked_candidate(
                score_state,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                block_reason="outside_tau_window",
            )

        quote_rejection_reason = self._quote_rejection_reason(score_state)
        if quote_rejection_reason is not None:
            return self._blocked_candidate(
                score_state,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                block_reason=quote_rejection_reason,
            )

        low_liquidity_chop_reason = self._low_liquidity_chop_reason(feature_state=feature_state)
        if low_liquidity_chop_reason is not None:
            return self._blocked_candidate(
                score_state,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                block_reason=low_liquidity_chop_reason,
                regime=regime,
            )

        yes_evaluation = self._evaluate_side_candidate(
            score_state,
            side="YES",
            entry_price_cents=buy_yes_price_cents or 0,
        )
        no_evaluation = self._evaluate_side_candidate(
            score_state,
            side="NO",
            entry_price_cents=buy_no_price_cents or 0,
        )
        chosen_evaluation = self._select_side_evaluation(
            yes_evaluation=yes_evaluation,
            no_evaluation=no_evaluation,
            invert=self.config.invert_model_signal,
        )
        if chosen_evaluation is None:
            return self._blocked_candidate(
                score_state,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                block_reason=self._rejection_reason(
                    score_state,
                    raw_model_edge=raw_model_edge,
                    yes_post_cost_edge=yes_post_cost_edge,
                    no_post_cost_edge=no_post_cost_edge,
                    yes_evaluation=yes_evaluation,
                    no_evaluation=no_evaluation,
                ),
            )

        side = chosen_evaluation.side
        chosen_buckets = build_chosen_side_buckets(
            side=side,
            tau_minutes=score_state.tau_minutes,
            entry_price_cents=chosen_evaluation.entry_price_cents,
            predicted_yes_probability=predicted_yes_probability,
            chosen_edge_cents=chosen_evaluation.post_cost_edge * 100.0,
        )
        stacking_signature = self._build_stacking_signature(side=side, chosen_buckets=chosen_buckets)
        reference_price_cents = chosen_evaluation.entry_price_cents
        max_acceptable_entry_price_cents = chosen_evaluation.max_acceptable_entry_price_cents
        post_cost_edge = chosen_evaluation.post_cost_edge
        contracts = chosen_evaluation.contracts
        entry_cost_dollars = chosen_evaluation.entry_cost_dollars
        fees_dollars = chosen_evaluation.fees_dollars
        cash_required_dollars = chosen_evaluation.cash_required_dollars

        if self.config.allow_stacking and self._stacking_signature_is_seen(score_state.ticker, stacking_signature):
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="duplicate_signature",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        bucket_policy = evaluate_bucket_ban_policy(
            enabled=self.config.enable_bucket_ban_policy,
            side=side,
            buckets=chosen_buckets,
            banned_yes_tau_buckets=self.config.banned_yes_tau_buckets,
            banned_yes_price_buckets=self.config.banned_yes_price_buckets,
            banned_yes_probability_buckets=self.config.banned_yes_probability_buckets,
            banned_no_price_buckets=self.config.banned_no_price_buckets,
        )
        if bucket_policy.is_blocked:
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="blocked_by_bucket_policy",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
                bucket_policy=bucket_policy,
            )

        combo_policy = evaluate_combo_ban_policy(
            enabled=self.config.enable_combo_ban_policy,
            side=side,
            buckets=chosen_buckets,
            banned_combo_buckets=self.config.banned_combo_buckets,
        )
        if combo_policy.is_blocked:
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="blocked_by_combo_policy",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
                bucket_policy=combo_policy,
            )

        if regime.regime_label in self.config.blocked_regime_labels:
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason=f"blocked_by_regime_{regime.regime_label}",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        if self.config.apply_regime_hard_gate and side == "YES" and regime.regime_label == "downtrend":
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="blocked_by_regime_downtrend",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        if not self.config.allow_stacking and self._ticker_is_locked(score_state.ticker):
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="ticker_locked",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        if self._cooldown_is_active():
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="cooldown_active",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        available_cash, _deployed_capital, equity, _reserved_cash = self._portfolio_metrics()
        if available_cash + 1e-12 < cash_required_dollars:
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="insufficient_cash",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        reserve_cash = (self.config.reserve_cash_pct / 100.0) * equity
        if available_cash - cash_required_dollars < reserve_cash - 1e-12:
            return self._blocked_candidate(
                score_state,
                side=side,
                predicted_no_probability=predicted_no_probability,
                raw_model_edge=raw_model_edge,
                post_cost_edge=post_cost_edge,
                yes_post_cost_edge=yes_post_cost_edge,
                no_post_cost_edge=no_post_cost_edge,
                reference_price_cents=reference_price_cents,
                max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
                block_reason="reserve_violation",
                contracts=contracts,
                entry_cost_dollars=entry_cost_dollars,
                fees_dollars=fees_dollars,
                cash_required_dollars=cash_required_dollars,
                regime=regime,
                chosen_buckets=chosen_buckets,
            )

        return _DecisionCandidate(
            ticker=score_state.ticker,
            event_time=score_state.event_time,
            approved=True,
            side=side,
            predicted_yes_probability=predicted_yes_probability,
            predicted_no_probability=predicted_no_probability,
            feature_basis_market_prob=score_state.market_prob,
            raw_model_edge=raw_model_edge,
            post_cost_edge=post_cost_edge,
            yes_post_cost_edge=yes_post_cost_edge,
            no_post_cost_edge=no_post_cost_edge,
            tau_minutes=score_state.tau_minutes,
            reference_price_cents=reference_price_cents,
            max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
            last_yes_price_cents=score_state.last_yes_price_cents,
            yes_bid_cents=score_state.yes_bid_cents,
            yes_ask_cents=score_state.yes_ask_cents,
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
            bucket_policy_dimension=None,
            bucket_policy_bucket=None,
            bucket_policy_side=None,
            block_reason=None,
            contracts=contracts,
            entry_cost_dollars=entry_cost_dollars,
            fees_dollars=fees_dollars,
            cash_required_dollars=cash_required_dollars,
        )

    def _blocked_candidate(
        self,
        score_state: KalshiLightGBMScoreState,
        *,
        block_reason: str,
        side: str | None = None,
        predicted_no_probability: float | None = None,
        raw_model_edge: float | None = None,
        post_cost_edge: float | None = None,
        yes_post_cost_edge: float | None = None,
        no_post_cost_edge: float | None = None,
        reference_price_cents: int | None = None,
        max_acceptable_entry_price_cents: int | None = None,
        contracts: int | None = None,
        entry_cost_dollars: float = 0.0,
        fees_dollars: float = 0.0,
        cash_required_dollars: float = 0.0,
        regime: KalshiRegimeEvaluation | None = None,
        chosen_buckets: Any | None = None,
        bucket_policy: KalshiBucketPolicyEvaluation | None = None,
    ) -> _DecisionCandidate:
        resolved_regime = regime or evaluate_kxbtc15m_regime_for_state(score_state)
        if chosen_buckets is None:
            chosen_buckets = build_chosen_side_buckets(
                side=side,
                tau_minutes=score_state.tau_minutes,
                entry_price_cents=reference_price_cents,
                predicted_yes_probability=score_state.predicted_yes_probability,
                chosen_edge_cents=None if post_cost_edge is None else post_cost_edge * 100.0,
            )
        return _DecisionCandidate(
            ticker=score_state.ticker,
            event_time=score_state.event_time,
            approved=False,
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            predicted_no_probability=(
                (1.0 - score_state.predicted_yes_probability)
                if predicted_no_probability is None
                else predicted_no_probability
            ),
            feature_basis_market_prob=score_state.market_prob,
            raw_model_edge=(
                (score_state.predicted_yes_probability - score_state.market_prob)
                if raw_model_edge is None
                else raw_model_edge
            ),
            post_cost_edge=post_cost_edge,
            yes_post_cost_edge=yes_post_cost_edge,
            no_post_cost_edge=no_post_cost_edge,
            tau_minutes=score_state.tau_minutes,
            reference_price_cents=reference_price_cents,
            max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
            last_yes_price_cents=score_state.last_yes_price_cents,
            yes_bid_cents=score_state.yes_bid_cents,
            yes_ask_cents=score_state.yes_ask_cents,
            buy_yes_price_cents=score_state.buy_yes_price_cents,
            buy_no_price_cents=score_state.buy_no_price_cents,
            quote_mid_prob=score_state.quote_mid_prob,
            quote_spread_cents=score_state.quote_spread_cents,
            quote_age_seconds=score_state.quote_age_seconds,
            regime_label=resolved_regime.regime_label,
            bearish_vote_count=resolved_regime.bearish_vote_count,
            bullish_vote_count=resolved_regime.bullish_vote_count,
            regime_price_momentum_bearish=resolved_regime.price_momentum_bearish,
            regime_signed_flow_bearish=resolved_regime.signed_flow_bearish,
            regime_yes_share_bearish=resolved_regime.yes_share_bearish,
            regime_price_momentum_bullish=resolved_regime.price_momentum_bullish,
            regime_signed_flow_bullish=resolved_regime.signed_flow_bullish,
            regime_yes_share_bullish=resolved_regime.yes_share_bullish,
            tau_bucket=chosen_buckets.tau_bucket,
            price_bucket=chosen_buckets.price_bucket,
            chosen_side_probability_bucket=chosen_buckets.chosen_side_probability_bucket,
            chosen_side_edge_bucket=chosen_buckets.chosen_side_edge_bucket,
            bucket_policy_dimension=None if bucket_policy is None else bucket_policy.blocked_dimension,
            bucket_policy_bucket=None if bucket_policy is None else bucket_policy.blocked_bucket,
            bucket_policy_side=None if bucket_policy is None else bucket_policy.blocked_side,
            block_reason=block_reason,
            contracts=self._default_contract_count() if contracts is None else contracts,
            entry_cost_dollars=entry_cost_dollars,
            fees_dollars=fees_dollars,
            cash_required_dollars=cash_required_dollars,
        )

    def _quote_rejection_reason(self, score_state: KalshiLightGBMScoreState) -> str | None:
        if score_state.quote_age_seconds is None:
            return "missing_quote"
        if score_state.quote_age_seconds > self.config.quote_max_age_seconds:
            return "stale_quote"
        if (
            score_state.yes_bid_cents is not None
            and score_state.yes_ask_cents is not None
            and score_state.yes_bid_cents >= score_state.yes_ask_cents
            and (score_state.no_bid_cents is None or score_state.no_ask_cents is None)
        ):
            return "crossed_quote"
        if (
            score_state.no_bid_cents is not None
            and score_state.no_ask_cents is not None
            and score_state.no_bid_cents >= score_state.no_ask_cents
            and (score_state.yes_bid_cents is None or score_state.yes_ask_cents is None)
        ):
            return "crossed_quote"
        usable_yes_quote = self._side_has_usable_quote(score_state, side="YES")
        usable_no_quote = self._side_has_usable_quote(score_state, side="NO")
        if not usable_yes_quote and not usable_no_quote:
            if (
                score_state.yes_bid_cents is not None
                and score_state.yes_ask_cents is not None
                and score_state.yes_bid_cents >= score_state.yes_ask_cents
            ):
                return "crossed_quote"
            if (
                score_state.no_bid_cents is not None
                and score_state.no_ask_cents is not None
                and score_state.no_bid_cents >= score_state.no_ask_cents
            ):
                return "crossed_quote"
            return "missing_quote"
        tolerance_cents = self.config.quote_consistency_tolerance_cents
        if (
            score_state.yes_bid_cents is not None
            and score_state.no_ask_cents is not None
            and abs((score_state.yes_bid_cents + score_state.no_ask_cents) - 100) > tolerance_cents
        ):
            return "inconsistent_quote"
        if (
            score_state.yes_ask_cents is not None
            and score_state.no_bid_cents is not None
            and abs((score_state.yes_ask_cents + score_state.no_bid_cents) - 100) > tolerance_cents
        ):
            return "inconsistent_quote"
        return None

    def _side_has_usable_quote(self, score_state: KalshiLightGBMScoreState, *, side: str) -> bool:
        normalized_side = side.upper()
        if normalized_side == "YES":
            if score_state.buy_yes_price_cents is None:
                return False
            if (
                score_state.yes_bid_cents is not None
                and score_state.yes_ask_cents is not None
                and score_state.yes_bid_cents >= score_state.yes_ask_cents
            ):
                return False
            return True
        if score_state.buy_no_price_cents is None:
            return False
        if (
            score_state.no_bid_cents is not None
            and score_state.no_ask_cents is not None
            and score_state.no_bid_cents >= score_state.no_ask_cents
        ):
            return False
        return True

    def _evaluate_side_candidate(
        self,
        score_state: KalshiLightGBMScoreState,
        *,
        side: str,
        entry_price_cents: int,
    ) -> _SideEvaluation | None:
        if not self._side_has_usable_quote(score_state, side=side):
            return None
        if entry_price_cents < self.config.price_band_min_cents or entry_price_cents > self.config.price_band_max_cents:
            return None
        contracts = self._resolve_contracts(
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            entry_price_cents=entry_price_cents,
        )
        if contracts <= 0:
            return None
        max_acceptable_entry_price_cents = find_max_acceptable_entry_price_cents(
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            config=self.config,
            contracts=contracts,
        )
        post_cost_edge, entry_cost_dollars, fees_dollars, cash_required_dollars = calculate_cost_metrics(
            side=side,
            predicted_yes_probability=score_state.predicted_yes_probability,
            displayed_entry_price_cents=entry_price_cents,
            contracts=contracts,
            slippage=self.config.slippage,
        )
        if (
            max_acceptable_entry_price_cents is None
            or entry_price_cents > max_acceptable_entry_price_cents
            or post_cost_edge + 1e-12 < self.config.edge_threshold
        ):
            return None
        return _SideEvaluation(
            side=side,
            entry_price_cents=entry_price_cents,
            max_acceptable_entry_price_cents=max_acceptable_entry_price_cents,
            post_cost_edge=post_cost_edge,
            contracts=contracts,
            entry_cost_dollars=entry_cost_dollars,
            fees_dollars=fees_dollars,
            cash_required_dollars=cash_required_dollars,
        )

    def _select_side_evaluation(
        self,
        *,
        yes_evaluation: _SideEvaluation | None,
        no_evaluation: _SideEvaluation | None,
        invert: bool,
    ) -> _SideEvaluation | None:
        evaluations = [evaluation for evaluation in (yes_evaluation, no_evaluation) if evaluation is not None]
        if not evaluations:
            return None
        evaluations.sort(key=lambda item: (item.post_cost_edge, -item.entry_price_cents), reverse=True)
        chosen = evaluations[0]
        if not invert:
            return chosen
        inverted_side = "NO" if chosen.side == "YES" else "YES"
        return no_evaluation if inverted_side == "NO" else yes_evaluation

    def _rejection_reason(
        self,
        score_state: KalshiLightGBMScoreState,
        *,
        raw_model_edge: float,
        yes_post_cost_edge: float | None,
        no_post_cost_edge: float | None,
        yes_evaluation: _SideEvaluation | None,
        no_evaluation: _SideEvaluation | None,
    ) -> str:
        buy_yes_price_cents = score_state.buy_yes_price_cents
        buy_no_price_cents = score_state.buy_no_price_cents
        if (
            buy_yes_price_cents is not None
            and buy_no_price_cents is not None
            and (
                (buy_yes_price_cents < self.config.price_band_min_cents or buy_yes_price_cents > self.config.price_band_max_cents)
                and (buy_no_price_cents < self.config.price_band_min_cents or buy_no_price_cents > self.config.price_band_max_cents)
            )
        ):
            return "outside_price_band"

        available_cash, _deployed_capital, equity, _reserved_cash = self._portfolio_metrics()
        reserve_cash = (self.config.reserve_cash_pct / 100.0) * equity
        candidate_cash_requirements: list[float] = []
        for side, entry_price_cents in (("YES", buy_yes_price_cents), ("NO", buy_no_price_cents)):
            if entry_price_cents is None:
                continue
            if entry_price_cents < self.config.price_band_min_cents or entry_price_cents > self.config.price_band_max_cents:
                continue
            _edge_1, _entry_1, _fees_1, cash_required_1 = calculate_cost_metrics(
                side=side,
                predicted_yes_probability=score_state.predicted_yes_probability,
                displayed_entry_price_cents=entry_price_cents,
                contracts=1,
                slippage=self.config.slippage,
            )
            candidate_cash_requirements.append(cash_required_1)
        if available_cash <= 0:
            return "insufficient_cash"
        if candidate_cash_requirements and available_cash + 1e-12 < min(candidate_cash_requirements):
            return "insufficient_cash"
        if candidate_cash_requirements and available_cash - min(candidate_cash_requirements) < reserve_cash - 1e-12:
            return "reserve_violation"

        if raw_model_edge > 0 and yes_evaluation is None and yes_post_cost_edge is not None:
            return "spread_destroyed_edge"
        if raw_model_edge < 0 and no_evaluation is None and no_post_cost_edge is not None:
            return "spread_destroyed_edge"
        if yes_post_cost_edge is not None or no_post_cost_edge is not None:
            return "insufficient_post_cost_edge"
        return "missing_quote"

    def _default_contract_count(self) -> int:
        return self.config.contracts_per_order

    def _resolve_contracts(
        self,
        *,
        side: str,
        predicted_yes_probability: float,
        entry_price_cents: int,
    ) -> int:
        available_cash, deployed_capital, equity, _reserved_cash = self._portfolio_metrics()
        reserve_cash = (self.config.reserve_cash_pct / 100.0) * equity
        deployable_cash = max(0.0, available_cash - reserve_cash)

        if self.config.kelly_fraction_multiplier is not None:
            kelly = calculate_kelly_sizing_metrics(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=entry_price_cents,
                slippage=self.config.slippage,
                fraction_multiplier=self.config.kelly_fraction_multiplier,
                fraction_cap=(
                    None
                    if self.config.kelly_fraction_cap_pct is None
                    else (self.config.kelly_fraction_cap_pct / 100.0)
                ),
            )
            target_fraction = kelly.capped_fraction_of_equity
            cash_budget = min(target_fraction * equity, deployable_cash)
            if cash_budget <= 0 or kelly.per_contract_cash_required_dollars > cash_budget + 1e-12:
                return 0
            return self._max_contracts_for_cash_budget(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                entry_price_cents=entry_price_cents,
                cash_budget=cash_budget,
            )

        if self.config.capital_pct_per_order is not None:
            cash_budget = min(equity * (self.config.capital_pct_per_order / 100.0), deployable_cash)
            if cash_budget <= 0:
                return 0
            _edge_1, _entry_1, _fees_1, cash_required_1 = calculate_cost_metrics(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=entry_price_cents,
                contracts=1,
                slippage=self.config.slippage,
            )
            if cash_required_1 > cash_budget + 1e-12:
                return 0
            return self._max_contracts_for_cash_budget(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                entry_price_cents=entry_price_cents,
                cash_budget=cash_budget,
            )

        return self.config.contracts_per_order

    def _max_contracts_for_cash_budget(
        self,
        *,
        side: str,
        predicted_yes_probability: float,
        entry_price_cents: int,
        cash_budget: float,
    ) -> int:
        upper_bound = max(1, int(cash_budget / max(entry_price_cents / 100.0, 1e-9)) + 1)
        low = 1
        high = upper_bound
        best = 1
        while low <= high:
            mid = (low + high) // 2
            _edge_mid, _entry_mid, _fees_mid, cash_required_mid = calculate_cost_metrics(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=entry_price_cents,
                contracts=mid,
                slippage=self.config.slippage,
            )
            if cash_required_mid <= cash_budget + 1e-12:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _portfolio_metrics(self) -> tuple[float, float, float, float]:
        pending_cash_required = sum(item.cash_required_dollars for item in self._pending_reservations.values())
        pending_entry_cost = sum(item.entry_cost_dollars for item in self._pending_reservations.values())
        local_open_cash_required = sum(item.cash_required_dollars for item in self._local_open_positions.values())
        local_open_entry_cost = sum(item.entry_cost_dollars for item in self._local_open_positions.values())

        available_cash = (
            self._baseline_snapshot.available_cash_dollars
            - pending_cash_required
            - local_open_cash_required
        )
        deployed_capital = (
            self._baseline_snapshot.deployed_capital_dollars
            + pending_entry_cost
            + local_open_entry_cost
        )
        equity = available_cash + deployed_capital
        return available_cash, deployed_capital, equity, pending_cash_required

    def _combined_open_positions(self) -> list[KalshiPortfolioPosition]:
        combined = list(self._baseline_snapshot.open_positions)
        combined.extend(self._local_open_positions.values())
        return combined

    def _ticker_is_locked(self, ticker: str) -> bool:
        if any(position.ticker == ticker for position in self._baseline_snapshot.open_positions):
            return True
        if any(position.ticker == ticker for position in self._local_open_positions.values()):
            return True
        if any(reservation.ticker == ticker for reservation in self._pending_reservations.values()):
            return True
        return False

    def _cooldown_is_active(self) -> bool:
        if self.config.trade_cooldown_seconds <= 0:
            return False
        if self._last_trade_opened_at is None:
            return False
        return (utc_now() - self._last_trade_opened_at).total_seconds() < self.config.trade_cooldown_seconds

    def _reserve_trade_intent(
        self,
        candidate: _DecisionCandidate,
        *,
        stacking_signature: StackingSignature | None = None,
    ) -> KalshiTradeIntent:
        decision_id = str(uuid.uuid4())
        generated_at = utc_now()
        self._last_trade_opened_at = generated_at
        resolved_signature = (
            self._build_stacking_signature_from_candidate(candidate)
            if stacking_signature is None
            else stacking_signature
        )
        reservation = _PendingReservation(
            decision_id=decision_id,
            ticker=candidate.ticker,
            side=candidate.side or "YES",
            contracts=candidate.contracts,
            entry_cost_dollars=candidate.entry_cost_dollars,
            fees_dollars=candidate.fees_dollars,
            cash_required_dollars=candidate.cash_required_dollars,
            created_at=generated_at,
            expires_at=generated_at + timedelta(seconds=self.config.reservation_ttl_seconds),
            is_acknowledged=False,
            stacking_signature=resolved_signature,
        )
        self._pending_reservations[decision_id] = reservation
        self._mark_stacking_signature_active(reservation)
        return KalshiTradeIntent(
            decision_id=decision_id,
            ticker=candidate.ticker,
            side=candidate.side or "YES",
            contracts=candidate.contracts,
            reference_price_cents=candidate.reference_price_cents or 0,
            max_acceptable_entry_price_cents=candidate.max_acceptable_entry_price_cents or 0,
            predicted_yes_probability=candidate.predicted_yes_probability,
            predicted_no_probability=candidate.predicted_no_probability,
            feature_basis_market_prob=candidate.feature_basis_market_prob,
            raw_model_edge=candidate.raw_model_edge or 0.0,
            post_cost_edge=candidate.post_cost_edge or 0.0,
            yes_post_cost_edge=candidate.yes_post_cost_edge,
            no_post_cost_edge=candidate.no_post_cost_edge,
            last_yes_price_cents=candidate.last_yes_price_cents,
            yes_bid_cents=candidate.yes_bid_cents,
            yes_ask_cents=candidate.yes_ask_cents,
            buy_yes_price_cents=candidate.buy_yes_price_cents,
            buy_no_price_cents=candidate.buy_no_price_cents,
            quote_mid_prob=candidate.quote_mid_prob,
            quote_spread_cents=candidate.quote_spread_cents,
            quote_age_seconds=candidate.quote_age_seconds,
            estimated_entry_cost_dollars=candidate.entry_cost_dollars,
            estimated_fees_dollars=candidate.fees_dollars,
            estimated_cash_required_dollars=candidate.cash_required_dollars,
            generated_at=generated_at,
            stacking_signature=resolved_signature,
        )

    def _decision_signature(self, state: KalshiSignalDecisionState) -> tuple[Any, ...]:
        return (
            state.approved,
            state.side,
            round(state.predicted_yes_probability, 10),
            round(state.predicted_no_probability, 10),
            round(state.feature_basis_market_prob, 10),
            None if state.raw_model_edge is None else round(state.raw_model_edge, 10),
            None if state.post_cost_edge is None else round(state.post_cost_edge, 10),
            None if state.yes_post_cost_edge is None else round(state.yes_post_cost_edge, 10),
            None if state.no_post_cost_edge is None else round(state.no_post_cost_edge, 10),
            round(state.tau_minutes, 10),
            state.reference_price_cents,
            state.max_acceptable_entry_price_cents,
            state.last_yes_price_cents,
            state.yes_bid_cents,
            state.yes_ask_cents,
            state.buy_yes_price_cents,
            state.buy_no_price_cents,
            None if state.quote_mid_prob is None else round(state.quote_mid_prob, 10),
            state.quote_spread_cents,
            None if state.quote_age_seconds is None else round(state.quote_age_seconds, 10),
            state.regime_label,
            state.bearish_vote_count,
            state.bullish_vote_count,
            state.regime_price_momentum_bearish,
            state.regime_signed_flow_bearish,
            state.regime_yes_share_bearish,
            state.regime_price_momentum_bullish,
            state.regime_signed_flow_bullish,
            state.regime_yes_share_bullish,
            state.tau_bucket,
            state.price_bucket,
            state.chosen_side_probability_bucket,
            state.chosen_side_edge_bucket,
            state.bucket_policy_dimension,
            state.bucket_policy_bucket,
            state.bucket_policy_side,
            state.block_reason,
            None if state.trade_intent is None else state.trade_intent.contracts,
            None if state.trade_intent is None else round(state.trade_intent.estimated_entry_cost_dollars, 10),
            None if state.trade_intent is None else round(state.trade_intent.estimated_fees_dollars, 10),
            None if state.trade_intent is None else round(state.trade_intent.estimated_cash_required_dollars, 10),
        )

    def _candidate_signature(self, candidate: _DecisionCandidate) -> tuple[Any, ...]:
        return (
            candidate.approved,
            candidate.side,
            round(candidate.predicted_yes_probability, 10),
            round(candidate.predicted_no_probability, 10),
            round(candidate.feature_basis_market_prob, 10),
            None if candidate.raw_model_edge is None else round(candidate.raw_model_edge, 10),
            None if candidate.post_cost_edge is None else round(candidate.post_cost_edge, 10),
            None if candidate.yes_post_cost_edge is None else round(candidate.yes_post_cost_edge, 10),
            None if candidate.no_post_cost_edge is None else round(candidate.no_post_cost_edge, 10),
            round(candidate.tau_minutes, 10),
            candidate.reference_price_cents,
            candidate.max_acceptable_entry_price_cents,
            candidate.last_yes_price_cents,
            candidate.yes_bid_cents,
            candidate.yes_ask_cents,
            candidate.buy_yes_price_cents,
            candidate.buy_no_price_cents,
            None if candidate.quote_mid_prob is None else round(candidate.quote_mid_prob, 10),
            candidate.quote_spread_cents,
            None if candidate.quote_age_seconds is None else round(candidate.quote_age_seconds, 10),
            candidate.regime_label,
            candidate.bearish_vote_count,
            candidate.bullish_vote_count,
            candidate.regime_price_momentum_bearish,
            candidate.regime_signed_flow_bearish,
            candidate.regime_yes_share_bearish,
            candidate.regime_price_momentum_bullish,
            candidate.regime_signed_flow_bullish,
            candidate.regime_yes_share_bullish,
            candidate.tau_bucket,
            candidate.price_bucket,
            candidate.chosen_side_probability_bucket,
            candidate.chosen_side_edge_bucket,
            candidate.bucket_policy_dimension,
            candidate.bucket_policy_bucket,
            candidate.bucket_policy_side,
            candidate.block_reason,
            candidate.contracts if candidate.approved else None,
            round(candidate.entry_cost_dollars, 10) if candidate.approved else None,
            round(candidate.fees_dollars, 10) if candidate.approved else None,
            round(candidate.cash_required_dollars, 10) if candidate.approved else None,
        )

    def _build_stacking_signature(
        self,
        *,
        side: str | None,
        chosen_buckets: Any | None,
    ) -> StackingSignature | None:
        if side is None or chosen_buckets is None:
            return None
        return (
            side,
            chosen_buckets.tau_bucket,
            chosen_buckets.price_bucket,
            chosen_buckets.chosen_side_probability_bucket,
            chosen_buckets.chosen_side_edge_bucket,
        )

    def _build_stacking_signature_from_candidate(self, candidate: _DecisionCandidate) -> StackingSignature | None:
        if candidate.side is None:
            return None
        return (
            candidate.side,
            candidate.tau_bucket,
            candidate.price_bucket,
            candidate.chosen_side_probability_bucket,
            candidate.chosen_side_edge_bucket,
        )

    def _stacking_signature_is_seen(self, ticker: str, signature: StackingSignature | None) -> bool:
        if signature is None:
            return False
        if signature in self._completed_stacking_signatures.get(ticker, set()):
            return True
        return self._active_stacking_signatures.get(ticker, {}).get(signature, 0) > 0

    def _mark_stacking_signature_active(self, reservation: _PendingReservation) -> None:
        signature = reservation.stacking_signature
        if signature is None:
            return
        ticker_counts = self._active_stacking_signatures[reservation.ticker]
        ticker_counts[signature] = ticker_counts.get(signature, 0) + 1

    def _release_stacking_signature_active(self, reservation: _PendingReservation) -> None:
        signature = reservation.stacking_signature
        if signature is None:
            return
        ticker_counts = self._active_stacking_signatures.get(reservation.ticker)
        if not ticker_counts:
            return
        current = ticker_counts.get(signature, 0)
        if current <= 1:
            ticker_counts.pop(signature, None)
        else:
            ticker_counts[signature] = current - 1
        if not ticker_counts:
            self._active_stacking_signatures.pop(reservation.ticker, None)

    def _mark_stacking_signature_completed(self, reservation: _PendingReservation) -> None:
        signature = reservation.stacking_signature
        if signature is None:
            return
        self._completed_stacking_signatures[reservation.ticker].add(signature)

    async def _publish_update(self, update: KalshiSignalDecisionUpdate) -> None:
        await self._logger.write(
            "signal_decision",
            {
                "ticker": update.ticker,
                "approved": update.approved,
                "side": update.side,
                "predicted_yes_probability": update.predicted_yes_probability,
                "predicted_no_probability": update.predicted_no_probability,
                "feature_basis_market_prob": update.feature_basis_market_prob,
                "raw_model_edge": update.raw_model_edge,
                "post_cost_edge": update.post_cost_edge,
                "yes_post_cost_edge": update.yes_post_cost_edge,
                "no_post_cost_edge": update.no_post_cost_edge,
                "tau_minutes": update.tau_minutes,
                "reference_price_cents": update.reference_price_cents,
                "max_acceptable_entry_price_cents": update.max_acceptable_entry_price_cents,
                "last_yes_price_cents": update.last_yes_price_cents,
                "yes_bid_cents": update.yes_bid_cents,
                "yes_ask_cents": update.yes_ask_cents,
                "buy_yes_price_cents": update.buy_yes_price_cents,
                "buy_no_price_cents": update.buy_no_price_cents,
                "quote_mid_prob": update.quote_mid_prob,
                "quote_spread_cents": update.quote_spread_cents,
                "quote_age_seconds": update.quote_age_seconds,
                "tau_bucket": update.tau_bucket,
                "price_bucket": update.price_bucket,
                "chosen_side_probability_bucket": update.chosen_side_probability_bucket,
                "chosen_side_edge_bucket": update.chosen_side_edge_bucket,
                "regime_label": update.regime_label,
                "bearish_vote_count": update.bearish_vote_count,
                "bullish_vote_count": update.bullish_vote_count,
                "regime_price_momentum_bearish": update.regime_price_momentum_bearish,
                "regime_signed_flow_bearish": update.regime_signed_flow_bearish,
                "regime_yes_share_bearish": update.regime_yes_share_bearish,
                "regime_price_momentum_bullish": update.regime_price_momentum_bullish,
                "regime_signed_flow_bullish": update.regime_signed_flow_bullish,
                "regime_yes_share_bullish": update.regime_yes_share_bullish,
                **bucket_policy_fields(
                    KalshiBucketPolicyEvaluation(
                        is_blocked=True,
                        blocked_dimension=update.bucket_policy_dimension,
                        blocked_bucket=update.bucket_policy_bucket,
                        blocked_side=update.bucket_policy_side,
                    )
                    if update.bucket_policy_dimension is not None
                    else None
                ),
                "block_reason": update.block_reason,
                "decision_id": None if update.trade_intent is None else update.trade_intent.decision_id,
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
            },
            event_time=update.event_time,
        )
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)
        if update.approved and update.trade_intent is not None:
            for callback in list(self._intent_callbacks):
                result = callback(update.trade_intent)
                if inspect.isawaitable(result):
                    await result
            for queue in self._intent_queues:
                await queue.put(update.trade_intent)
