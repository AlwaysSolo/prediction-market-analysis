from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

import numpy as np
from scipy.stats import norm

from src.live.kalshi.spot_features import (
    SPOT_DIAGNOSTIC_FIELDS,
    SPOT_MODEL_FEATURE_SET,
    SPOT_V1_FEATURE_ORDER,
    SPOT_V1_FEATURE_SCHEMA,
)
from src.live.kalshi.types import KalshiTickerState, KalshiTickerUpdate

ROLLING_WINDOW_SECONDS = (30, 120, 300)
DEFAULT_FEATURE_SCHEMA = "default"
LINEAR_V1_FEATURE_SCHEMA = "linear_v1"
FEATURE_SCHEMA_CHOICES = (DEFAULT_FEATURE_SCHEMA, LINEAR_V1_FEATURE_SCHEMA, SPOT_V1_FEATURE_SCHEMA)
FEATURE_ORDER = [
    "z_implied",
    "tau_minutes",
    "price_momentum",
    "abs_price_momentum",
    "price_direction",
    "distance_from_mid",
    "last_trade_count",
    "last_trade_side_sign",
    "last_trade_signed_count",
    "time_since_last_trade_seconds",
    "minutes_since_market_open",
    "trade_count_30s",
    "contracts_sum_30s",
    "signed_contracts_sum_30s",
    "yes_taker_share_30s",
    "price_return_30s",
    "price_volatility_30s",
    "trade_count_120s",
    "contracts_sum_120s",
    "signed_contracts_sum_120s",
    "yes_taker_share_120s",
    "price_return_120s",
    "price_volatility_120s",
    "trade_count_300s",
    "contracts_sum_300s",
    "signed_contracts_sum_300s",
    "yes_taker_share_300s",
    "price_return_300s",
    "price_volatility_300s",
]
HOURLY_CONTEXT_FEATURE_ORDER = (
    "kxbtcd_atm_z_implied",
    "kxbtcd_atm_price_momentum",
    "kxbtcd_atm_abs_price_momentum",
    "kxbtcd_atm_distance_from_mid",
    "kxbtcd_atm_trade_count_300s",
    "kxbtcd_atm_signed_contracts_sum_300s",
    "kxbtcd_atm_price_return_300s",
    "kxbtcd_atm_price_volatility_300s",
    "kxbtcd_atm_yes_taker_share_300s",
    "k15_minus_k1h_atm_z",
    "k15_minus_k1h_atm_price_return_300s",
    "k15_k1h_atm_direction_agreement",
)
LINEAR_V1_DROPPED_FEATURES = (
    "price_direction",
    "last_trade_count",
    "last_trade_side_sign",
    "trade_count_120s",
    "contracts_sum_120s",
    "signed_contracts_sum_120s",
    "yes_taker_share_120s",
    "price_return_120s",
    "price_volatility_120s",
)
LINEAR_V1_DERIVED_FEATURE_ORDER = (
    "k15_z_tau_decay",
    "k15_return_300s_tau_decay",
    "k15_k1h_z_product",
    "k15_k1h_return_product_300s",
)
LINEAR_V1_DERIVED_FORMULAS = {
    "tau_decay": "max(tau_minutes, 0.0) + 1.0",
    "k15_z_tau_decay": "z_implied / tau_decay",
    "k15_return_300s_tau_decay": "price_return_300s / tau_decay",
    "k15_k1h_z_product": "z_implied * kxbtcd_atm_z_implied",
    "k15_k1h_return_product_300s": "price_return_300s * kxbtcd_atm_price_return_300s",
}
LINEAR_V1_FEATURE_ORDER = tuple(
    [
        feature_name
        for feature_name in [*FEATURE_ORDER, *HOURLY_CONTEXT_FEATURE_ORDER]
        if feature_name not in LINEAR_V1_DROPPED_FEATURES
    ]
    + list(LINEAR_V1_DERIVED_FEATURE_ORDER)
)
ALL_FEATURE_ORDER = tuple(
    dict.fromkeys([*FEATURE_ORDER, *HOURLY_CONTEXT_FEATURE_ORDER, *LINEAR_V1_DERIVED_FEATURE_ORDER, *SPOT_V1_FEATURE_ORDER])
)
HOURLY_CONTEXT_REQUIRED_FEATURE_NAMES = frozenset(
    [*HOURLY_CONTEXT_FEATURE_ORDER, "k15_k1h_z_product", "k15_k1h_return_product_300s"]
)
SPOT_V1_ALL_REQUIRED_FEATURE_NAMES = frozenset([*FEATURE_ORDER, *SPOT_V1_FEATURE_ORDER])


def feature_order_hash(feature_order: tuple[str, ...] | list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(str(name) for name in feature_order).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class KalshiFeatureEngineConfig:
    min_market_prob: float = 0.01
    max_market_prob: float = 0.99
    tau_max_minutes: float = 15.0
    z_posinf: float = 3.0
    z_neginf: float = -3.0
    rolling_window_seconds: tuple[int, ...] = ROLLING_WINDOW_SECONDS
    publish_series_tickers: tuple[str, ...] = ()
    hourly_context_target_series_ticker: str | None = None
    hourly_context_series_ticker: str | None = None
    hourly_context_max_staleness_seconds: float = 300.0
    hourly_context_min_recent_trade_count_300s: float = 1.0
    feature_schema: str = DEFAULT_FEATURE_SCHEMA
    external_spot_max_age_ms: int = 500
    external_spot_required_for_schema: bool = False
    external_spot_divergence_sigma_threshold: float = 4.0
    external_spot_divergence_lookback_seconds: float = 300.0
    external_spot_divergence_detrend_halflife_seconds: float = 3600.0

    def normalized_publish_series_tickers(self) -> tuple[str, ...]:
        return tuple(sorted({ticker for ticker in self.publish_series_tickers if ticker}))


def clean_probability(
    probability: float,
    config: KalshiFeatureEngineConfig | None = None,
) -> float:
    config = config or KalshiFeatureEngineConfig()
    return float(np.clip(probability, config.min_market_prob, config.max_market_prob))


def clean_z_implied(
    market_prob: float,
    config: KalshiFeatureEngineConfig | None = None,
) -> float:
    config = config or KalshiFeatureEngineConfig()
    z_implied = float(norm.ppf(market_prob))
    return float(np.nan_to_num(z_implied, posinf=config.z_posinf, neginf=config.z_neginf))


def linear_v1_tau_decay(tau_minutes: float | None) -> float | None:
    if tau_minutes is None:
        return None
    return max(float(tau_minutes), 0.0) + 1.0


def build_linear_v1_derived_feature_lookup(
    feature_lookup: dict[str, float | None],
) -> dict[str, float | None]:
    tau_decay = linear_v1_tau_decay(feature_lookup.get("tau_minutes"))
    z_implied = feature_lookup.get("z_implied")
    price_return_300s = feature_lookup.get("price_return_300s")
    hourly_z_implied = feature_lookup.get("kxbtcd_atm_z_implied")
    hourly_price_return_300s = feature_lookup.get("kxbtcd_atm_price_return_300s")
    return {
        "k15_z_tau_decay": (
            None if tau_decay is None or z_implied is None else float(z_implied) / float(tau_decay)
        ),
        "k15_return_300s_tau_decay": (
            None
            if tau_decay is None or price_return_300s is None
            else float(price_return_300s) / float(tau_decay)
        ),
        "k15_k1h_z_product": (
            None
            if z_implied is None or hourly_z_implied is None
            else float(z_implied) * float(hourly_z_implied)
        ),
        "k15_k1h_return_product_300s": (
            None
            if price_return_300s is None or hourly_price_return_300s is None
            else float(price_return_300s) * float(hourly_price_return_300s)
        ),
    }


def build_feature_components(
    market_prob: float,
    previous_market_prob: float | None = None,
    config: KalshiFeatureEngineConfig | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    config = config or KalshiFeatureEngineConfig()
    market_prob = clean_probability(market_prob, config)
    previous_market_prob = market_prob if previous_market_prob is None else clean_probability(previous_market_prob, config)

    price_momentum = float(market_prob - previous_market_prob)
    z_implied = clean_z_implied(market_prob, config)
    abs_price_momentum = abs(price_momentum)
    price_direction = float(np.sign(price_momentum))
    distance_from_mid = abs(market_prob - 0.5)

    return (
        market_prob,
        previous_market_prob,
        z_implied,
        price_momentum,
        abs_price_momentum,
        price_direction,
        distance_from_mid,
    )


def build_feature_row(
    market_prob: float,
    tau_minutes: float,
    previous_market_prob: float | None = None,
    *,
    last_trade_count: float = 0.0,
    last_trade_side_sign: float = 0.0,
    last_trade_signed_count: float = 0.0,
    time_since_last_trade_seconds: float = 0.0,
    minutes_since_market_open: float = 0.0,
    trade_count_30s: float = 0.0,
    contracts_sum_30s: float = 0.0,
    signed_contracts_sum_30s: float = 0.0,
    yes_taker_share_30s: float = 0.0,
    price_return_30s: float = 0.0,
    price_volatility_30s: float = 0.0,
    trade_count_120s: float = 0.0,
    contracts_sum_120s: float = 0.0,
    signed_contracts_sum_120s: float = 0.0,
    yes_taker_share_120s: float = 0.0,
    price_return_120s: float = 0.0,
    price_volatility_120s: float = 0.0,
    trade_count_300s: float = 0.0,
    contracts_sum_300s: float = 0.0,
    signed_contracts_sum_300s: float = 0.0,
    yes_taker_share_300s: float = 0.0,
    price_return_300s: float = 0.0,
    price_volatility_300s: float = 0.0,
    kxbtcd_atm_z_implied: float | None = None,
    kxbtcd_atm_price_momentum: float | None = None,
    kxbtcd_atm_abs_price_momentum: float | None = None,
    kxbtcd_atm_distance_from_mid: float | None = None,
    kxbtcd_atm_trade_count_300s: float | None = None,
    kxbtcd_atm_signed_contracts_sum_300s: float | None = None,
    kxbtcd_atm_price_return_300s: float | None = None,
    kxbtcd_atm_price_volatility_300s: float | None = None,
    kxbtcd_atm_yes_taker_share_300s: float | None = None,
    k15_minus_k1h_atm_z: float | None = None,
    k15_minus_k1h_atm_price_return_300s: float | None = None,
    k15_k1h_atm_direction_agreement: float | None = None,
    feature_names: tuple[str, ...] | None = None,
    config: KalshiFeatureEngineConfig | None = None,
) -> np.ndarray:
    (
        _market_prob,
        _previous_market_prob,
        z_implied,
        price_momentum,
        abs_price_momentum,
        price_direction,
        distance_from_mid,
    ) = build_feature_components(
        market_prob=market_prob,
        previous_market_prob=previous_market_prob,
        config=config,
    )

    lookup: dict[str, float | None] = {
        "z_implied": z_implied,
        "tau_minutes": float(tau_minutes),
        "price_momentum": price_momentum,
        "abs_price_momentum": abs_price_momentum,
        "price_direction": price_direction,
        "distance_from_mid": distance_from_mid,
        "last_trade_count": float(last_trade_count),
        "last_trade_side_sign": float(last_trade_side_sign),
        "last_trade_signed_count": float(last_trade_signed_count),
        "time_since_last_trade_seconds": float(time_since_last_trade_seconds),
        "minutes_since_market_open": float(minutes_since_market_open),
        "trade_count_30s": float(trade_count_30s),
        "contracts_sum_30s": float(contracts_sum_30s),
        "signed_contracts_sum_30s": float(signed_contracts_sum_30s),
        "yes_taker_share_30s": float(yes_taker_share_30s),
        "price_return_30s": float(price_return_30s),
        "price_volatility_30s": float(price_volatility_30s),
        "trade_count_120s": float(trade_count_120s),
        "contracts_sum_120s": float(contracts_sum_120s),
        "signed_contracts_sum_120s": float(signed_contracts_sum_120s),
        "yes_taker_share_120s": float(yes_taker_share_120s),
        "price_return_120s": float(price_return_120s),
        "price_volatility_120s": float(price_volatility_120s),
        "trade_count_300s": float(trade_count_300s),
        "contracts_sum_300s": float(contracts_sum_300s),
        "signed_contracts_sum_300s": float(signed_contracts_sum_300s),
        "yes_taker_share_300s": float(yes_taker_share_300s),
        "price_return_300s": float(price_return_300s),
        "price_volatility_300s": float(price_volatility_300s),
        "kxbtcd_atm_z_implied": kxbtcd_atm_z_implied,
        "kxbtcd_atm_price_momentum": kxbtcd_atm_price_momentum,
        "kxbtcd_atm_abs_price_momentum": kxbtcd_atm_abs_price_momentum,
        "kxbtcd_atm_distance_from_mid": kxbtcd_atm_distance_from_mid,
        "kxbtcd_atm_trade_count_300s": kxbtcd_atm_trade_count_300s,
        "kxbtcd_atm_signed_contracts_sum_300s": kxbtcd_atm_signed_contracts_sum_300s,
        "kxbtcd_atm_price_return_300s": kxbtcd_atm_price_return_300s,
        "kxbtcd_atm_price_volatility_300s": kxbtcd_atm_price_volatility_300s,
        "kxbtcd_atm_yes_taker_share_300s": kxbtcd_atm_yes_taker_share_300s,
        "k15_minus_k1h_atm_z": k15_minus_k1h_atm_z,
        "k15_minus_k1h_atm_price_return_300s": k15_minus_k1h_atm_price_return_300s,
        "k15_k1h_atm_direction_agreement": k15_k1h_atm_direction_agreement,
    }
    lookup.update(build_linear_v1_derived_feature_lookup(lookup))
    selected_feature_names = tuple(feature_names) if feature_names is not None else tuple(FEATURE_ORDER)
    values = [float(lookup.get(feature_name, 0.0) or 0.0) for feature_name in selected_feature_names]
    return np.array([values], dtype=np.float32)


@dataclass(frozen=True)
class KalshiFeatureState:
    ticker: str
    event_time: datetime
    last_yes_price_cents: int | None
    previous_yes_price_cents: int | None
    market_prob: float | None
    previous_market_prob: float | None
    close_time: datetime | None
    open_time: datetime | None
    is_open: bool
    trade_id: str | None
    count: int | None
    taker_side: str | None
    last_trade_time: datetime | None
    last_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
    no_bid_cents: int | None
    no_ask_cents: int | None
    volume: int | None
    open_interest: int | None
    dollar_volume: int | None
    dollar_open_interest: int | None
    ticker_update_time: datetime | None
    trade_yes_prob: float | None
    quote_mid_prob: float | None
    quote_spread_cents: int | None
    buy_yes_price_cents: int | None
    buy_no_price_cents: int | None
    quote_age_seconds: float | None
    last_to_mid_gap: float | None
    z_implied: float | None
    tau_minutes: float | None
    price_momentum: float | None
    abs_price_momentum: float | None
    price_direction: float | None
    distance_from_mid: float | None
    last_trade_count: float | None
    last_trade_side_sign: float | None
    last_trade_signed_count: float | None
    time_since_last_trade_seconds: float | None
    minutes_since_market_open: float | None
    trade_count_30s: float | None
    contracts_sum_30s: float | None
    signed_contracts_sum_30s: float | None
    yes_taker_share_30s: float | None
    price_return_30s: float | None
    price_volatility_30s: float | None
    trade_count_120s: float | None
    contracts_sum_120s: float | None
    signed_contracts_sum_120s: float | None
    yes_taker_share_120s: float | None
    price_return_120s: float | None
    price_volatility_120s: float | None
    trade_count_300s: float | None
    contracts_sum_300s: float | None
    signed_contracts_sum_300s: float | None
    yes_taker_share_300s: float | None
    price_return_300s: float | None
    price_volatility_300s: float | None
    in_training_window: bool
    is_scoreable: bool
    kxbtcd_atm_ticker: str | None = None
    kxbtcd_atm_z_implied: float | None = None
    kxbtcd_atm_price_momentum: float | None = None
    kxbtcd_atm_abs_price_momentum: float | None = None
    kxbtcd_atm_distance_from_mid: float | None = None
    kxbtcd_atm_trade_count_300s: float | None = None
    kxbtcd_atm_signed_contracts_sum_300s: float | None = None
    kxbtcd_atm_price_return_300s: float | None = None
    kxbtcd_atm_price_volatility_300s: float | None = None
    kxbtcd_atm_yes_taker_share_300s: float | None = None
    k15_minus_k1h_atm_z: float | None = None
    k15_minus_k1h_atm_price_return_300s: float | None = None
    k15_k1h_atm_direction_agreement: float | None = None
    btc_spot_price: float | None = None
    btc_spot_twap_60s: float | None = None
    btc_spot_age_ms: float | None = None
    btc_spot_is_fresh: bool | None = None
    btc_spot_venues_fresh: float | None = None
    btc_spot_venue_divergence_bps: float | None = None
    btc_vol_effective_sample_size: float | None = None
    btc_spot_source: str | None = None
    btc_log_moneyness: float | None = None
    btc_log_moneyness_twap60: float | None = None
    btc_log_moneyness_per_minute: float | None = None
    btc_spot_return_30s: float | None = None
    btc_spot_return_120s: float | None = None
    btc_spot_return_300s: float | None = None
    btc_spot_return_900s: float | None = None
    btc_spot_vol_120s: float | None = None
    btc_spot_vol_300s: float | None = None
    btc_spot_vol_900s: float | None = None
    btc_spot_vol_1800s: float | None = None
    btc_spot_vol_ewma_hl300: float | None = None
    btc_bs_yes_prob_120s: float | None = None
    btc_bs_yes_prob_300s: float | None = None
    btc_bs_yes_prob_900s: float | None = None
    btc_bs_yes_prob_1800s: float | None = None
    btc_bs_yes_prob_ewma: float | None = None
    btc_kalshi_minus_bs_prob_120s: float | None = None
    btc_kalshi_minus_bs_prob_300s: float | None = None
    btc_kalshi_minus_bs_prob_900s: float | None = None
    btc_kalshi_minus_bs_prob_1800s: float | None = None
    btc_kalshi_minus_bs_prob_ewma: float | None = None
    btc_kalshi_minus_bs_logodds_120s: float | None = None
    btc_kalshi_minus_bs_logodds_300s: float | None = None
    btc_kalshi_minus_bs_logodds_900s: float | None = None
    btc_kalshi_minus_bs_logodds_1800s: float | None = None
    btc_kalshi_minus_bs_logodds_ewma: float | None = None
    btc_kalshi_minus_bs_prob_120s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_300s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_900s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_1800s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_ewma_detrended: float | None = None
    btc_kalshi_implied_vol: float | None = None
    received_at: datetime | None = field(default=None, compare=False)
    event_id: str = field(default="", compare=False)
    raw_event_id: str | None = field(default=None, compare=False)
    source: str = field(default="snapshot", compare=False)

    def feature_values(self) -> tuple[float, ...] | None:
        values = (
            self.z_implied,
            self.tau_minutes,
            self.price_momentum,
            self.abs_price_momentum,
            self.price_direction,
            self.distance_from_mid,
            self.last_trade_count,
            self.last_trade_side_sign,
            self.last_trade_signed_count,
            self.time_since_last_trade_seconds,
            self.minutes_since_market_open,
            self.trade_count_30s,
            self.contracts_sum_30s,
            self.signed_contracts_sum_30s,
            self.yes_taker_share_30s,
            self.price_return_30s,
            self.price_volatility_30s,
            self.trade_count_120s,
            self.contracts_sum_120s,
            self.signed_contracts_sum_120s,
            self.yes_taker_share_120s,
            self.price_return_120s,
            self.price_volatility_120s,
            self.trade_count_300s,
            self.contracts_sum_300s,
            self.signed_contracts_sum_300s,
            self.yes_taker_share_300s,
            self.price_return_300s,
            self.price_volatility_300s,
        )
        if any(value is None for value in values):
            return None
        return values  # type: ignore[return-value]

    def feature_lookup(self) -> dict[str, float | None]:
        lookup = {
            "z_implied": self.z_implied,
            "tau_minutes": self.tau_minutes,
            "price_momentum": self.price_momentum,
            "abs_price_momentum": self.abs_price_momentum,
            "price_direction": self.price_direction,
            "distance_from_mid": self.distance_from_mid,
            "last_trade_count": self.last_trade_count,
            "last_trade_side_sign": self.last_trade_side_sign,
            "last_trade_signed_count": self.last_trade_signed_count,
            "time_since_last_trade_seconds": self.time_since_last_trade_seconds,
            "minutes_since_market_open": self.minutes_since_market_open,
            "trade_count_30s": self.trade_count_30s,
            "contracts_sum_30s": self.contracts_sum_30s,
            "signed_contracts_sum_30s": self.signed_contracts_sum_30s,
            "yes_taker_share_30s": self.yes_taker_share_30s,
            "price_return_30s": self.price_return_30s,
            "price_volatility_30s": self.price_volatility_30s,
            "trade_count_120s": self.trade_count_120s,
            "contracts_sum_120s": self.contracts_sum_120s,
            "signed_contracts_sum_120s": self.signed_contracts_sum_120s,
            "yes_taker_share_120s": self.yes_taker_share_120s,
            "price_return_120s": self.price_return_120s,
            "price_volatility_120s": self.price_volatility_120s,
            "trade_count_300s": self.trade_count_300s,
            "contracts_sum_300s": self.contracts_sum_300s,
            "signed_contracts_sum_300s": self.signed_contracts_sum_300s,
            "yes_taker_share_300s": self.yes_taker_share_300s,
            "price_return_300s": self.price_return_300s,
            "price_volatility_300s": self.price_volatility_300s,
            "kxbtcd_atm_z_implied": self.kxbtcd_atm_z_implied,
            "kxbtcd_atm_price_momentum": self.kxbtcd_atm_price_momentum,
            "kxbtcd_atm_abs_price_momentum": self.kxbtcd_atm_abs_price_momentum,
            "kxbtcd_atm_distance_from_mid": self.kxbtcd_atm_distance_from_mid,
            "kxbtcd_atm_trade_count_300s": self.kxbtcd_atm_trade_count_300s,
            "kxbtcd_atm_signed_contracts_sum_300s": self.kxbtcd_atm_signed_contracts_sum_300s,
            "kxbtcd_atm_price_return_300s": self.kxbtcd_atm_price_return_300s,
            "kxbtcd_atm_price_volatility_300s": self.kxbtcd_atm_price_volatility_300s,
            "kxbtcd_atm_yes_taker_share_300s": self.kxbtcd_atm_yes_taker_share_300s,
            "k15_minus_k1h_atm_z": self.k15_minus_k1h_atm_z,
            "k15_minus_k1h_atm_price_return_300s": self.k15_minus_k1h_atm_price_return_300s,
            "k15_k1h_atm_direction_agreement": self.k15_k1h_atm_direction_agreement,
            "btc_spot_price": self.btc_spot_price,
            "btc_spot_twap_60s": self.btc_spot_twap_60s,
            "btc_spot_age_ms": self.btc_spot_age_ms,
            "btc_spot_is_fresh": (None if self.btc_spot_is_fresh is None else float(self.btc_spot_is_fresh)),
            "btc_spot_venues_fresh": self.btc_spot_venues_fresh,
            "btc_spot_venue_divergence_bps": self.btc_spot_venue_divergence_bps,
            "btc_vol_effective_sample_size": self.btc_vol_effective_sample_size,
            "btc_log_moneyness": self.btc_log_moneyness,
            "btc_log_moneyness_twap60": self.btc_log_moneyness_twap60,
            "btc_log_moneyness_per_minute": self.btc_log_moneyness_per_minute,
            "btc_spot_return_30s": self.btc_spot_return_30s,
            "btc_spot_return_120s": self.btc_spot_return_120s,
            "btc_spot_return_300s": self.btc_spot_return_300s,
            "btc_spot_return_900s": self.btc_spot_return_900s,
            "btc_spot_vol_120s": self.btc_spot_vol_120s,
            "btc_spot_vol_300s": self.btc_spot_vol_300s,
            "btc_spot_vol_900s": self.btc_spot_vol_900s,
            "btc_spot_vol_1800s": self.btc_spot_vol_1800s,
            "btc_spot_vol_ewma_hl300": self.btc_spot_vol_ewma_hl300,
            "btc_bs_yes_prob_120s": self.btc_bs_yes_prob_120s,
            "btc_bs_yes_prob_300s": self.btc_bs_yes_prob_300s,
            "btc_bs_yes_prob_900s": self.btc_bs_yes_prob_900s,
            "btc_bs_yes_prob_1800s": self.btc_bs_yes_prob_1800s,
            "btc_bs_yes_prob_ewma": self.btc_bs_yes_prob_ewma,
            "btc_kalshi_minus_bs_prob_120s": self.btc_kalshi_minus_bs_prob_120s,
            "btc_kalshi_minus_bs_prob_300s": self.btc_kalshi_minus_bs_prob_300s,
            "btc_kalshi_minus_bs_prob_900s": self.btc_kalshi_minus_bs_prob_900s,
            "btc_kalshi_minus_bs_prob_1800s": self.btc_kalshi_minus_bs_prob_1800s,
            "btc_kalshi_minus_bs_prob_ewma": self.btc_kalshi_minus_bs_prob_ewma,
            "btc_kalshi_minus_bs_logodds_120s": self.btc_kalshi_minus_bs_logodds_120s,
            "btc_kalshi_minus_bs_logodds_300s": self.btc_kalshi_minus_bs_logodds_300s,
            "btc_kalshi_minus_bs_logodds_900s": self.btc_kalshi_minus_bs_logodds_900s,
            "btc_kalshi_minus_bs_logodds_1800s": self.btc_kalshi_minus_bs_logodds_1800s,
            "btc_kalshi_minus_bs_logodds_ewma": self.btc_kalshi_minus_bs_logodds_ewma,
            "btc_kalshi_minus_bs_prob_120s_detrended": self.btc_kalshi_minus_bs_prob_120s_detrended,
            "btc_kalshi_minus_bs_prob_300s_detrended": self.btc_kalshi_minus_bs_prob_300s_detrended,
            "btc_kalshi_minus_bs_prob_900s_detrended": self.btc_kalshi_minus_bs_prob_900s_detrended,
            "btc_kalshi_minus_bs_prob_1800s_detrended": self.btc_kalshi_minus_bs_prob_1800s_detrended,
            "btc_kalshi_minus_bs_prob_ewma_detrended": self.btc_kalshi_minus_bs_prob_ewma_detrended,
            "btc_kalshi_implied_vol": self.btc_kalshi_implied_vol,
        }
        lookup.update(build_linear_v1_derived_feature_lookup(lookup))
        return lookup

    def feature_row(self, feature_names: tuple[str, ...] | None = None) -> np.ndarray | None:
        selected_feature_names = tuple(feature_names) if feature_names is not None else tuple(FEATURE_ORDER)
        lookup = self.feature_lookup()
        values: list[float] = []
        for feature_name in selected_feature_names:
            value = lookup.get(feature_name)
            if value is None:
                if feature_name in FEATURE_ORDER or feature_name in HOURLY_CONTEXT_FEATURE_ORDER or feature_name in SPOT_MODEL_FEATURE_SET:
                    return None
                value = 0.0
            values.append(float(value))
        return np.array([values], dtype=np.float32)


@dataclass(frozen=True)
class KalshiFeatureUpdate:
    ticker: str
    event_time: datetime
    last_yes_price_cents: int | None
    previous_yes_price_cents: int | None
    market_prob: float | None
    previous_market_prob: float | None
    close_time: datetime | None
    open_time: datetime | None
    is_open: bool
    trade_id: str | None
    count: int | None
    taker_side: str | None
    last_trade_time: datetime | None
    last_price_cents: int | None
    yes_bid_cents: int | None
    yes_ask_cents: int | None
    no_bid_cents: int | None
    no_ask_cents: int | None
    volume: int | None
    open_interest: int | None
    dollar_volume: int | None
    dollar_open_interest: int | None
    ticker_update_time: datetime | None
    trade_yes_prob: float | None
    quote_mid_prob: float | None
    quote_spread_cents: int | None
    buy_yes_price_cents: int | None
    buy_no_price_cents: int | None
    quote_age_seconds: float | None
    last_to_mid_gap: float | None
    z_implied: float | None
    tau_minutes: float | None
    price_momentum: float | None
    abs_price_momentum: float | None
    price_direction: float | None
    distance_from_mid: float | None
    last_trade_count: float | None
    last_trade_side_sign: float | None
    last_trade_signed_count: float | None
    time_since_last_trade_seconds: float | None
    minutes_since_market_open: float | None
    trade_count_30s: float | None
    contracts_sum_30s: float | None
    signed_contracts_sum_30s: float | None
    yes_taker_share_30s: float | None
    price_return_30s: float | None
    price_volatility_30s: float | None
    trade_count_120s: float | None
    contracts_sum_120s: float | None
    signed_contracts_sum_120s: float | None
    yes_taker_share_120s: float | None
    price_return_120s: float | None
    price_volatility_120s: float | None
    trade_count_300s: float | None
    contracts_sum_300s: float | None
    signed_contracts_sum_300s: float | None
    yes_taker_share_300s: float | None
    price_return_300s: float | None
    price_volatility_300s: float | None
    in_training_window: bool
    is_scoreable: bool
    kxbtcd_atm_ticker: str | None = None
    kxbtcd_atm_z_implied: float | None = None
    kxbtcd_atm_price_momentum: float | None = None
    kxbtcd_atm_abs_price_momentum: float | None = None
    kxbtcd_atm_distance_from_mid: float | None = None
    kxbtcd_atm_trade_count_300s: float | None = None
    kxbtcd_atm_signed_contracts_sum_300s: float | None = None
    kxbtcd_atm_price_return_300s: float | None = None
    kxbtcd_atm_price_volatility_300s: float | None = None
    kxbtcd_atm_yes_taker_share_300s: float | None = None
    k15_minus_k1h_atm_z: float | None = None
    k15_minus_k1h_atm_price_return_300s: float | None = None
    k15_k1h_atm_direction_agreement: float | None = None
    btc_spot_price: float | None = None
    btc_spot_twap_60s: float | None = None
    btc_spot_age_ms: float | None = None
    btc_spot_is_fresh: bool | None = None
    btc_spot_venues_fresh: float | None = None
    btc_spot_venue_divergence_bps: float | None = None
    btc_vol_effective_sample_size: float | None = None
    btc_spot_source: str | None = None
    btc_log_moneyness: float | None = None
    btc_log_moneyness_twap60: float | None = None
    btc_log_moneyness_per_minute: float | None = None
    btc_spot_return_30s: float | None = None
    btc_spot_return_120s: float | None = None
    btc_spot_return_300s: float | None = None
    btc_spot_return_900s: float | None = None
    btc_spot_vol_120s: float | None = None
    btc_spot_vol_300s: float | None = None
    btc_spot_vol_900s: float | None = None
    btc_spot_vol_1800s: float | None = None
    btc_spot_vol_ewma_hl300: float | None = None
    btc_bs_yes_prob_120s: float | None = None
    btc_bs_yes_prob_300s: float | None = None
    btc_bs_yes_prob_900s: float | None = None
    btc_bs_yes_prob_1800s: float | None = None
    btc_bs_yes_prob_ewma: float | None = None
    btc_kalshi_minus_bs_prob_120s: float | None = None
    btc_kalshi_minus_bs_prob_300s: float | None = None
    btc_kalshi_minus_bs_prob_900s: float | None = None
    btc_kalshi_minus_bs_prob_1800s: float | None = None
    btc_kalshi_minus_bs_prob_ewma: float | None = None
    btc_kalshi_minus_bs_logodds_120s: float | None = None
    btc_kalshi_minus_bs_logodds_300s: float | None = None
    btc_kalshi_minus_bs_logodds_900s: float | None = None
    btc_kalshi_minus_bs_logodds_1800s: float | None = None
    btc_kalshi_minus_bs_logodds_ewma: float | None = None
    btc_kalshi_minus_bs_prob_120s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_300s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_900s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_1800s_detrended: float | None = None
    btc_kalshi_minus_bs_prob_ewma_detrended: float | None = None
    btc_kalshi_implied_vol: float | None = None
    received_at: datetime | None = None
    event_id: str = ""
    raw_event_id: str | None = None
    source: str = "snapshot"

    def feature_values(self) -> tuple[float, ...] | None:
        values = (
            self.z_implied,
            self.tau_minutes,
            self.price_momentum,
            self.abs_price_momentum,
            self.price_direction,
            self.distance_from_mid,
            self.last_trade_count,
            self.last_trade_side_sign,
            self.last_trade_signed_count,
            self.time_since_last_trade_seconds,
            self.minutes_since_market_open,
            self.trade_count_30s,
            self.contracts_sum_30s,
            self.signed_contracts_sum_30s,
            self.yes_taker_share_30s,
            self.price_return_30s,
            self.price_volatility_30s,
            self.trade_count_120s,
            self.contracts_sum_120s,
            self.signed_contracts_sum_120s,
            self.yes_taker_share_120s,
            self.price_return_120s,
            self.price_volatility_120s,
            self.trade_count_300s,
            self.contracts_sum_300s,
            self.signed_contracts_sum_300s,
            self.yes_taker_share_300s,
            self.price_return_300s,
            self.price_volatility_300s,
        )
        if any(value is None for value in values):
            return None
        return values  # type: ignore[return-value]

    def feature_lookup(self) -> dict[str, float | None]:
        lookup = {
            "z_implied": self.z_implied,
            "tau_minutes": self.tau_minutes,
            "price_momentum": self.price_momentum,
            "abs_price_momentum": self.abs_price_momentum,
            "price_direction": self.price_direction,
            "distance_from_mid": self.distance_from_mid,
            "last_trade_count": self.last_trade_count,
            "last_trade_side_sign": self.last_trade_side_sign,
            "last_trade_signed_count": self.last_trade_signed_count,
            "time_since_last_trade_seconds": self.time_since_last_trade_seconds,
            "minutes_since_market_open": self.minutes_since_market_open,
            "trade_count_30s": self.trade_count_30s,
            "contracts_sum_30s": self.contracts_sum_30s,
            "signed_contracts_sum_30s": self.signed_contracts_sum_30s,
            "yes_taker_share_30s": self.yes_taker_share_30s,
            "price_return_30s": self.price_return_30s,
            "price_volatility_30s": self.price_volatility_30s,
            "trade_count_120s": self.trade_count_120s,
            "contracts_sum_120s": self.contracts_sum_120s,
            "signed_contracts_sum_120s": self.signed_contracts_sum_120s,
            "yes_taker_share_120s": self.yes_taker_share_120s,
            "price_return_120s": self.price_return_120s,
            "price_volatility_120s": self.price_volatility_120s,
            "trade_count_300s": self.trade_count_300s,
            "contracts_sum_300s": self.contracts_sum_300s,
            "signed_contracts_sum_300s": self.signed_contracts_sum_300s,
            "yes_taker_share_300s": self.yes_taker_share_300s,
            "price_return_300s": self.price_return_300s,
            "price_volatility_300s": self.price_volatility_300s,
            "kxbtcd_atm_z_implied": self.kxbtcd_atm_z_implied,
            "kxbtcd_atm_price_momentum": self.kxbtcd_atm_price_momentum,
            "kxbtcd_atm_abs_price_momentum": self.kxbtcd_atm_abs_price_momentum,
            "kxbtcd_atm_distance_from_mid": self.kxbtcd_atm_distance_from_mid,
            "kxbtcd_atm_trade_count_300s": self.kxbtcd_atm_trade_count_300s,
            "kxbtcd_atm_signed_contracts_sum_300s": self.kxbtcd_atm_signed_contracts_sum_300s,
            "kxbtcd_atm_price_return_300s": self.kxbtcd_atm_price_return_300s,
            "kxbtcd_atm_price_volatility_300s": self.kxbtcd_atm_price_volatility_300s,
            "kxbtcd_atm_yes_taker_share_300s": self.kxbtcd_atm_yes_taker_share_300s,
            "k15_minus_k1h_atm_z": self.k15_minus_k1h_atm_z,
            "k15_minus_k1h_atm_price_return_300s": self.k15_minus_k1h_atm_price_return_300s,
            "k15_k1h_atm_direction_agreement": self.k15_k1h_atm_direction_agreement,
            "btc_spot_price": self.btc_spot_price,
            "btc_spot_twap_60s": self.btc_spot_twap_60s,
            "btc_spot_age_ms": self.btc_spot_age_ms,
            "btc_spot_is_fresh": (None if self.btc_spot_is_fresh is None else float(self.btc_spot_is_fresh)),
            "btc_spot_venues_fresh": self.btc_spot_venues_fresh,
            "btc_spot_venue_divergence_bps": self.btc_spot_venue_divergence_bps,
            "btc_vol_effective_sample_size": self.btc_vol_effective_sample_size,
            "btc_log_moneyness": self.btc_log_moneyness,
            "btc_log_moneyness_twap60": self.btc_log_moneyness_twap60,
            "btc_log_moneyness_per_minute": self.btc_log_moneyness_per_minute,
            "btc_spot_return_30s": self.btc_spot_return_30s,
            "btc_spot_return_120s": self.btc_spot_return_120s,
            "btc_spot_return_300s": self.btc_spot_return_300s,
            "btc_spot_return_900s": self.btc_spot_return_900s,
            "btc_spot_vol_120s": self.btc_spot_vol_120s,
            "btc_spot_vol_300s": self.btc_spot_vol_300s,
            "btc_spot_vol_900s": self.btc_spot_vol_900s,
            "btc_spot_vol_1800s": self.btc_spot_vol_1800s,
            "btc_spot_vol_ewma_hl300": self.btc_spot_vol_ewma_hl300,
            "btc_bs_yes_prob_120s": self.btc_bs_yes_prob_120s,
            "btc_bs_yes_prob_300s": self.btc_bs_yes_prob_300s,
            "btc_bs_yes_prob_900s": self.btc_bs_yes_prob_900s,
            "btc_bs_yes_prob_1800s": self.btc_bs_yes_prob_1800s,
            "btc_bs_yes_prob_ewma": self.btc_bs_yes_prob_ewma,
            "btc_kalshi_minus_bs_prob_120s": self.btc_kalshi_minus_bs_prob_120s,
            "btc_kalshi_minus_bs_prob_300s": self.btc_kalshi_minus_bs_prob_300s,
            "btc_kalshi_minus_bs_prob_900s": self.btc_kalshi_minus_bs_prob_900s,
            "btc_kalshi_minus_bs_prob_1800s": self.btc_kalshi_minus_bs_prob_1800s,
            "btc_kalshi_minus_bs_prob_ewma": self.btc_kalshi_minus_bs_prob_ewma,
            "btc_kalshi_minus_bs_logodds_120s": self.btc_kalshi_minus_bs_logodds_120s,
            "btc_kalshi_minus_bs_logodds_300s": self.btc_kalshi_minus_bs_logodds_300s,
            "btc_kalshi_minus_bs_logodds_900s": self.btc_kalshi_minus_bs_logodds_900s,
            "btc_kalshi_minus_bs_logodds_1800s": self.btc_kalshi_minus_bs_logodds_1800s,
            "btc_kalshi_minus_bs_logodds_ewma": self.btc_kalshi_minus_bs_logodds_ewma,
            "btc_kalshi_minus_bs_prob_120s_detrended": self.btc_kalshi_minus_bs_prob_120s_detrended,
            "btc_kalshi_minus_bs_prob_300s_detrended": self.btc_kalshi_minus_bs_prob_300s_detrended,
            "btc_kalshi_minus_bs_prob_900s_detrended": self.btc_kalshi_minus_bs_prob_900s_detrended,
            "btc_kalshi_minus_bs_prob_1800s_detrended": self.btc_kalshi_minus_bs_prob_1800s_detrended,
            "btc_kalshi_minus_bs_prob_ewma_detrended": self.btc_kalshi_minus_bs_prob_ewma_detrended,
            "btc_kalshi_implied_vol": self.btc_kalshi_implied_vol,
        }
        lookup.update(build_linear_v1_derived_feature_lookup(lookup))
        return lookup

    def feature_row(self, feature_names: tuple[str, ...] | None = None) -> np.ndarray | None:
        selected_feature_names = tuple(feature_names) if feature_names is not None else tuple(FEATURE_ORDER)
        lookup = self.feature_lookup()
        values: list[float] = []
        for feature_name in selected_feature_names:
            value = lookup.get(feature_name)
            if value is None:
                if feature_name in FEATURE_ORDER or feature_name in HOURLY_CONTEXT_FEATURE_ORDER or feature_name in SPOT_MODEL_FEATURE_SET:
                    return None
                value = 0.0
            values.append(float(value))
        return np.array([values], dtype=np.float32)


@dataclass(frozen=True)
class _TradeObservation:
    event_time: datetime
    market_prob: float
    count: float
    side_sign: float


class KalshiTradeFeatureAccumulator:
    def __init__(self, config: KalshiFeatureEngineConfig | None = None):
        self.config = config or KalshiFeatureEngineConfig()
        self._max_window_seconds = max(self.config.rolling_window_seconds)
        self._observations: deque[_TradeObservation] = deque()
        self._last_trade_time: datetime | None = None
        self._last_trade_count: float = 0.0
        self._last_trade_side_sign: float = 0.0
        self._last_trade_signed_count: float = 0.0

    def build_feature_state_from_update(self, update: KalshiTickerUpdate) -> KalshiFeatureState:
        return self._build_feature_state(
            ticker=update.ticker,
            event_time=update.event_time,
            last_yes_price_cents=update.last_yes_price_cents,
            previous_yes_price_cents=update.previous_yes_price_cents,
            close_time=update.close_time,
            open_time=update.open_time,
            is_open=update.is_open,
            trade_id=update.trade_id,
            count=update.count,
            taker_side=update.taker_side,
            last_trade_time=update.last_trade_time,
            last_price_cents=update.last_price_cents,
            yes_bid_cents=update.yes_bid_cents,
            yes_ask_cents=update.yes_ask_cents,
            no_bid_cents=update.no_bid_cents,
            no_ask_cents=update.no_ask_cents,
            volume=update.volume,
            open_interest=update.open_interest,
            dollar_volume=update.dollar_volume,
            dollar_open_interest=update.dollar_open_interest,
            ticker_update_time=update.ticker_update_time,
            received_at=update.received_at,
            event_id=update.event_id,
            raw_event_id=update.raw_event_id,
            source=update.source,
        )

    def build_feature_state_from_state(
        self,
        state: KalshiTickerState,
        event_time: datetime,
    ) -> KalshiFeatureState:
        return self._build_feature_state(
            ticker=state.ticker,
            event_time=event_time,
            last_yes_price_cents=state.last_yes_price_cents,
            previous_yes_price_cents=state.previous_yes_price_cents,
            close_time=state.close_time,
            open_time=state.open_time,
            is_open=state.is_open,
            trade_id=state.trade_id,
            count=state.count,
            taker_side=state.taker_side,
            last_trade_time=state.last_trade_time,
            last_price_cents=state.last_price_cents,
            yes_bid_cents=state.yes_bid_cents,
            yes_ask_cents=state.yes_ask_cents,
            no_bid_cents=state.no_bid_cents,
            no_ask_cents=state.no_ask_cents,
            volume=state.volume,
            open_interest=state.open_interest,
            dollar_volume=state.dollar_volume,
            dollar_open_interest=state.dollar_open_interest,
            ticker_update_time=state.ticker_update_time,
            received_at=state.received_at,
            event_id=state.event_id,
            raw_event_id=state.raw_event_id,
            source="snapshot",
        )

    def _build_feature_state(
        self,
        *,
        ticker: str,
        event_time: datetime,
        last_yes_price_cents: int | None,
        previous_yes_price_cents: int | None,
        close_time: datetime | None,
        open_time: datetime | None,
        is_open: bool,
        trade_id: str | None,
        count: int | None,
        taker_side: str | None,
        last_trade_time: datetime | None,
        last_price_cents: int | None,
        yes_bid_cents: int | None,
        yes_ask_cents: int | None,
        no_bid_cents: int | None,
        no_ask_cents: int | None,
        volume: int | None,
        open_interest: int | None,
        dollar_volume: int | None,
        dollar_open_interest: int | None,
        ticker_update_time: datetime | None,
        received_at: datetime | None,
        event_id: str,
        raw_event_id: str | None,
        source: str,
    ) -> KalshiFeatureState:
        market_prob: float | None = None
        previous_market_prob: float | None = None
        trade_yes_prob: float | None = None
        quote_mid_prob: float | None = None
        quote_spread_cents: int | None = None
        buy_yes_price_cents: int | None = yes_ask_cents
        buy_no_price_cents: int | None = None
        quote_age_seconds: float | None = None
        last_to_mid_gap: float | None = None
        z_implied: float | None = None
        tau_minutes: float | None = None
        price_momentum: float | None = None
        abs_price_momentum: float | None = None
        price_direction: float | None = None
        distance_from_mid: float | None = None

        price_cents = last_yes_price_cents if last_yes_price_cents is not None else last_price_cents
        if price_cents is not None:
            trade_yes_prob = price_cents / 100.0
        if price_cents is not None:
            previous_prob_input = None
            if previous_yes_price_cents is not None:
                previous_prob_input = previous_yes_price_cents / 100.0
            (
                market_prob,
                previous_market_prob,
                z_implied,
                price_momentum,
                abs_price_momentum,
                price_direction,
                distance_from_mid,
            ) = build_feature_components(
                market_prob=price_cents / 100.0,
                previous_market_prob=previous_prob_input,
                config=self.config,
            )

        if yes_bid_cents is not None and yes_ask_cents is not None:
            quote_mid_prob = (yes_bid_cents + yes_ask_cents) / 200.0
            quote_spread_cents = yes_ask_cents - yes_bid_cents
        elif no_bid_cents is not None and no_ask_cents is not None:
            quote_mid_prob = (200.0 - no_bid_cents - no_ask_cents) / 200.0
            quote_spread_cents = no_ask_cents - no_bid_cents
        if buy_yes_price_cents is None and no_bid_cents is not None:
            buy_yes_price_cents = 100 - no_bid_cents
        if no_ask_cents is not None:
            buy_no_price_cents = no_ask_cents
        elif yes_bid_cents is not None:
            buy_no_price_cents = 100 - yes_bid_cents
        if ticker_update_time is not None:
            quote_age_seconds = max(0.0, (event_time - ticker_update_time).total_seconds())
        if trade_yes_prob is not None and quote_mid_prob is not None:
            last_to_mid_gap = trade_yes_prob - quote_mid_prob

        if close_time is not None:
            tau_minutes = (close_time - event_time).total_seconds() / 60.0

        time_since_last_trade_seconds = 0.0
        if self._last_trade_time is not None:
            time_since_last_trade_seconds = max(0.0, (event_time - self._last_trade_time).total_seconds())
        elif last_trade_time is not None:
            time_since_last_trade_seconds = max(0.0, (event_time - last_trade_time).total_seconds())
            self._last_trade_time = last_trade_time

        minutes_since_market_open = 0.0
        if open_time is not None:
            minutes_since_market_open = max(0.0, (event_time - open_time).total_seconds() / 60.0)

        if source == "trade" and market_prob is not None and count is not None:
            side_sign = _trade_side_sign(taker_side)
            self._last_trade_count = float(count)
            self._last_trade_side_sign = side_sign
            self._last_trade_signed_count = float(count) * side_sign
            self._observations.append(
                _TradeObservation(
                    event_time=event_time,
                    market_prob=market_prob,
                    count=float(count),
                    side_sign=side_sign,
                )
            )
            self._last_trade_time = event_time

        self._prune(event_time)
        rolling_values = self._rolling_values(event_time)
        in_training_window = tau_minutes is not None and 0.0 < tau_minutes <= self.config.tau_max_minutes
        is_scoreable = is_open and market_prob is not None and tau_minutes is not None and in_training_window

        return KalshiFeatureState(
            ticker=ticker,
            event_time=event_time,
            last_yes_price_cents=last_yes_price_cents,
            previous_yes_price_cents=previous_yes_price_cents,
            market_prob=market_prob,
            previous_market_prob=previous_market_prob,
            close_time=close_time,
            open_time=open_time,
            is_open=is_open,
            trade_id=trade_id,
            count=count,
            taker_side=taker_side,
            last_trade_time=self._last_trade_time,
            last_price_cents=last_price_cents,
            yes_bid_cents=yes_bid_cents,
            yes_ask_cents=yes_ask_cents,
            no_bid_cents=no_bid_cents,
            no_ask_cents=no_ask_cents,
            volume=volume,
            open_interest=open_interest,
            dollar_volume=dollar_volume,
            dollar_open_interest=dollar_open_interest,
            ticker_update_time=ticker_update_time,
            trade_yes_prob=trade_yes_prob,
            quote_mid_prob=quote_mid_prob,
            quote_spread_cents=quote_spread_cents,
            buy_yes_price_cents=buy_yes_price_cents,
            buy_no_price_cents=buy_no_price_cents,
            quote_age_seconds=quote_age_seconds,
            last_to_mid_gap=last_to_mid_gap,
            z_implied=z_implied,
            tau_minutes=tau_minutes,
            price_momentum=price_momentum,
            abs_price_momentum=abs_price_momentum,
            price_direction=price_direction,
            distance_from_mid=distance_from_mid,
            last_trade_count=self._last_trade_count,
            last_trade_side_sign=self._last_trade_side_sign,
            last_trade_signed_count=self._last_trade_signed_count,
            time_since_last_trade_seconds=time_since_last_trade_seconds,
            minutes_since_market_open=minutes_since_market_open,
            trade_count_30s=rolling_values["trade_count_30s"],
            contracts_sum_30s=rolling_values["contracts_sum_30s"],
            signed_contracts_sum_30s=rolling_values["signed_contracts_sum_30s"],
            yes_taker_share_30s=rolling_values["yes_taker_share_30s"],
            price_return_30s=rolling_values["price_return_30s"],
            price_volatility_30s=rolling_values["price_volatility_30s"],
            trade_count_120s=rolling_values["trade_count_120s"],
            contracts_sum_120s=rolling_values["contracts_sum_120s"],
            signed_contracts_sum_120s=rolling_values["signed_contracts_sum_120s"],
            yes_taker_share_120s=rolling_values["yes_taker_share_120s"],
            price_return_120s=rolling_values["price_return_120s"],
            price_volatility_120s=rolling_values["price_volatility_120s"],
            trade_count_300s=rolling_values["trade_count_300s"],
            contracts_sum_300s=rolling_values["contracts_sum_300s"],
            signed_contracts_sum_300s=rolling_values["signed_contracts_sum_300s"],
            yes_taker_share_300s=rolling_values["yes_taker_share_300s"],
            price_return_300s=rolling_values["price_return_300s"],
            price_volatility_300s=rolling_values["price_volatility_300s"],
            in_training_window=in_training_window,
            is_scoreable=is_scoreable,
            received_at=received_at,
            event_id=event_id,
            raw_event_id=raw_event_id,
            source=source,
        )

    def _prune(self, event_time: datetime) -> None:
        cutoff = event_time - timedelta(seconds=self._max_window_seconds)
        while self._observations and self._observations[0].event_time < cutoff:
            self._observations.popleft()

    def _rolling_values(self, event_time: datetime) -> dict[str, float]:
        values: dict[str, float] = {}
        observations = list(self._observations)
        for window_seconds in self.config.rolling_window_seconds:
            cutoff = event_time - timedelta(seconds=window_seconds)
            window = [obs for obs in observations if obs.event_time >= cutoff]
            trade_count = float(len(window))
            contracts_sum = float(sum(obs.count for obs in window))
            signed_contracts_sum = float(sum(obs.count * obs.side_sign for obs in window))
            yes_contracts_sum = float(sum(obs.count for obs in window if obs.side_sign > 0))
            if contracts_sum <= 0:
                yes_taker_share = 0.0
            else:
                yes_taker_share = yes_contracts_sum / contracts_sum
            if not window:
                price_return = 0.0
                price_volatility = 0.0
            else:
                price_return = float(window[-1].market_prob - window[0].market_prob)
                if len(window) <= 1:
                    price_volatility = 0.0
                else:
                    price_deltas = np.diff(np.asarray([obs.market_prob for obs in window], dtype=np.float64))
                    price_volatility = float(np.std(price_deltas))
            suffix = f"{window_seconds}s"
            values[f"trade_count_{suffix}"] = trade_count
            values[f"contracts_sum_{suffix}"] = contracts_sum
            values[f"signed_contracts_sum_{suffix}"] = signed_contracts_sum
            values[f"yes_taker_share_{suffix}"] = yes_taker_share
            values[f"price_return_{suffix}"] = price_return
            values[f"price_volatility_{suffix}"] = price_volatility
        return values


def _trade_side_sign(taker_side: str | None) -> float:
    if taker_side == "yes":
        return 1.0
    if taker_side == "no":
        return -1.0
    return 0.0


def feature_state_from_ticker_update(
    update: KalshiTickerUpdate,
    config: KalshiFeatureEngineConfig | None = None,
) -> KalshiFeatureState:
    accumulator = KalshiTradeFeatureAccumulator(config=config)
    return accumulator.build_feature_state_from_update(update)


def feature_state_from_ticker_state(
    state: KalshiTickerState,
    event_time: datetime,
    config: KalshiFeatureEngineConfig | None = None,
) -> KalshiFeatureState:
    accumulator = KalshiTradeFeatureAccumulator(config=config)
    return accumulator.build_feature_state_from_state(state, event_time=event_time)


def feature_update_from_state(state: KalshiFeatureState) -> KalshiFeatureUpdate:
    return KalshiFeatureUpdate(
        ticker=state.ticker,
        event_time=state.event_time,
        last_yes_price_cents=state.last_yes_price_cents,
        previous_yes_price_cents=state.previous_yes_price_cents,
        market_prob=state.market_prob,
        previous_market_prob=state.previous_market_prob,
        close_time=state.close_time,
        open_time=state.open_time,
        is_open=state.is_open,
        trade_id=state.trade_id,
        count=state.count,
        taker_side=state.taker_side,
        last_trade_time=state.last_trade_time,
        last_price_cents=state.last_price_cents,
        yes_bid_cents=state.yes_bid_cents,
        yes_ask_cents=state.yes_ask_cents,
        no_bid_cents=state.no_bid_cents,
        no_ask_cents=state.no_ask_cents,
        volume=state.volume,
        open_interest=state.open_interest,
        dollar_volume=state.dollar_volume,
        dollar_open_interest=state.dollar_open_interest,
        ticker_update_time=state.ticker_update_time,
        trade_yes_prob=state.trade_yes_prob,
        quote_mid_prob=state.quote_mid_prob,
        quote_spread_cents=state.quote_spread_cents,
        buy_yes_price_cents=state.buy_yes_price_cents,
        buy_no_price_cents=state.buy_no_price_cents,
        quote_age_seconds=state.quote_age_seconds,
        last_to_mid_gap=state.last_to_mid_gap,
        z_implied=state.z_implied,
        tau_minutes=state.tau_minutes,
        price_momentum=state.price_momentum,
        abs_price_momentum=state.abs_price_momentum,
        price_direction=state.price_direction,
        distance_from_mid=state.distance_from_mid,
        last_trade_count=state.last_trade_count,
        last_trade_side_sign=state.last_trade_side_sign,
        last_trade_signed_count=state.last_trade_signed_count,
        time_since_last_trade_seconds=state.time_since_last_trade_seconds,
        minutes_since_market_open=state.minutes_since_market_open,
        trade_count_30s=state.trade_count_30s,
        contracts_sum_30s=state.contracts_sum_30s,
        signed_contracts_sum_30s=state.signed_contracts_sum_30s,
        yes_taker_share_30s=state.yes_taker_share_30s,
        price_return_30s=state.price_return_30s,
        price_volatility_30s=state.price_volatility_30s,
        trade_count_120s=state.trade_count_120s,
        contracts_sum_120s=state.contracts_sum_120s,
        signed_contracts_sum_120s=state.signed_contracts_sum_120s,
        yes_taker_share_120s=state.yes_taker_share_120s,
        price_return_120s=state.price_return_120s,
        price_volatility_120s=state.price_volatility_120s,
        trade_count_300s=state.trade_count_300s,
        contracts_sum_300s=state.contracts_sum_300s,
        signed_contracts_sum_300s=state.signed_contracts_sum_300s,
        yes_taker_share_300s=state.yes_taker_share_300s,
        price_return_300s=state.price_return_300s,
        price_volatility_300s=state.price_volatility_300s,
        in_training_window=state.in_training_window,
        is_scoreable=state.is_scoreable,
        kxbtcd_atm_ticker=state.kxbtcd_atm_ticker,
        kxbtcd_atm_z_implied=state.kxbtcd_atm_z_implied,
        kxbtcd_atm_price_momentum=state.kxbtcd_atm_price_momentum,
        kxbtcd_atm_abs_price_momentum=state.kxbtcd_atm_abs_price_momentum,
        kxbtcd_atm_distance_from_mid=state.kxbtcd_atm_distance_from_mid,
        kxbtcd_atm_trade_count_300s=state.kxbtcd_atm_trade_count_300s,
        kxbtcd_atm_signed_contracts_sum_300s=state.kxbtcd_atm_signed_contracts_sum_300s,
        kxbtcd_atm_price_return_300s=state.kxbtcd_atm_price_return_300s,
        kxbtcd_atm_price_volatility_300s=state.kxbtcd_atm_price_volatility_300s,
        kxbtcd_atm_yes_taker_share_300s=state.kxbtcd_atm_yes_taker_share_300s,
        k15_minus_k1h_atm_z=state.k15_minus_k1h_atm_z,
        k15_minus_k1h_atm_price_return_300s=state.k15_minus_k1h_atm_price_return_300s,
        k15_k1h_atm_direction_agreement=state.k15_k1h_atm_direction_agreement,
        btc_spot_price=state.btc_spot_price,
        btc_spot_twap_60s=state.btc_spot_twap_60s,
        btc_spot_age_ms=state.btc_spot_age_ms,
        btc_spot_is_fresh=state.btc_spot_is_fresh,
        btc_spot_venues_fresh=state.btc_spot_venues_fresh,
        btc_spot_venue_divergence_bps=state.btc_spot_venue_divergence_bps,
        btc_vol_effective_sample_size=state.btc_vol_effective_sample_size,
        btc_spot_source=state.btc_spot_source,
        btc_log_moneyness=state.btc_log_moneyness,
        btc_log_moneyness_twap60=state.btc_log_moneyness_twap60,
        btc_log_moneyness_per_minute=state.btc_log_moneyness_per_minute,
        btc_spot_return_30s=state.btc_spot_return_30s,
        btc_spot_return_120s=state.btc_spot_return_120s,
        btc_spot_return_300s=state.btc_spot_return_300s,
        btc_spot_return_900s=state.btc_spot_return_900s,
        btc_spot_vol_120s=state.btc_spot_vol_120s,
        btc_spot_vol_300s=state.btc_spot_vol_300s,
        btc_spot_vol_900s=state.btc_spot_vol_900s,
        btc_spot_vol_1800s=state.btc_spot_vol_1800s,
        btc_spot_vol_ewma_hl300=state.btc_spot_vol_ewma_hl300,
        btc_bs_yes_prob_120s=state.btc_bs_yes_prob_120s,
        btc_bs_yes_prob_300s=state.btc_bs_yes_prob_300s,
        btc_bs_yes_prob_900s=state.btc_bs_yes_prob_900s,
        btc_bs_yes_prob_1800s=state.btc_bs_yes_prob_1800s,
        btc_bs_yes_prob_ewma=state.btc_bs_yes_prob_ewma,
        btc_kalshi_minus_bs_prob_120s=state.btc_kalshi_minus_bs_prob_120s,
        btc_kalshi_minus_bs_prob_300s=state.btc_kalshi_minus_bs_prob_300s,
        btc_kalshi_minus_bs_prob_900s=state.btc_kalshi_minus_bs_prob_900s,
        btc_kalshi_minus_bs_prob_1800s=state.btc_kalshi_minus_bs_prob_1800s,
        btc_kalshi_minus_bs_prob_ewma=state.btc_kalshi_minus_bs_prob_ewma,
        btc_kalshi_minus_bs_logodds_120s=state.btc_kalshi_minus_bs_logodds_120s,
        btc_kalshi_minus_bs_logodds_300s=state.btc_kalshi_minus_bs_logodds_300s,
        btc_kalshi_minus_bs_logodds_900s=state.btc_kalshi_minus_bs_logodds_900s,
        btc_kalshi_minus_bs_logodds_1800s=state.btc_kalshi_minus_bs_logodds_1800s,
        btc_kalshi_minus_bs_logodds_ewma=state.btc_kalshi_minus_bs_logodds_ewma,
        btc_kalshi_minus_bs_prob_120s_detrended=state.btc_kalshi_minus_bs_prob_120s_detrended,
        btc_kalshi_minus_bs_prob_300s_detrended=state.btc_kalshi_minus_bs_prob_300s_detrended,
        btc_kalshi_minus_bs_prob_900s_detrended=state.btc_kalshi_minus_bs_prob_900s_detrended,
        btc_kalshi_minus_bs_prob_1800s_detrended=state.btc_kalshi_minus_bs_prob_1800s_detrended,
        btc_kalshi_minus_bs_prob_ewma_detrended=state.btc_kalshi_minus_bs_prob_ewma_detrended,
        btc_kalshi_implied_vol=state.btc_kalshi_implied_vol,
        received_at=state.received_at,
        event_id=state.event_id,
        raw_event_id=state.raw_event_id,
        source=state.source,
    )
