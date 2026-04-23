from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm

from src.live.data.btc_spot_feed import BTCSpotUpdate

SPOT_RETURN_WINDOWS_SECONDS = (30, 120, 300, 900)
SPOT_VOL_WINDOWS_SECONDS = (120, 300, 900, 1800)
SPOT_VOL_VARIANT_KEYS = ("120s", "300s", "900s", "1800s", "ewma")
SPOT_V1_FEATURE_SCHEMA = "spot_v1"
SPOT_V1_FEATURE_ORDER = (
    "btc_log_moneyness",
    "btc_log_moneyness_twap60",
    "btc_log_moneyness_per_minute",
    "btc_spot_return_30s",
    "btc_spot_return_120s",
    "btc_spot_return_300s",
    "btc_spot_return_900s",
    "btc_spot_vol_120s",
    "btc_spot_vol_300s",
    "btc_spot_vol_900s",
    "btc_spot_vol_1800s",
    "btc_spot_vol_ewma_hl300",
    "btc_bs_yes_prob_120s",
    "btc_bs_yes_prob_300s",
    "btc_bs_yes_prob_900s",
    "btc_bs_yes_prob_1800s",
    "btc_bs_yes_prob_ewma",
    "btc_kalshi_minus_bs_prob_120s",
    "btc_kalshi_minus_bs_prob_300s",
    "btc_kalshi_minus_bs_prob_900s",
    "btc_kalshi_minus_bs_prob_1800s",
    "btc_kalshi_minus_bs_prob_ewma",
    "btc_kalshi_minus_bs_logodds_120s",
    "btc_kalshi_minus_bs_logodds_300s",
    "btc_kalshi_minus_bs_logodds_900s",
    "btc_kalshi_minus_bs_logodds_1800s",
    "btc_kalshi_minus_bs_logodds_ewma",
    "btc_kalshi_minus_bs_prob_120s_detrended",
    "btc_kalshi_minus_bs_prob_300s_detrended",
    "btc_kalshi_minus_bs_prob_900s_detrended",
    "btc_kalshi_minus_bs_prob_1800s_detrended",
    "btc_kalshi_minus_bs_prob_ewma_detrended",
    "btc_kalshi_implied_vol",
)
SPOT_DIAGNOSTIC_FIELDS = (
    "btc_spot_price",
    "btc_spot_twap_60s",
    "btc_spot_age_ms",
    "btc_spot_is_fresh",
    "btc_spot_venues_fresh",
    "btc_spot_venue_divergence_bps",
    "btc_vol_effective_sample_size",
    "btc_spot_source",
)
SPOT_MODEL_FEATURE_SET = frozenset(SPOT_V1_FEATURE_ORDER)


def safe_log_odds(probability: float | None) -> float | None:
    if probability is None or not np.isfinite(probability):
        return None
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(math.log(clipped / (1.0 - clipped)))


def _tau_years(tau_minutes: float | None) -> float | None:
    if tau_minutes is None:
        return None
    return max(float(tau_minutes), 0.0) / (60.0 * 24.0 * 365.0)


def digital_yes_probability(
    *,
    spot_price: float | None,
    strike_price: float | None,
    tau_minutes: float | None,
    annualized_volatility: float | None,
) -> float | None:
    if (
        spot_price is None
        or strike_price is None
        or tau_minutes is None
        or annualized_volatility is None
        or spot_price <= 0.0
        or strike_price <= 0.0
        or annualized_volatility <= 0.0
    ):
        return None
    tau_years = _tau_years(tau_minutes)
    if tau_years is None or tau_years <= 0.0:
        return 1.0 if spot_price > strike_price else 0.0
    sigma_root_t = float(annualized_volatility) * math.sqrt(tau_years)
    if sigma_root_t <= 0.0:
        return 1.0 if spot_price > strike_price else 0.0
    d2 = (math.log(float(spot_price) / float(strike_price)) - 0.5 * float(annualized_volatility) ** 2 * tau_years) / sigma_root_t
    return float(norm.cdf(d2))


def implied_digital_volatility(
    *,
    target_probability: float | None,
    spot_price: float | None,
    strike_price: float | None,
    tau_minutes: float | None,
    max_iterations: int = 80,
) -> float | None:
    if (
        target_probability is None
        or spot_price is None
        or strike_price is None
        or tau_minutes is None
        or spot_price <= 0.0
        or strike_price <= 0.0
    ):
        return None
    probability = min(max(float(target_probability), 1e-6), 1.0 - 1e-6)
    low = 1e-4
    high = 5.0
    low_value = digital_yes_probability(
        spot_price=spot_price,
        strike_price=strike_price,
        tau_minutes=tau_minutes,
        annualized_volatility=low,
    )
    high_value = digital_yes_probability(
        spot_price=spot_price,
        strike_price=strike_price,
        tau_minutes=tau_minutes,
        annualized_volatility=high,
    )
    if low_value is None or high_value is None:
        return None
    if probability <= low_value:
        return low
    if probability >= high_value:
        return high
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        value = digital_yes_probability(
            spot_price=spot_price,
            strike_price=strike_price,
            tau_minutes=tau_minutes,
            annualized_volatility=mid,
        )
        if value is None:
            return None
        if abs(value - probability) <= 1e-6:
            return mid
        if value < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _ewma_alpha(delta_seconds: float, half_life_seconds: float) -> float:
    if half_life_seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(math.log(0.5) * max(delta_seconds, 0.0) / half_life_seconds)


@dataclass
class SpotFeatureTracker:
    detrend_halflife_seconds: float = 3600.0
    alert_window_seconds: float = 300.0
    _last_event_time: datetime | None = None
    _baseline_by_variant: dict[str, float] = field(default_factory=dict)
    _primary_detrended_history: deque[tuple[datetime, float]] = field(default_factory=deque)

    def enrich(
        self,
        *,
        event_time: datetime,
        quote_mid_prob: float | None,
        tau_minutes: float | None,
        strike_price: float | None,
        spot_update: BTCSpotUpdate | None,
    ) -> dict[str, float | None]:
        values: dict[str, float | None] = {
            field_name: None for field_name in [*SPOT_DIAGNOSTIC_FIELDS, *SPOT_V1_FEATURE_ORDER]
        }
        if spot_update is None:
            return values

        values.update(
            {
                "btc_spot_price": spot_update.btc_spot_price,
                "btc_spot_twap_60s": spot_update.btc_spot_twap_60s,
                "btc_spot_age_ms": spot_update.btc_spot_age_ms,
                "btc_spot_is_fresh": bool(spot_update.btc_spot_is_fresh),
                "btc_spot_venues_fresh": float(spot_update.btc_spot_venues_fresh),
                "btc_spot_venue_divergence_bps": spot_update.btc_spot_venue_divergence_bps,
                "btc_vol_effective_sample_size": spot_update.btc_vol_effective_sample_size,
                "btc_spot_source": spot_update.btc_spot_source,
                "btc_spot_return_30s": spot_update.btc_spot_return_30s,
                "btc_spot_return_120s": spot_update.btc_spot_return_120s,
                "btc_spot_return_300s": spot_update.btc_spot_return_300s,
                "btc_spot_return_900s": spot_update.btc_spot_return_900s,
                "btc_spot_vol_120s": spot_update.btc_spot_vol_120s,
                "btc_spot_vol_300s": spot_update.btc_spot_vol_300s,
                "btc_spot_vol_900s": spot_update.btc_spot_vol_900s,
                "btc_spot_vol_1800s": spot_update.btc_spot_vol_1800s,
                "btc_spot_vol_ewma_hl300": spot_update.btc_spot_vol_ewma_hl300,
            }
        )

        if spot_update.btc_spot_price is not None and strike_price is not None and strike_price > 0.0:
            values["btc_log_moneyness"] = float(math.log(spot_update.btc_spot_price / strike_price))
        if spot_update.btc_spot_twap_60s is not None and strike_price is not None and strike_price > 0.0:
            values["btc_log_moneyness_twap60"] = float(math.log(spot_update.btc_spot_twap_60s / strike_price))
        if values["btc_log_moneyness"] is not None and tau_minutes is not None:
            values["btc_log_moneyness_per_minute"] = float(values["btc_log_moneyness"]) / max(float(tau_minutes), 1.0 / 60.0)

        variant_to_vol = {
            "120s": spot_update.btc_spot_vol_120s,
            "300s": spot_update.btc_spot_vol_300s,
            "900s": spot_update.btc_spot_vol_900s,
            "1800s": spot_update.btc_spot_vol_1800s,
            "ewma": spot_update.btc_spot_vol_ewma_hl300,
        }
        delta_seconds = 0.0 if self._last_event_time is None else max(0.0, (event_time - self._last_event_time).total_seconds())
        alpha = _ewma_alpha(delta_seconds, self.detrend_halflife_seconds)
        kalshi_log_odds = safe_log_odds(quote_mid_prob)
        for variant, volatility in variant_to_vol.items():
            bs_probability = digital_yes_probability(
                spot_price=spot_update.btc_spot_price,
                strike_price=strike_price,
                tau_minutes=tau_minutes,
                annualized_volatility=volatility,
            )
            values[f"btc_bs_yes_prob_{variant}"] = bs_probability
            if quote_mid_prob is None or bs_probability is None:
                values[f"btc_kalshi_minus_bs_prob_{variant}"] = None
                values[f"btc_kalshi_minus_bs_logodds_{variant}"] = None
                values[f"btc_kalshi_minus_bs_prob_{variant}_detrended"] = None
                continue
            raw_gap = float(quote_mid_prob) - float(bs_probability)
            values[f"btc_kalshi_minus_bs_prob_{variant}"] = raw_gap
            bs_log_odds = safe_log_odds(bs_probability)
            values[f"btc_kalshi_minus_bs_logodds_{variant}"] = (
                None if kalshi_log_odds is None or bs_log_odds is None else kalshi_log_odds - bs_log_odds
            )
            baseline = self._baseline_by_variant.get(variant)
            if baseline is None:
                baseline = raw_gap
            else:
                baseline = baseline + alpha * (raw_gap - baseline)
            self._baseline_by_variant[variant] = baseline
            detrended = raw_gap - baseline
            values[f"btc_kalshi_minus_bs_prob_{variant}_detrended"] = detrended
            if variant == "ewma":
                self._primary_detrended_history.append((event_time, float(detrended)))
        self._last_event_time = event_time
        self._prune_history(event_time)
        values["btc_kalshi_implied_vol"] = implied_digital_volatility(
            target_probability=quote_mid_prob,
            spot_price=spot_update.btc_spot_price,
            strike_price=strike_price,
            tau_minutes=tau_minutes,
        )
        return values

    def primary_detrended_zscore(self, event_time: datetime) -> float | None:
        self._prune_history(event_time)
        if len(self._primary_detrended_history) < 2:
            return None
        values = np.asarray([value for _timestamp, value in self._primary_detrended_history], dtype=np.float64)
        std = float(values.std())
        if std <= 0.0:
            return None
        return float((values[-1] - values.mean()) / std)

    def _prune_history(self, event_time: datetime) -> None:
        cutoff = event_time - timedelta(seconds=self.alert_window_seconds)
        while self._primary_detrended_history and self._primary_detrended_history[0][0] < cutoff:
            self._primary_detrended_history.popleft()
