from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGIME_DOWNTREND = "downtrend"
REGIME_NEUTRAL = "neutral"
REGIME_UPTREND = "uptrend"
REGIME_UNKNOWN = "unknown"

PRICE_MOMENTUM_BEARISH_THRESHOLD = -0.02
SIGNED_FLOW_BEARISH_THRESHOLD = -5.0
YES_SHARE_BEARISH_THRESHOLD = 0.40

PRICE_MOMENTUM_BULLISH_THRESHOLD = 0.02
SIGNED_FLOW_BULLISH_THRESHOLD = 5.0
YES_SHARE_BULLISH_THRESHOLD = 0.60

STRUCTURAL_CONTEXT_PRICE_RETURN_BEARISH_THRESHOLD = -0.01
STRUCTURAL_CONTEXT_SIGNED_FLOW_BEARISH_THRESHOLD = -8.0
STRUCTURAL_TARGET_ALIGNMENT_BEARISH_THRESHOLD = -0.005

STRUCTURAL_CONTEXT_PRICE_RETURN_BULLISH_THRESHOLD = 0.01
STRUCTURAL_CONTEXT_SIGNED_FLOW_BULLISH_THRESHOLD = 8.0
STRUCTURAL_TARGET_ALIGNMENT_BULLISH_THRESHOLD = 0.005


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class KalshiRegimeEvaluation:
    regime_label: str
    bearish_vote_count: int
    bullish_vote_count: int
    price_momentum_bearish: bool
    signed_flow_bearish: bool
    yes_share_bearish: bool
    price_momentum_bullish: bool
    signed_flow_bullish: bool
    yes_share_bullish: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime_label": self.regime_label,
            "bearish_vote_count": self.bearish_vote_count,
            "bullish_vote_count": self.bullish_vote_count,
            "regime_price_momentum_bearish": self.price_momentum_bearish,
            "regime_signed_flow_bearish": self.signed_flow_bearish,
            "regime_yes_share_bearish": self.yes_share_bearish,
            "regime_price_momentum_bullish": self.price_momentum_bullish,
            "regime_signed_flow_bullish": self.signed_flow_bullish,
            "regime_yes_share_bullish": self.yes_share_bullish,
        }


def evaluate_kxbtc15m_regime(
    *,
    price_momentum: float | None,
    signed_contracts_sum_300s: float | None,
    yes_taker_share_300s: float | None,
) -> KalshiRegimeEvaluation:
    normalized_price_momentum = _safe_float(price_momentum)
    normalized_signed_flow = _safe_float(signed_contracts_sum_300s)
    normalized_yes_share = _safe_float(yes_taker_share_300s)

    price_momentum_bearish = (
        normalized_price_momentum is not None
        and normalized_price_momentum <= PRICE_MOMENTUM_BEARISH_THRESHOLD
    )
    signed_flow_bearish = (
        normalized_signed_flow is not None
        and normalized_signed_flow <= SIGNED_FLOW_BEARISH_THRESHOLD
    )
    yes_share_bearish = (
        normalized_yes_share is not None
        and normalized_yes_share <= YES_SHARE_BEARISH_THRESHOLD
    )

    price_momentum_bullish = (
        normalized_price_momentum is not None
        and normalized_price_momentum >= PRICE_MOMENTUM_BULLISH_THRESHOLD
    )
    signed_flow_bullish = (
        normalized_signed_flow is not None
        and normalized_signed_flow >= SIGNED_FLOW_BULLISH_THRESHOLD
    )
    yes_share_bullish = (
        normalized_yes_share is not None
        and normalized_yes_share >= YES_SHARE_BULLISH_THRESHOLD
    )

    bearish_vote_count = int(price_momentum_bearish) + int(signed_flow_bearish) + int(yes_share_bearish)
    bullish_vote_count = int(price_momentum_bullish) + int(signed_flow_bullish) + int(yes_share_bullish)

    regime_label = REGIME_NEUTRAL
    if bearish_vote_count >= 2:
        regime_label = REGIME_DOWNTREND
    elif bullish_vote_count >= 2:
        regime_label = REGIME_UPTREND

    return KalshiRegimeEvaluation(
        regime_label=regime_label,
        bearish_vote_count=bearish_vote_count,
        bullish_vote_count=bullish_vote_count,
        price_momentum_bearish=price_momentum_bearish,
        signed_flow_bearish=signed_flow_bearish,
        yes_share_bearish=yes_share_bearish,
        price_momentum_bullish=price_momentum_bullish,
        signed_flow_bullish=signed_flow_bullish,
        yes_share_bullish=yes_share_bullish,
    )


def evaluate_kxbtc15m_regime_for_state(state: Any) -> KalshiRegimeEvaluation:
    return evaluate_kxbtc15m_regime(
        price_momentum=getattr(state, "price_momentum", None),
        signed_contracts_sum_300s=getattr(state, "signed_contracts_sum_300s", None),
        yes_taker_share_300s=getattr(state, "yes_taker_share_300s", None),
    )


def evaluate_kxbtc15m_structural_regime(
    *,
    target_price_return_300s: float | None,
    context_price_return_300s: float | None,
    context_signed_contracts_sum_300s: float | None,
    direction_agreement: float | None,
) -> KalshiRegimeEvaluation:
    normalized_target_return = _safe_float(target_price_return_300s)
    normalized_context_return = _safe_float(context_price_return_300s)
    normalized_context_flow = _safe_float(context_signed_contracts_sum_300s)
    normalized_direction_agreement = _safe_float(direction_agreement)

    if (
        normalized_context_return is None
        and normalized_context_flow is None
        and normalized_direction_agreement is None
    ):
        return KalshiRegimeEvaluation(
            regime_label=REGIME_UNKNOWN,
            bearish_vote_count=0,
            bullish_vote_count=0,
            price_momentum_bearish=False,
            signed_flow_bearish=False,
            yes_share_bearish=False,
            price_momentum_bullish=False,
            signed_flow_bullish=False,
            yes_share_bullish=False,
        )

    context_return_bearish = (
        normalized_context_return is not None
        and normalized_context_return <= STRUCTURAL_CONTEXT_PRICE_RETURN_BEARISH_THRESHOLD
    )
    context_flow_bearish = (
        normalized_context_flow is not None
        and normalized_context_flow <= STRUCTURAL_CONTEXT_SIGNED_FLOW_BEARISH_THRESHOLD
    )
    aligned_bearish = (
        normalized_direction_agreement is not None
        and normalized_direction_agreement > 0.0
        and normalized_target_return is not None
        and normalized_target_return <= STRUCTURAL_TARGET_ALIGNMENT_BEARISH_THRESHOLD
        and normalized_context_return is not None
        and normalized_context_return < 0.0
    )

    context_return_bullish = (
        normalized_context_return is not None
        and normalized_context_return >= STRUCTURAL_CONTEXT_PRICE_RETURN_BULLISH_THRESHOLD
    )
    context_flow_bullish = (
        normalized_context_flow is not None
        and normalized_context_flow >= STRUCTURAL_CONTEXT_SIGNED_FLOW_BULLISH_THRESHOLD
    )
    aligned_bullish = (
        normalized_direction_agreement is not None
        and normalized_direction_agreement > 0.0
        and normalized_target_return is not None
        and normalized_target_return >= STRUCTURAL_TARGET_ALIGNMENT_BULLISH_THRESHOLD
        and normalized_context_return is not None
        and normalized_context_return > 0.0
    )

    bearish_vote_count = int(context_return_bearish) + int(context_flow_bearish) + int(aligned_bearish)
    bullish_vote_count = int(context_return_bullish) + int(context_flow_bullish) + int(aligned_bullish)

    regime_label = REGIME_NEUTRAL
    if bearish_vote_count >= 2:
        regime_label = REGIME_DOWNTREND
    elif bullish_vote_count >= 2:
        regime_label = REGIME_UPTREND

    return KalshiRegimeEvaluation(
        regime_label=regime_label,
        bearish_vote_count=bearish_vote_count,
        bullish_vote_count=bullish_vote_count,
        price_momentum_bearish=context_return_bearish,
        signed_flow_bearish=context_flow_bearish,
        yes_share_bearish=aligned_bearish,
        price_momentum_bullish=context_return_bullish,
        signed_flow_bullish=context_flow_bullish,
        yes_share_bullish=aligned_bullish,
    )


def evaluate_kxbtc15m_structural_regime_for_state(state: Any) -> KalshiRegimeEvaluation:
    return evaluate_kxbtc15m_structural_regime(
        target_price_return_300s=getattr(state, "price_return_300s", None),
        context_price_return_300s=getattr(state, "kxbtcd_atm_price_return_300s", None),
        context_signed_contracts_sum_300s=getattr(state, "kxbtcd_atm_signed_contracts_sum_300s", None),
        direction_agreement=getattr(state, "k15_k1h_atm_direction_agreement", None),
    )
