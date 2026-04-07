from __future__ import annotations

from src.live.kalshi.regime import evaluate_kxbtc15m_regime


def test_regime_classifier_marks_downtrend_on_two_bearish_votes() -> None:
    regime = evaluate_kxbtc15m_regime(
        price_momentum=-0.03,
        signed_contracts_sum_300s=-6.0,
        yes_taker_share_300s=0.55,
    )

    assert regime.regime_label == "downtrend"
    assert regime.bearish_vote_count == 2
    assert regime.bullish_vote_count == 0


def test_regime_classifier_marks_uptrend_on_two_bullish_votes() -> None:
    regime = evaluate_kxbtc15m_regime(
        price_momentum=0.03,
        signed_contracts_sum_300s=6.0,
        yes_taker_share_300s=0.45,
    )

    assert regime.regime_label == "uptrend"
    assert regime.bullish_vote_count == 2
    assert regime.bearish_vote_count == 0


def test_regime_classifier_marks_neutral_when_votes_are_mixed() -> None:
    regime = evaluate_kxbtc15m_regime(
        price_momentum=-0.03,
        signed_contracts_sum_300s=6.0,
        yes_taker_share_300s=0.50,
    )

    assert regime.regime_label == "neutral"
    assert regime.bearish_vote_count == 1
    assert regime.bullish_vote_count == 1


def test_regime_classifier_honors_exact_threshold_boundaries() -> None:
    regime = evaluate_kxbtc15m_regime(
        price_momentum=-0.02,
        signed_contracts_sum_300s=-5.0,
        yes_taker_share_300s=0.40,
    )

    assert regime.regime_label == "downtrend"
    assert regime.price_momentum_bearish is True
    assert regime.signed_flow_bearish is True
    assert regime.yes_share_bearish is True
