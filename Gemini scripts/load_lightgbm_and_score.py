from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.features import FEATURE_ORDER, build_feature_row  # noqa: E402
from src.live.kalshi.scorer import (  # noqa: E402
    DEFAULT_LIGHTGBM_MODEL_FILE,
    KalshiLightGBMScorerConfig,
    load_lightgbm_model_artifact,
    predict_yes_probability,
)


def score_yes_probability(
    model_file: str | Path,
    market_prob: float,
    tau_minutes: float,
    previous_market_prob: float | None = None,
) -> float:
    config = KalshiLightGBMScorerConfig(model_file=Path(model_file))
    model = load_lightgbm_model_artifact(config)
    features = build_feature_row(
        market_prob=market_prob,
        tau_minutes=tau_minutes,
        previous_market_prob=previous_market_prob,
    )
    return predict_yes_probability(model, features, config=config)


def parse_probability_args(
    yes_price_cents: float | None,
    market_prob: float | None,
    previous_yes_price_cents: float | None,
    previous_market_prob: float | None,
) -> tuple[float, float | None]:
    if yes_price_cents is None and market_prob is None:
        raise ValueError("Pass either --yes-price-cents or --market-prob.")
    if yes_price_cents is not None and market_prob is not None:
        raise ValueError("Pass only one of --yes-price-cents or --market-prob.")
    if previous_yes_price_cents is not None and previous_market_prob is not None:
        raise ValueError("Pass only one of --previous-yes-price-cents or --previous-market-prob.")

    current_prob = yes_price_cents / 100.0 if yes_price_cents is not None else market_prob
    prev_prob = (
        previous_yes_price_cents / 100.0
        if previous_yes_price_cents is not None
        else previous_market_prob
    )
    return float(current_prob), None if prev_prob is None else float(prev_prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the trained LightGBM model and score one market state.")
    parser.add_argument(
        "--model-file",
        default=str(DEFAULT_LIGHTGBM_MODEL_FILE),
        help="Path to the saved LightGBM model.txt artifact",
    )
    parser.add_argument("--yes-price-cents", type=float, help="Current Kalshi YES price in cents")
    parser.add_argument("--market-prob", type=float, help="Current market probability in [0, 1]")
    parser.add_argument("--previous-yes-price-cents", type=float, help="Previous Kalshi YES price in cents")
    parser.add_argument("--previous-market-prob", type=float, help="Previous market probability in [0, 1]")
    parser.add_argument("--tau-minutes", type=float, required=True, help="Minutes remaining until expiry")
    args = parser.parse_args()

    market_prob, previous_market_prob = parse_probability_args(
        yes_price_cents=args.yes_price_cents,
        market_prob=args.market_prob,
        previous_yes_price_cents=args.previous_yes_price_cents,
        previous_market_prob=args.previous_market_prob,
    )
    features = build_feature_row(
        market_prob=market_prob,
        tau_minutes=args.tau_minutes,
        previous_market_prob=previous_market_prob,
    )
    probability_yes = score_yes_probability(
        model_file=args.model_file,
        market_prob=market_prob,
        tau_minutes=args.tau_minutes,
        previous_market_prob=previous_market_prob,
    )

    print(f"Model file: {args.model_file}")
    print(f"Feature order: {FEATURE_ORDER}")
    print(f"Feature row: {features[0].tolist()}")
    print(f"Predicted YES probability: {probability_yes:.6f}")


if __name__ == "__main__":
    main()
