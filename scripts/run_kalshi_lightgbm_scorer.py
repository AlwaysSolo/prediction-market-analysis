from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi import (  # noqa: E402
    DEFAULT_LIGHTGBM_MODEL_FILE,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureStateEngine,
    KalshiLightGBMScorer,
    KalshiLightGBMScorerConfig,
    KalshiMarketDataCollector,
)

load_dotenv()


async def _run(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment(args.environment)
    credentials = KalshiCredentials.from_env(environment)
    collector_config = KalshiCollectorConfig(
        environment=environment,
        credentials=credentials,
        log_dir=Path(args.log_dir),
        series_tickers=tuple(args.series),
        market_tickers=tuple(args.ticker),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
    )
    collector = KalshiMarketDataCollector(collector_config)
    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(
        feature_engine,
        KalshiLightGBMScorerConfig(model_file=Path(args.model_file)),
    )
    queue = scorer.subscribe_queue()

    await collector.start()
    await feature_engine.start()
    await scorer.start()
    print(f"Collector + feature engine + LightGBM scorer started in {environment.value} mode")
    print(f"Model file: {Path(args.model_file)}")
    try:
        while True:
            update = await queue.get()
            print(
                update.ticker,
                f"market_prob={update.market_prob:.4f}",
                f"pred_yes={update.predicted_yes_probability:.6f}",
                f"edge={update.model_edge:+.6f}",
                f"tau={update.tau_minutes:.2f}",
            )
    finally:
        await scorer.stop()
        await feature_engine.stop()
        await collector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Kalshi LightGBM scorer on top of the feature engine.")
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument("--series", action="append", default=[], help="Series ticker prefix filter")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-dir", default="output/live/kalshi/raw")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    parser.add_argument("--model-file", default=str(DEFAULT_LIGHTGBM_MODEL_FILE))
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
