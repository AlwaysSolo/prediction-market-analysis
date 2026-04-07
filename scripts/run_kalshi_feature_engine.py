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
    FEATURE_ORDER,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureStateEngine,
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
    engine = KalshiFeatureStateEngine(collector)
    queue = engine.subscribe_queue()

    await collector.start()
    await engine.start()
    print(f"Collector + feature engine started in {environment.value} mode")
    print(f"Feature order: {FEATURE_ORDER}")
    try:
        while True:
            update = await queue.get()
            feature_values = update.feature_values()
            print(
                update.ticker,
                f"prob={update.market_prob:.4f}" if update.market_prob is not None else "prob=?",
                f"tau={update.tau_minutes:.2f}" if update.tau_minutes is not None else "tau=?",
                f"momentum={update.price_momentum:+.4f}" if update.price_momentum is not None else "momentum=?",
                f"features={list(feature_values) if feature_values is not None else None}",
            )
    finally:
        await engine.stop()
        await collector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Kalshi feature engine on top of the market collector.")
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument("--series", action="append", default=[], help="Series ticker prefix filter")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-dir", default="output/live/kalshi/raw")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
