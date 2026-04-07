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
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiMarketDataCollector,
)

load_dotenv()


async def _run(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment(args.environment)
    credentials = KalshiCredentials.from_env(environment)
    config = KalshiCollectorConfig(
        environment=environment,
        credentials=credentials,
        log_dir=Path(args.log_dir),
        series_tickers=tuple(args.series),
        market_tickers=tuple(args.ticker),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
    )
    collector = KalshiMarketDataCollector(config)
    queue = collector.subscribe_queue()
    await collector.start()
    print(f"Collector started in {environment.value} mode")
    try:
        while True:
            update = await queue.get()
            print(
                update.ticker,
                update.last_yes_price_cents,
                update.price_momentum,
                f"tau={update.tau_minutes:.2f}" if update.tau_minutes is not None else "tau=?",
            )
    finally:
        await collector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Kalshi market data collector.")
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument("--series", action="append", default=[], help="Series ticker prefix filter")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-dir", default="output/live/kalshi/raw")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
