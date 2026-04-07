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
    KalshiSignalRiskConfig,
    KalshiSignalRiskEngine,
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
    signal_config = KalshiSignalRiskConfig.from_env(environment)
    collector = KalshiMarketDataCollector(collector_config)
    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(
        feature_engine,
        KalshiLightGBMScorerConfig(model_file=Path(args.model_file)),
    )
    signal_engine = KalshiSignalRiskEngine(scorer, signal_config)
    queue = signal_engine.subscribe_queue()

    await collector.start()
    await feature_engine.start()
    await scorer.start()
    await signal_engine.start()
    print(f"Collector + feature engine + scorer + signal/risk engine started in {environment.value} mode")
    print(
        "Signal config:",
        f"edge={signal_config.edge_threshold_cents:.2f}c",
        f"tau={signal_config.min_tau_minutes:.1f}-{signal_config.max_tau_minutes:.1f}",
        f"contracts={signal_config.contracts_per_order}",
        f"stacking={signal_config.allow_stacking}",
        f"cash=${signal_config.starting_cash_dollars:.2f}",
        f"band={signal_config.price_band_min_cents}-{signal_config.price_band_max_cents}c",
    )
    try:
        while True:
            update = await queue.get()
            print(
                update.ticker,
                f"tau={update.tau_minutes:.2f}",
                f"side={update.side}",
                f"raw_edge={update.raw_model_edge:+.6f}" if update.raw_model_edge is not None else "raw_edge=?",
                f"post_cost_edge={update.post_cost_edge:+.6f}" if update.post_cost_edge is not None else "post_cost_edge=?",
                f"price={update.reference_price_cents}c" if update.reference_price_cents is not None else "price=?",
                (
                    f"max_price={update.max_acceptable_entry_price_cents}c"
                    if update.max_acceptable_entry_price_cents is not None
                    else "max_price=?"
                ),
                "APPROVED" if update.approved else f"BLOCKED:{update.block_reason}",
            )
    finally:
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()
        await collector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Kalshi signal/risk engine on top of the scorer.")
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
