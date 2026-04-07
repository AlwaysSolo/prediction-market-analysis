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
    KalshiExecutionConfig,
    KalshiExecutionEngine,
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
    print(f"Initializing live stack for {environment.value}...")
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
    signal_engine = KalshiSignalRiskEngine(scorer, KalshiSignalRiskConfig.from_env(environment))
    execution_engine = KalshiExecutionEngine(signal_engine, KalshiExecutionConfig.from_env(environment))
    signal_queue = signal_engine.subscribe_queue()
    queue = execution_engine.subscribe_queue()

    print("Starting collector and loading market metadata...")
    await collector.start()
    print("Collector ready")
    await feature_engine.start()
    print("Feature engine ready")
    await scorer.start()
    print("Scorer ready")
    await signal_engine.start()
    print("Signal engine ready")
    await execution_engine.start()
    print("Execution engine ready")

    print(f"Full live stack started in {environment.value} mode")
    print(
        "Execution config:",
        f"mode={execution_engine.config.mode.value}",
        f"live_enabled={execution_engine.config.enable_live_trading}",
        f"simulate_fills={execution_engine.config.simulate_immediate_fills}",
        f"invert_signal={signal_engine.config.invert_model_signal}",
        f"subaccount={execution_engine.config.subaccount}",
        f"reconcile={execution_engine.config.reconcile_interval_seconds:.1f}s",
    )

    try:
        while True:
            signal_task = asyncio.create_task(signal_queue.get())
            execution_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {signal_task, execution_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            for completed in done:
                update = completed.result()

                if completed is signal_task:
                    print(
                        "SIGNAL",
                        update.ticker,
                        f"tau={update.tau_minutes:.2f}",
                        f"side={update.side}",
                        f"raw_edge={update.raw_model_edge:+.6f}" if update.raw_model_edge is not None else "raw_edge=?",
                        (
                            f"post_cost_edge={update.post_cost_edge:+.6f}"
                            if update.post_cost_edge is not None
                            else "post_cost_edge=?"
                        ),
                        f"price={update.reference_price_cents}c" if update.reference_price_cents is not None else "price=?",
                        (
                            f"max_price={update.max_acceptable_entry_price_cents}c"
                            if update.max_acceptable_entry_price_cents is not None
                            else "max_price=?"
                        ),
                        "APPROVED" if update.approved else f"BLOCKED:{update.block_reason}",
                    )
                    continue

                portfolio_snapshot = execution_engine.get_portfolio_snapshot()
                portfolio_cash = (
                    f"${portfolio_snapshot.available_cash_dollars:.2f}"
                    if portfolio_snapshot is not None
                    else (f"${update.available_cash_dollars:.2f}" if update.available_cash_dollars is not None else "n/a")
                )
                print(
                    "EXECUTION",
                    update.ticker,
                    f"decision={update.decision_id}",
                    f"side={update.side}",
                    f"contracts={update.contracts}",
                    f"limit={update.limit_price_cents}c",
                    f"status={update.status}",
                    f"filled={update.filled_contracts}/{update.contracts}",
                    f"fill_price={update.fill_price_cents}c" if update.fill_price_cents is not None else "fill_price=?",
                    f"cash={portfolio_cash}",
                    (
                        f"pnl=${update.realized_pnl_dollars:+.2f}"
                        if update.realized_pnl_dollars is not None
                        else ""
                    ),
                    (
                        f"cum_pnl=${update.cumulative_realized_pnl_dollars:+.2f}"
                        if update.cumulative_realized_pnl_dollars is not None
                        else ""
                    ),
                    (
                        f"result={update.settlement_result}"
                        if update.settlement_result is not None
                        else ""
                    ),
                    f"msg={update.message}" if update.message else "",
                )
    finally:
        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()
        await collector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full live Kalshi stack through the execution engine.")
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument("--series", action="append", default=[], help="Series ticker prefix filter")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-dir", default="output/live/kalshi/raw")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    parser.add_argument("--model-file", default=str(DEFAULT_LIGHTGBM_MODEL_FILE))
    args = parser.parse_args()
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
