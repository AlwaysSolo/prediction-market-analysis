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
from src.live.kalshi.data_download import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXISTING_ROOTS,
    KXBTC15MQuoteAwareDataDownloader,
)

load_dotenv()


def _build_downloader(args: argparse.Namespace) -> KXBTC15MQuoteAwareDataDownloader:
    existing_roots = tuple(Path(root) for root in (args.existing_root or ()))
    if not existing_roots:
        existing_roots = DEFAULT_EXISTING_ROOTS
    return KXBTC15MQuoteAwareDataDownloader(
        root_dir=Path(args.root_dir),
        series_ticker=args.series_ticker,
        existing_roots=existing_roots,
        requests_per_second=args.requests_per_second,
    )


def _print_summary(title: str, payload: dict) -> None:
    print(title)
    print(payload)


async def _capture_live_quotes(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment(args.environment)
    credentials = KalshiCredentials.from_env(environment)
    config = KalshiCollectorConfig(
        environment=environment,
        credentials=credentials,
        log_dir=Path(args.root_dir) / "raw" / "live_ticker",
        series_tickers=(args.series_ticker,),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
        log_stream_messages=True,
    )
    collector = KalshiMarketDataCollector(config)
    queue = collector.subscribe_queue()
    await collector.start()
    print(
        {
            "status": "collector_started",
            "environment": environment.value,
            "series_ticker": args.series_ticker,
            "log_dir": str(config.log_dir),
        }
    )
    try:
        while True:
            update = await queue.get()
            print(
                {
                    "ticker": update.ticker,
                    "last_yes_price_cents": update.last_yes_price_cents,
                    "yes_bid_cents": update.yes_bid_cents,
                    "yes_ask_cents": update.yes_ask_cents,
                    "source": update.source,
                }
            )
    finally:
        await collector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync quote-aware KXBTC15M training data from Kalshi.")
    parser.add_argument("--root-dir", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--series-ticker", default="KXBTC15M")
    parser.add_argument("--existing-root", action="append", default=[])
    parser.add_argument("--requests-per-second", type=float, default=6.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Build or refresh the ticker manifest.")
    discover_parser.set_defaults(command_name="discover")

    sync_parser = subparsers.add_parser("sync", help="Sync manifest, markets, trades, candles, and coverage.")
    sync_parser.add_argument("--ticker", action="append", default=[])
    sync_parser.add_argument("--skip-markets", action="store_true")
    sync_parser.add_argument("--skip-trades", action="store_true")
    sync_parser.add_argument("--skip-candles", action="store_true")
    sync_parser.add_argument("--skip-live-quote-materialization", action="store_true")
    sync_parser.add_argument("--max-workers", type=int, default=4)
    sync_parser.set_defaults(command_name="sync")

    live_parser = subparsers.add_parser("capture-live", help="Capture raw KXBTC15M live ticker and trade messages.")
    live_parser.add_argument("--environment", choices=["demo", "production"], default="production")
    live_parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    live_parser.set_defaults(command_name="capture-live")

    materialize_parser = subparsers.add_parser(
        "materialize-live-quotes",
        help="Convert raw live quote logs into curated parquet partitions.",
    )
    materialize_parser.set_defaults(command_name="materialize-live-quotes")

    coverage_parser = subparsers.add_parser("coverage", help="Generate a coverage report from curated outputs.")
    coverage_parser.set_defaults(command_name="coverage")

    args = parser.parse_args()

    if args.command_name == "capture-live":
        asyncio.run(_capture_live_quotes(args))
        return

    downloader = _build_downloader(args)

    if args.command_name == "discover":
        manifest = downloader.build_ticker_universe()
        _print_summary(
            "Ticker manifest refreshed.",
            {
                "series_ticker": args.series_ticker,
                "ticker_count": len(manifest),
                "manifest_path": str(downloader.manifest_path),
            },
        )
        return

    if args.command_name == "sync":
        coverage = downloader.sync(
            tickers=tuple(args.ticker),
            sync_markets=not args.skip_markets,
            sync_trades=not args.skip_trades,
            sync_candles=not args.skip_candles,
            materialize_live_quotes=not args.skip_live_quote_materialization,
            max_workers=args.max_workers,
        )
        _print_summary("Sync completed.", coverage["summary"])
        return

    if args.command_name == "materialize-live-quotes":
        rows = downloader.materialize_live_quote_logs()
        _print_summary(
            "Live quote logs materialized.",
            {
                "rows_written": rows,
                "curated_dir": str(downloader.curated_dir / "live_quotes"),
            },
        )
        return

    if args.command_name == "coverage":
        report = downloader.generate_coverage_report()
        _print_summary("Coverage report generated.", report["summary"])
        return


if __name__ == "__main__":
    main()
