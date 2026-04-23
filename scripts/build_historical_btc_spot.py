from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.data.historical_btc_spot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_CACHE_ROOT,
    HistoricalBTCSpotBackfillConfig,
    build_historical_btc_spot_backfill,
    default_markets_path,
    infer_date_range_from_markets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical BTC external spot parquet from public Coinbase and Kraken trades."
    )
    parser.add_argument("--markets-path", help="Optional KXBTC15M markets parquet used to infer the default date range.")
    parser.add_argument("--start-date", help="UTC start date in YYYY-MM-DD format. Defaults to the earliest KXBTC15M market open date.")
    parser.add_argument("--end-date", help="UTC end date in YYYY-MM-DD format. Defaults to the latest KXBTC15M market close date.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-cache-root", default=str(DEFAULT_RAW_CACHE_ROOT))
    parser.add_argument("--environment", default="backfill")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing normalized parquet partitions.")
    parser.add_argument("--requests-per-second", type=float, default=3.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-latency-ms", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    markets_path = Path(args.markets_path).expanduser() if args.markets_path else default_markets_path()
    inferred_start, inferred_end = infer_date_range_from_markets(markets_path)
    start_date = inferred_start if args.start_date is None else date.fromisoformat(str(args.start_date))
    end_date = inferred_end if args.end_date is None else date.fromisoformat(str(args.end_date))

    config = HistoricalBTCSpotBackfillConfig(
        output_root=Path(args.output_root).expanduser(),
        raw_cache_root=Path(args.raw_cache_root).expanduser(),
        environment=str(args.environment),
        requests_per_second=float(args.requests_per_second),
        http_timeout_seconds=float(args.http_timeout_seconds),
        max_latency_ms=int(args.max_latency_ms),
        random_seed=int(args.random_seed),
    )
    results = build_historical_btc_spot_backfill(
        config=config,
        start_date=start_date,
        end_date=end_date,
        overwrite=bool(args.overwrite),
    )
    written_rows = sum(result.row_count for result in results)
    written_days = sum(1 for result in results if not result.skipped)
    skipped_days = sum(1 for result in results if result.skipped)
    print(
        {
            "markets_path": str(markets_path),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "output_root": str(config.output_root),
            "raw_cache_root": str(config.raw_cache_root),
            "written_days": written_days,
            "skipped_days": skipped_days,
            "written_rows": written_rows,
        }
    )


if __name__ == "__main__":
    main()
